from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch_scatter import scatter, segment_coo, segment_csr

if TYPE_CHECKING:
    from torch.nn.modules.module import _IncompatibleKeys

def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[
        distances != 0
    ]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


def radius_graph_pbc(
    atom_pos,
    cell,
    natoms,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc=[True, True, True],
):
    device = atom_pos.device
    batch_size = len(natoms)

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    
    #max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]
    max_rep = [1]*3

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image
def one_way_neighbors(
    atom_pos,
    cell,
    natoms,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc=[True, True, True],
):
    device = atom_pos.device
    batch_size = len(natoms)

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    max_rep = [1]*3

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image

def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
            distance_sort[:, max_num_neighbors_threshold]
            + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute the number of neighbors
        num_neighbors_thresholded = num_neighbors.clamp(
            max=num_included_per_atom
        )

        num_neighbors_image = segment_csr(
            num_neighbors_thresholded, image_indptr
        )

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_included
    )
    # Remove "unused pairs" with infinisste distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def get_distances(positions, cell, cutoff_distance, skin=0.01, strain=np.zeros((3, 3))):
    """Get distances to atoms in a periodic unitcell.

    Parameters
    ----------

    positions: atomic positions. array-like (natoms, 3)
    cell: unit cell. array-like (3, 3)
    cutoff_distance: Maximum distance to get neighbor distances for. float
    skin: A tolerance for the cutoff_distance. float
    strain: array-like (3, 3)

    Returns
    -------

    distances : an array of dimension (atom_i, atom_j, distance) The shape is
    (natoms, natoms, nunitcells) where nunitcells is the total number of unit
    cells required to tile the space to be sure all neighbors will be found. The
    atoms that are outside the cutoff distance are zeroed.

    offsets

    """
    positions = np.array(positions)
    cell = np.array(cell)
    strain_tensor = np.eye(3) + strain
    cell = np.dot(strain_tensor, cell.T).T
    positions = np.dot(strain_tensor, positions.T).T

    inverse_cell = np.linalg.inv(cell)
    num_repeats = cutoff_distance * np.linalg.norm(inverse_cell, axis=0)

    fractional_coords = np.dot(positions, inverse_cell) % 1
    mins = np.min(np.floor(fractional_coords - num_repeats), axis=0)
    maxs = np.max(np.ceil(fractional_coords + num_repeats), axis=0)

    # Now we generate a set of cell offsets
    v0_range = np.arange(mins[0], maxs[0])
    v1_range = np.arange(mins[1], maxs[1])
    v2_range = np.arange(mins[2], maxs[2])

    xhat = np.array([1, 0, 0])
    yhat = np.array([0, 1, 0])
    zhat = np.array([0, 0, 1])

    v0_range = v0_range[:, None] * xhat[None, :]
    v1_range = v1_range[:, None] * yhat[None, :]
    v2_range = v2_range[:, None] * zhat[None, :]

    offsets = (
        v0_range[:, None, None] + v1_range[None, :, None] + v2_range[None, None, :]
    )

    offsets = np.int_(offsets.reshape(-1, 3))
    # Now we have a vector of unit cell offsets (offset_index, 3)
    # We convert that to cartesian coordinate offsets
    cart_offsets = np.dot(offsets, cell)

    # we need to offset each coord by each offset.
    # This array is (atom_index, offset, 3)
    shifted_cart_coords = positions[:, None] + cart_offsets[None, :]

    # Next, we subtract each position from the array of positions
    # (atom_i, atom_j, positionvector, 3)
    pv = shifted_cart_coords - positions[:, None, None]

    # This is the distance squared
    # (atom_i, atom_j, distance_ij)
    d2 = np.sum(pv**2, axis=3)

    # The gradient of sqrt is nan at r=0, so we do this round about way to
    # avoid that.
    zeros = np.equal(d2, 0.0)
    adjusted = np.where(zeros, np.ones_like(d2), d2)
    d = np.where(zeros, np.zeros_like(d2), np.sqrt(adjusted))

    distances = np.where(d < (cutoff_distance + skin), d, np.zeros_like(d))
    return distances, offsets


def get_neighbors(i, distances, offsets, oneway=False):
    """Get the indices and distances of neighbors to atom i.

    Parameters
    ----------

    i: int, index of the atom to get neighbors for.
    distances: the distances returned from `get_distances`.

    Returns
    -------
    indices: a list of indices for the neighbors corresponding to the index of the
    original atom list.
    offsets: a list of unit cell offsets to generate the position of the neighbor.
    """

    di = distances[i]

    within_cutoff = di > 0.0

    indices = np.arange(len(distances))

    inds = indices[np.where(within_cutoff)[0]]
    offs = offsets[np.where(within_cutoff)[1]]

    return inds, np.int_(offs)


def get_neighbors_oneway(
    positions, cell, cutoff_radius, skin=0.01, strain=np.zeros((3, 3))
):
    """A one-way neighbor list.

      Parameters
      ----------

      positions: atomic positions. array-like (natoms, 3)
      cell: unit cell. array-like (3, 3)
      cutoff_radius: Maximum radius to get neighbor distances for. float
      skin: A tolerance for the cutoff_radius. float
      strain: array-like (3, 3)

      Returns
      -------
      indices, offsets

    Note: this function works with autograd, but it has been very difficult to
    translate to Tensorflow. The challenge is because TF treats iteration
    differently, and does not like when you try to modify variables outside the
    iteration scope.

    """

    strain_tensor = np.eye(3) + strain
    cell = np.dot(strain_tensor, cell.T).T
    positions = np.dot(strain_tensor, positions.T).T
    inverse_cell = np.linalg.inv(cell)
    h = 1 / np.linalg.norm(inverse_cell, axis=0)
    N = np.floor(2 * cutoff_radius / h) + 1

    scaled = np.dot(positions, inverse_cell)
    scaled0 = np.dot(positions, inverse_cell) % 1.0

    # this is autograd compatible.
    offsets = np.int_((scaled0 - scaled).round())

    positions0 = positions + np.dot(offsets, cell)
    natoms = len(positions)
    indices = np.arange(natoms)

    v0_range = np.arange(0, N[0] + 1)
    v1_range = np.arange(-N[1], N[1] + 1)
    v2_range = np.arange(-N[2], N[2] + 1)

    xhat = np.array([1, 0, 0])
    yhat = np.array([0, 1, 0])
    zhat = np.array([0, 0, 1])

    v0_range = v0_range[:, None] * xhat[None, :]
    v1_range = v1_range[:, None] * yhat[None, :]
    v2_range = v2_range[:, None] * zhat[None, :]

    N = v0_range[:, None, None] + v1_range[None, :, None] + v2_range[None, None, :]

    N = N.reshape(-1, 3)

    neighbors = [np.empty(0, int) for a in range(natoms)]
    displacements = [np.empty((0, 3), int) for a in range(natoms)]

    def offset_mapfn(n):
        n1, n2, n3 = n
        if n1 == 0 and (n2 < 0 or (n2 == 0 and n3 < 0)):
            return
        displacement = np.dot((n1, n2, n3), cell)

        def atoms_mapfn(a):
            d = positions0 + displacement - positions0[a]
            i = indices[(d**2).sum(1) < (cutoff_radius**2 + skin)]
            if n1 == 0 and n2 == 0 and n3 == 0:
                i = i[i > a]
            neighbors[a] = np.concatenate((neighbors[a], i))
            disp = np.empty((len(i), 3), int)
            disp[:] = (n1, n2, n3)
            disp += offsets[i] - offsets[a]
            displacements[a] = np.concatenate((displacements[a], disp))

        list(map(atoms_mapfn, range(natoms)))

    list(map(offset_mapfn, N))

    return neighbors, displacements

# https://gist.github.com/Linux-cpp-lisp/692018c74b3906b63529e60619f5a207

@torch.jit.script
def torch_divmod(a, b) -> Tuple[torch.Tensor, torch.Tensor]:
    d = torch.div(a, b, rounding_mode="floor")
    m = a % b
    return d, m


@torch.jit.script
def primitive_neighbor_list(
    quantities: str,
    pbc: Tuple[bool, bool, bool],
    cell,
    positions,
    cutoff: float,
    self_interaction: bool = False,
    use_scaled_positions: bool = False,
    max_nbins: int = int(1e6),
):
    
    """
    Compute a neighbor list for an atomic configuration.
    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.
    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.
    Parameters:
    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are
            * 'i' : first atom index
            * 'j' : second atom index
            * 'd' : absolute distance
            * 'D' : distance vector
            * 'S' : shift vector (number of cell boundaries crossed by the bond
              between atom i and j). With the shift vector S, the
              distances D between atoms can be computed from:
              D = positions[j]-positions[i]+S.dot(cell)
    pbc: array_like
        3-tuple indicating giving periodic boundaries in the three Cartesian
        directions.
    cell: 3x3 matrix
        Unit cell vectors. Must be completed.
    positions: list of xyz-positions
        Atomic positions.  Anything that can be converted to an ndarray of
        shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2), ...]. If
        use_scaled_positions is set to true, this must be scaled positions.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:
            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.
    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    use_scaled_positions: bool
        If set to true, positions are expected to be scaled positions.
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.
    Returns:
    i, j, ... : array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a)-1, but the order of (i,j)
        pairs is not guaranteed.
    """

    # Naming conventions: Suffixes indicate the dimension of an array. The
    # following convention is used here:
    #     c: Cartesian index, can have values 0, 1, 2
    #     i: Global atom index, can have values 0..len(a)-1
    #     xyz: Bin index, three values identifying x-, y- and z-component of a
    #          spatial bin that is used to make neighbor search O(n)
    #     b: Linearized version of the 'xyz' bin index
    #     a: Bin-local atom index, i.e. index identifying an atom *within* a
    #        bin
    #     p: Pair index, can have value 0 or 1
    #     n: (Linear) neighbor index

    # Return empty neighbor list if no atoms are passed here
    if len(positions) == 0:
        assert False

    # Compute reciprocal lattice vectors.
    recip_cell = torch.linalg.pinv(cell).T
    b1_c, b2_c, b3_c = recip_cell[0], recip_cell[1], recip_cell[2]

    # Compute distances of cell faces.
    l1 = torch.linalg.norm(b1_c)
    l2 = torch.linalg.norm(b2_c)
    l3 = torch.linalg.norm(b3_c)
    pytorch_scalar_1 = torch.as_tensor(1.0)
    face_dist_c = torch.hstack(
        [
            1 / l1 if l1 > 0 else pytorch_scalar_1,
            1 / l2 if l2 > 0 else pytorch_scalar_1,
            1 / l3 if l3 > 0 else pytorch_scalar_1,
        ]
    )
    assert face_dist_c.shape == (3,)

    # we don't handle other fancier cutoffs
    max_cutoff: float = cutoff

    # We use a minimum bin size of 3 A
    bin_size = max(max_cutoff, 3)
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    nbins_c = torch.maximum(
        (face_dist_c / bin_size).to(dtype=torch.long), torch.ones(3, dtype=torch.long)
    )
    nbins = torch.prod(nbins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while nbins > max_nbins:
        nbins_c = torch.maximum(nbins_c // 2, torch.ones(3, dtype=torch.long))
        nbins = torch.prod(nbins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search = torch.ceil(bin_size * nbins_c / face_dist_c).to(dtype=torch.long)
    neigh_search_x, neigh_search_y, neigh_search_z = (
        neigh_search[0],
        neigh_search[1],
        neigh_search[2],
    )

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    pytorch_scalar_int_0 = torch.as_tensor(0, dtype=torch.long)
    neigh_search_x = (
        pytorch_scalar_int_0 if nbins_c[0] == 1 and not pbc[0] else neigh_search_x
    )
    neigh_search_y = (
        pytorch_scalar_int_0 if nbins_c[1] == 1 and not pbc[1] else neigh_search_y
    )
    neigh_search_z = (
        pytorch_scalar_int_0 if nbins_c[2] == 1 and not pbc[2] else neigh_search_z
    )

    # Sort atoms into bins.
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = torch.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = torch.linalg.solve(cell.T, positions.T).T
    bin_index_ic = torch.floor(scaled_positions_ic * nbins_c).to(dtype=torch.long)
    cell_shift_ic = torch.zeros_like(bin_index_ic)

    for c in range(3):
        if pbc[c]:
            # (Note: torch.divmod does not exist in older numpies)
            cell_shift_ic[:, c], bin_index_ic[:, c] = torch_divmod(
                bin_index_ic[:, c], nbins_c[c]
            )
        else:
            bin_index_ic[:, c] = torch.clip(bin_index_ic[:, c], 0, nbins_c[c] - 1)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = bin_index_ic[:, 0] + nbins_c[0] * (
        bin_index_ic[:, 1] + nbins_c[1] * bin_index_ic[:, 2]
    )

    # atom_i contains atom index in new sort order.
    atom_i = torch.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_natoms_per_bin = torch.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_natoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -torch.ones(nbins, max_natoms_per_bin.item(), dtype=torch.long)
    for i in range(int(max_natoms_per_bin.item())):
        # Create a mask array that identifies the first atom of each bin.
        mask = torch.cat(
            (torch.ones(1, dtype=torch.bool), bin_index_i[:-1] != bin_index_i[1:]),
            dim=0,
        )
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = torch.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_natoms_per_bin**2.
    # atom_pairs_pn_np = np.indices(
    #     (max_natoms_per_bin, max_natoms_per_bin), dtype=int
    # ).reshape(2, -1)
    atom_pairs_pn = torch.cartesian_prod(
        torch.arange(max_natoms_per_bin), torch.arange(max_natoms_per_bin)
    )
    atom_pairs_pn = atom_pairs_pn.T.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_natoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = torch.meshgrid(
        torch.arange(nbins_c[2]),
        torch.arange(nbins_c[1]),
        torch.arange(nbins_c[0]),
        indexing="ij",
    )
    # The memory layout of binx_xyz, biny_xyz, binz_xyz is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + nbins_c[0] * (biny_xyz + nbins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == torch.arange(torch.prod(nbins_c))).all()

    # First atoms in pair.
    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-int(neigh_search_z.item()), int(neigh_search_z.item()) + 1):
        for dy in range(-int(neigh_search_y.item()), int(neigh_search_y.item()) + 1):
            for dx in range(
                -int(neigh_search_x.item()), int(neigh_search_x.item()) + 1
            ):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = torch_divmod(binx_xyz + dx, nbins_c[0])
                shifty_xyz, neighbiny_xyz = torch_divmod(biny_xyz + dy, nbins_c[1])
                shiftz_xyz, neighbinz_xyz = torch_divmod(binz_xyz + dz, nbins_c[2])
                neighbin_b = (
                    neighbinx_xyz
                    + nbins_c[0] * (neighbiny_xyz + nbins_c[1] * neighbinz_xyz)
                ).ravel()

                # Second atom in pair.
                _secnd_at_neightuple_n = atoms_in_bin_ba[neighbin_b][
                    :, atom_pairs_pn[1]
                ]

                # Shift vectors.
                # TODO: was np.resize:
                # _cell_shift_vector_x_n_np = np.resize(
                #     shiftx_xyz.reshape(-1, 1).numpy(),
                #     (int(max_natoms_per_bin.item() ** 2), shiftx_xyz.numel()),
                # ).T
                # _cell_shift_vector_y_n_np = np.resize(
                #     shifty_xyz.reshape(-1, 1).numpy(),
                #     (int(max_natoms_per_bin.item() ** 2), shifty_xyz.numel()),
                # ).T
                # _cell_shift_vector_z_n_np = np.resize(
                #     shiftz_xyz.reshape(-1, 1).numpy(),
                #     (int(max_natoms_per_bin.item() ** 2), shiftz_xyz.numel()),
                # ).T
                # this basically just tiles shiftx_xyz.reshape(-1, 1) n times
                _cell_shift_vector_x_n = shiftx_xyz.reshape(-1, 1).repeat(
                    (1, int(max_natoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_x_n.shape == _cell_shift_vector_x_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_x_n.numpy(), _cell_shift_vector_x_n_np
                # )
                _cell_shift_vector_y_n = shifty_xyz.reshape(-1, 1).repeat(
                    (1, int(max_natoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_y_n.shape == _cell_shift_vector_y_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_y_n.numpy(), _cell_shift_vector_y_n_np
                # )
                _cell_shift_vector_z_n = shiftz_xyz.reshape(-1, 1).repeat(
                    (1, int(max_natoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_z_n.shape == _cell_shift_vector_z_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_z_n.numpy(), _cell_shift_vector_z_n_np
                # )

                # We have created too many pairs because we assumed each bin
                # has exactly max_natoms_per_bin atoms. Remove all surperfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = torch.logical_and(
                    _first_at_neightuple_n != -1, _secnd_at_neightuple_n != -1
                )
                if mask.sum() > 0:
                    first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
                    secnd_at_neightuple_nn += [_secnd_at_neightuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    # Flatten overall neighbor list.
    first_at_neightuple_n = torch.cat(first_at_neightuple_nn)
    secnd_at_neightuple_n = torch.cat(secnd_at_neightuple_nn)
    cell_shift_vector_n = torch.vstack(
        [
            torch.cat(cell_shift_vector_x_n),
            torch.cat(cell_shift_vector_y_n),
            torch.cat(cell_shift_vector_z_n),
        ]
    ).T

    # Add global cell shift to shift vectors
    cell_shift_vector_n += (
        cell_shift_ic[first_at_neightuple_n] - cell_shift_ic[secnd_at_neightuple_n]
    )

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = torch.logical_not(
            torch.logical_and(
                first_at_neightuple_n == secnd_at_neightuple_n,
                (cell_shift_vector_n == 0).all(dim=1),
            )
        )
        first_at_neightuple_n = first_at_neightuple_n[m]
        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For nonperiodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neightuple_n = first_at_neightuple_n[m]
            secnd_at_neightuple_n = secnd_at_neightuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    i = torch.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    secnd_at_neightuple_n = secnd_at_neightuple_n[i]
    cell_shift_vector_n = cell_shift_vector_n[i]

    # Compute distance vectors.
    distance_vector_nc = (
        positions[secnd_at_neightuple_n]
        - positions[first_at_neightuple_n]
        + cell_shift_vector_n.to(cell.dtype).matmul(cell)  # TODO! .T?
    )
    abs_distance_vector_n = torch.sqrt(
        torch.sum(distance_vector_nc * distance_vector_nc, dim=1)
    )

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    mask = abs_distance_vector_n < max_cutoff
    first_at_neightuple_n = first_at_neightuple_n[mask]
    secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]

    # Assemble return tuple.
    retvals = []
    for q in quantities:
        if q == "i":
            retvals += [first_at_neightuple_n]
        elif q == "j":
            retvals += [secnd_at_neightuple_n]
        elif q == "D":
            retvals += [distance_vector_nc]
        elif q == "d":
            retvals += [abs_distance_vector_n]
        elif q == "S":
            retvals += [cell_shift_vector_n]
        else:
            raise ValueError("Unsupported quantity specified.")

    return retvals