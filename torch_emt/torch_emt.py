"""Differentiable and vectorized torch implementation of EMT inspired from
 https://github.com/google/differentiable-atomistic-potentials/blob/master/dap/ag/emt.py

This implementation is completely differentiable and vectorized in PyTorch.
"""

import math
from ase.units import Bohr

import torch
from torch_scatter import scatter
from torch.autograd import grad

from .utils import primitive_neighbor_list


PARAM_DICT = {
    #      E0     s0    V0     eta2    kappa   lambda  n0
    #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
    13: (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),
    29: (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),
    47: (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),
    79: (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),
    28: (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030),
    46: (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),
    78: (-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802),
    # extra parameters - just for fun ...
    1: (-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547),
    6: (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322),
    7: (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222),
    8: (-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850),
}

BETA = 1.809  # (16 * pi / 3)**(1.0 / 3) / 2**0.5, preserve historical rounding


def _params_defined_check(numbers):
    for n in numbers:
        if n.item() not in PARAM_DICT.keys():
            raise ValueError(f"Parameters not defined for atomic number {n}")


def energy_and_forces(positions, numbers, cell, strain=None, device="cpu"):

    if strain is None:
        strain = torch.zeros((3, 3), device=device)

    strain_tensor = torch.eye(3, device=device) + strain
    cell = torch.mm(strain_tensor, cell.t()).t()
    positions = torch.mm(strain_tensor, positions.t()).t()
    par, rc_list, acut, rc = _calc_params(numbers)

    natoms = len(positions)

    positions.requires_grad = True
    i, j, d = primitive_neighbor_list(
        quantities="ijd",
        pbc=[True] * 3,
        cell=cell,
        positions=positions,
        cutoff=rc_list,
    )
    edge_index = torch.stack([j, i], dim=0)
    edge_dist = d

    # Calculate

    energy, sigma_per_node = _edge_energy(
        edge_dist, edge_index, numbers, par, acut, rc, rc_list
    )

    ds = -torch.log(sigma_per_node / 12) / (BETA * par[numbers][:, 3])
    x = par[numbers][:, 5] * ds
    y = torch.exp(-x)
    z = 6 * par[numbers][:, 2] * torch.exp(-par[numbers][:, 4] * ds)
    energy += sum(par[numbers][:, 0] * ((1 + x) * y - 1) + z)

    forces = -grad(energy, positions, create_graph=True, allow_unused=True)[0]

    return energy, forces


def _edge_energy(edge_dist, edge_index, numbers, params, acut, rc, rc_list):
    x = torch.exp(acut * (edge_dist - rc))
    theta = 1.0 / (1.0 + x)
    source_atomic_number, target_atomic_number = (
        numbers[edge_index[0]],
        numbers[edge_index[1]],
    )
    energy_per_edge = (
        0.5
        * params[target_atomic_number][:, 2]
        * torch.exp(
            -params[source_atomic_number][:, 4]
            * (edge_dist / BETA - params[source_atomic_number][:, 1])
        )
        * theta
        * (params[source_atomic_number][:, 6] / params[target_atomic_number][:, 6])
        / params[target_atomic_number][:, 9]
    )

    mask = edge_dist > rc_list
    energy_per_edge[mask] = 0

    sigma_per_edge = (
        torch.exp(
            -params[source_atomic_number][:, 3]
            * (edge_dist - BETA * params[source_atomic_number][:, 1])
        )
        * (params[source_atomic_number][:, 6] / params[target_atomic_number][:, 6])
        * theta
        / params[target_atomic_number][:, 8]
    )

    sigma_per_node = scatter(sigma_per_edge, edge_index[1], dim=0, reduce="sum")

    return -sum(energy_per_edge), sigma_per_node


def _calc_params(numbers, device="cpu"):
    rc = 0.0

    # TODO: this is memory inefficient but in future we can parametrize and learn this for all elements
    parameters = torch.zeros(119, 7, device=device)

    _params_defined_check(numbers)

    for atomic_number, params in PARAM_DICT.items():
        parameters[atomic_number] = torch.tensor(params)

    maxseq = torch.max(parameters[:, 1]) * Bohr

    rc = rc = BETA * maxseq * 0.5 * (math.sqrt(3) + math.sqrt(4))
    rr = rc * 2 * math.sqrt(4) / (math.sqrt(3) + math.sqrt(4))
    acut = math.log(9999.0) / (rr - rc)

    rc_list = rc + 0.5

    s0 = parameters[numbers][:, 1] * Bohr
    eta2 = parameters[numbers][:, 3] / Bohr
    kappa = parameters[numbers][:, 4] / Bohr
    x = eta2 * BETA * s0
    gamma1 = 0.0
    gamma2 = 0.0
    for i, n in enumerate([12, 6, 24]):
        r = s0 * BETA * math.sqrt(i + 1)
        x = n / (12 * (1.0 + torch.exp(acut * (r - rc))))
        gamma1 += x * torch.exp(-eta2 * (r - BETA * s0))
        gamma2 += x * torch.exp(-kappa / BETA * (r - BETA * s0))

    new_params = torch.zeros(119, 10, device=device)

    new_params[:, 0] = parameters[:, 0]  # E0
    new_params[:, 1][numbers] = s0  # s0
    new_params[:, 2] = parameters[:, 2]  # V0
    new_params[:, 3][numbers] = eta2  # eta2
    new_params[:, 4][numbers] = kappa  # kappa
    new_params[:, 5] = parameters[:, 5] / Bohr  # lambda
    new_params[:, 6] = parameters[:, 6] / Bohr**3  # n0
    new_params[:, 7] = rc  # rc
    new_params[:, 8][numbers] = gamma1  # gamma1
    new_params[:, 9][numbers] = gamma2  # gamma2
    return new_params, rc_list, acut, rc
