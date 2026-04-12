import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from torch_emt.calculator import EMTTorchCalc


@pytest.fixture
def cu_slab_with_adsorbate():
    """Cu fcc(111) slab with an O adsorbate."""
    substrate = fcc111("Cu", size=(2, 2, 3), a=3.6, vacuum=10.0)
    adsorbate = Atoms("O", positions=[(0, 0, 4.0)])
    add_adsorbate(substrate, adsorbate, 2.0, position="ontop")
    mask = [atom.position[2] < 1.0 for atom in substrate]
    substrate.set_constraint(FixAtoms(mask=mask))
    return substrate


def test_energy_matches_ase(cu_slab_with_adsorbate):
    atoms = cu_slab_with_adsorbate.copy()

    atoms.calc = EMT()
    ase_energy = atoms.get_potential_energy()

    atoms.calc = EMTTorchCalc(cpu=True)
    torch_energy = atoms.get_potential_energy()

    assert torch_energy == pytest.approx(ase_energy, abs=1e-3)


def test_forces_match_ase(cu_slab_with_adsorbate):
    atoms = cu_slab_with_adsorbate.copy()

    atoms.calc = EMT()
    ase_forces = atoms.get_forces()

    atoms.calc = EMTTorchCalc(cpu=True)
    torch_forces = atoms.get_forces()

    np.testing.assert_allclose(torch_forces, ase_forces, atol=1e-2)


def test_energy_is_differentiable():
    """Verify that gradients flow through the energy calculation."""
    from torch_emt.torch_emt import energy_and_forces

    atoms = fcc111("Cu", size=(1, 1, 2), a=3.6, vacuum=10.0)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    numbers = torch.tensor(atoms.get_atomic_numbers())
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32)

    energy, forces = energy_and_forces(positions, numbers, cell)

    assert energy.requires_grad
    assert forces is not None
    assert forces.shape == positions.shape
