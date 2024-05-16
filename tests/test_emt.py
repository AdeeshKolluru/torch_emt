import unittest
from emt_pytorch import energy, forces
import torch
import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from emt_gpu_calculator import EMTTorch
from ase.calculators.emt import EMT

class TestEnergyForcesEquality(unittest.TestCase):
    def setUp(self):
        # Define the lattice constant of the catalyst's substrate (fcc111 surface of Cu)
        a = 3.6  # Angstrom

        # Create the catalyst's substrate (fcc111 surface of Cu)
        self.substrate = fcc111("Cu", size=(2, 2, 3), a=a, vacuum=10.0)

        # Define the adsorbate
        adsorbate = Atoms("O", positions=[(0, 0, 4.0)])

        # Add the adsorbate to the substrate
        add_adsorbate(self.substrate, adsorbate, 2.0, position="ontop")

        # Fix the bottom layer of the substrate
        mask = [atom.position[2] < 1.0 for atom in self.substrate]
        self.substrate.set_constraint(FixAtoms(mask=mask))

        # Calculate energy and forces using ASE's EMT calculator
        emt_calc = EMT()
        self.substrate.set_calculator(emt_calc)
        self.emt_energy = self.substrate.get_potential_energy()
        self.emt_forces = self.substrate.get_forces()

    def test_energy_forces_equality(self):
        # Define lattice parameters and initial strain
        cell = torch.tensor(self.substrate.cell.array).float()

        # use emt calculator
        calc = EMTTorch(cpu=True)
        self.substrate.set_calculator(calc)
        energy_value = self.substrate.get_potential_energy()
        forces_value = self.substrate.get_forces()

        # Check if energies are equal
        self.assertAlmostEqual(self.emt_energy, energy_value, places=6)

        # Check if forces are equal
        np.testing.assert_allclose(self.emt_forces, forces_value, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
