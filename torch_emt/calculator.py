"""Differentiable and vectorized torch implementation of EMT Calculator
"""


from ase.calculators.calculator import Calculator
import numpy as np
import torch

from .utils import primitive_neighbor_list
from .torch_emt import energy_and_forces


class EMTTorchCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, cutoff=None, max_num_neighbors_threshold=30, cpu=False):
        Calculator.__init__(self)
        self.cutoff = cutoff
        self.max_num_neighbors_threshold = max_num_neighbors_threshold
        self.cpu = cpu
        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        energy, forces = energy_and_forces(
            torch.tensor(atoms.get_positions(), device=self.device).float(),
            torch.tensor(atoms.get_atomic_numbers(), device=self.device),
            torch.tensor(atoms.get_cell(), device=self.device).float(),
            device=self.device,
        )
        self.results["energy"] = energy.item()
        self.results["forces"] = forces.detach().cpu().numpy()
