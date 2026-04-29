"""Differentiable PyTorch implementation of the EMT potential."""

__version__ = "0.1.0"

from .calculator import EMTTorchCalc
from .torch_emt import energy_and_forces

__all__ = ["EMTTorchCalc", "energy_and_forces"]
