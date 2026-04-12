# torch_emt

[![CI](https://github.com/AdeeshKolluru/torch_emt/actions/workflows/ci.yml/badge.svg)](https://github.com/AdeeshKolluru/torch_emt/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/torch_emt.svg)](https://pypi.org/project/torch_emt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **differentiable** and **GPU-accelerated** PyTorch implementation of the [Effective Medium Theory (EMT)](https://doi.org/10.1016/0039-6028(96)80007-0) potential for atomistic simulations.

## Features

- Fully differentiable energy and force calculations via PyTorch autograd
- Seamless integration with [ASE](https://wiki.fysik.dtu.dk/ase/) as a drop-in calculator
- GPU acceleration with automatic CUDA detection
- Support for strain-dependent calculations
- Parameters for Al, Cu, Ag, Au, Ni, Pd, Pt, H, C, N, O

## Installation

```bash
pip install torch_emt
```

**From source:**

```bash
git clone https://github.com/AdeeshKolluru/torch_emt.git
cd torch_emt
pip install -e ".[dev]"
```

### Prerequisites

- PyTorch >= 1.11
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter) (install matching your PyTorch/CUDA version)

## Usage

### As an ASE calculator

```python
from ase.build import fcc111, add_adsorbate
from ase import Atoms
from torch_emt import EMTTorchCalc

# Build a Cu slab with an O adsorbate
slab = fcc111("Cu", size=(2, 2, 3), a=3.6, vacuum=10.0)
add_adsorbate(slab, Atoms("O", positions=[(0, 0, 0)]), 2.0, position="ontop")

# Attach the calculator and compute
slab.calc = EMTTorchCalc(cpu=True)  # set cpu=False to use CUDA
energy = slab.get_potential_energy()
forces = slab.get_forces()
```

### Direct energy and force computation

```python
import torch
from torch_emt import energy_and_forces

positions = torch.tensor([[0.0, 0.0, 0.0], [2.55, 0.0, 0.0]], dtype=torch.float32)
numbers = torch.tensor([29, 29])  # Cu atoms
cell = torch.eye(3) * 10.0

energy, forces = energy_and_forces(positions, numbers, cell)
```

## Citation

If you use this code in your research, please cite this repository:

```
@software{torch_emt,
  author = {Kolluru, Adeesh},
  title = {torch_emt: Differentiable PyTorch EMT Potential},
  url = {https://github.com/AdeeshKolluru/torch_emt},
}
```

## License

MIT
