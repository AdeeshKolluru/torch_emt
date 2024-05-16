# torch_emt

This repository contains a torch implementation of EMT potential. 

The logic resembles the code from [differentiable atomic potentials](https://github.com/google/differentiable-atomistic-potentials/tree/master) repository but is completely vectorized and significantly faster.

This has been mostly a weekend project that I haven't spent a long time on but could add value to anyone else working with this potential.

## TODOs

- [ ] Convert `get_neighbors_oneway` function to torch to make it completely differentiable.
- [ ] Wrap the code in a torch model and make the parameters learnable. 

