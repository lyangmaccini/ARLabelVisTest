# Regularized Geodesic Distances - Python Implementation

Python implementation of **"A Convex Optimization Framework for Regularized Geodesic Distances"** by Michal Edelstein, Nestor Guillen, Justin Solomon, and Mirela Ben-Chen (SIGGRAPH 2023).

Converted from the original MATLAB implementation: https://github.com/michaled/RGD

## Overview

This package computes geodesic distances on triangle meshes with various regularization terms to achieve smoother, more controlled distance fields. The method solves:

```
maximize ∫ u dA - α·R(u)
subject to: ||∇u|| ≤ 1, u(x₀) = 0
```

using ADMM (Alternating Direction Method of Multipliers).

## Installation

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Optional for interactive visualization
pip install pyvista
```

## Quick Start

```python
from mesh import Mesh
from rgd_admm import rgd_admm
from visualization import plot_distance_field
import matplotlib.pyplot as plt

# Load mesh
mesh = Mesh.from_file('bunny.off')

# Compute regularized geodesic distance
source = 0  # Source vertex index
u = rgd_admm(mesh, source, alpha_hat=0.05)

# Visualize
plot_distance_field(mesh, u, source)
plt.show()
```

## Features

### Regularization Types

1. **Dirichlet Energy** (`reg='D'`)
   - Penalizes ∫||∇u||² dA
   - Smooths the distance field
   - Default and most commonly used

2. **Vector Field Alignment** (`reg='vfa'`)
   - Aligns distance gradients with a given vector field
   - Useful for anisotropic distances
   - Requires `vector_field` parameter

3. **Hessian Energy** (`reg='H'`)
   - Penalizes second derivatives
   - Produces very smooth fields
   - Requires external `curved_hessian` library

### Core Functions

#### Single-Source Distances

```python
u = rgd_admm(mesh, source_index, 
             reg='D',           # Regularization type
             alpha_hat=0.05,    # Regularization weight
             max_iter=10000,    # Max iterations
             quiet=False)       # Show progress
```

#### All-Pairs Distances

```python
U = rgd_allpairs(mesh, 
                alpha_hat=0.03,
                max_iter=20000)

# U[i,j] = distance from vertex i to vertex j
```

#### Vector Field Alignment

```python
# Define sparse constraints
vf = smooth_vector_field(mesh, 
                        sparse_vf,
                        constraint_faces=[10, 20],
                        constraint_values=[[1, 0, 0], [0, 1, 0]])

# Compute aligned distance
u = rgd_admm(mesh, source,
            reg='vfa',
            alpha_hat=0.05,
            beta_hat=100,
            vector_field=vf)
```

## File Structure

```
├── mesh.py                 # Mesh class with differential operators
├── rgd_admm.py            # Single-source ADMM solver
├── rgd_allpairs.py        # All-pairs ADMM solver
├── utils.py               # Utility functions
├── visualization.py       # Plotting functions
├── demo_basic.py          # Basic usage demo
├── demo_vector_field.py   # Vector field alignment demo
├── demo_allpairs.py       # All-pairs computation demo
└── RGD_DOCUMENTATION.md   # Detailed documentation
```

## Examples

### Comparing Regularization Strengths

```python
# No regularization
u0 = rgd_admm(mesh, source, alpha_hat=0.0)

# Light smoothing
u1 = rgd_admm(mesh, source, alpha_hat=0.05)

# Heavy smoothing
u2 = rgd_admm(mesh, source, alpha_hat=0.15)

# Visualize comparison
plot_comparison(mesh, [u0, u1, u2], 
               ['No Reg', 'Light', 'Heavy'],
               source_vertex=source)
```

### Using Custom Meshes

```python
import numpy as np

# Create custom mesh
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

mesh = Mesh(vertices, faces, name='tetrahedron')
```

### Loading Meshes

```python
# From .OFF file
mesh = Mesh.from_file('model.off')

# Or using utils
from utils import read_off
vertices, faces = read_off('model.off')
mesh = Mesh(vertices, faces)
```

## Key Parameters

### `alpha_hat` (Regularization Weight)

- **Scale-invariant** regularization strength
- **Range**: 0.0 (no regularization) to ~0.2 (heavy smoothing)
- **Typical**: 0.05 for light smoothing
- Actual α = `alpha_hat * sqrt(total_area)`

### `beta_hat` (Vector Field Alignment)

- Controls alignment strength for `reg='vfa'`
- **Range**: 0 to ~200
- **Typical**: 50-100
- Actual β = `beta_hat * sqrt(total_area)`

### ADMM Parameters

- `max_iter`: Maximum iterations (default: 10000)
- `abs_tol`: Absolute tolerance (default: 5e-6)
- `rel_tol`: Relative tolerance (default: 0.01)
- `quiet`: Suppress output if True

## Performance

- **Single distance**: 1-10 seconds for 5K-50K vertices
- **All-pairs**: O(N²), practical for <5K vertices
- Main bottleneck: Cholesky factorization in u-step
- Pre-factorization used when penalty parameter is fixed

## Differences from MATLAB

1. **Sparse matrices**: Uses `scipy.sparse` (CSR/CSC formats)
2. **Linear solver**: Uses SciPy's sparse Cholesky (CHOLMOD via scikit-sparse if available)
3. **No CVX required**: ADMM implementation is self-contained
4. **Hessian regularization**: Requires separate installation

## Troubleshooting

### Import errors

```python
# If you see import errors, make sure files are in the same directory
# or add to Python path:
import sys
sys.path.append('/path/to/rgd')
```

### Slow convergence

- Increase `max_iter`
- Adjust `abs_tol` and `rel_tol`
- Try different `alpha_hat` values
- Check mesh quality (no degenerate triangles)

### Memory errors (all-pairs)

- Reduce mesh resolution
- Process in batches
- Use sparse distance matrices

## Citation

```bibtex
@article{edelstein2023convex,
  title={A Convex Optimization Framework for Regularized Geodesic Distances},
  author={Edelstein, Michal and Guillen, Nestor and Solomon, Justin and Ben-Chen, Mirela},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  year={2023}
}
```

## References

- **Paper**: https://mirelabc.github.io/publications/rgd.pdf
- **Supplemental**: https://mirelabc.github.io/publications/rgd_sup.pdf
- **Original MATLAB code**: https://github.com/michaled/RGD

## License

See LICENSE.txt

## Contact

For questions about the Python implementation, please refer to the documentation.

For questions about the method itself, contact: smichale@cs.technion.ac.il
