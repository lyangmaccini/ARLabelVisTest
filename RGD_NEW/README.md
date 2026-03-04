# Regularized Geodesic Distances - Python Implementation

Python implementation of the ADMM algorithm for computing regularized geodesic distances from:

> "A Convex Optimization Framework for Regularized Geodesic Distances"  
> by Michal Edelstein, Nestor Guillen, Justin Solomon, and Mirela Ben-Chen  
> ACM SIGGRAPH 2023

Original MATLAB code: https://github.com/michaled/RGD

## Overview

This implementation provides tools for computing geodesic distances on triangle meshes with various regularization options:

- **Dirichlet Energy Regularization**: Smooths the distance function
- **Vector Field Alignment**: Aligns distances with a given vector field
- **Hessian Regularization**: Higher-order smoothness (partial implementation)

## Features

- Pure Python implementation using NumPy and SciPy
- ADMM optimization with adaptive penalty parameter
- Support for OFF mesh file format
- Visualization utilities
- Easy-to-use API

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### Files

- `mesh_class.py` - Mesh data structure and differential operators
- `rgd_admm.py` - ADMM algorithm for regularized geodesic distances
- `utils.py` - Utility functions (vector field smoothing, interpolation)
- `demo.py` - Demo script with examples

## Quick Start

### Basic Usage

```python
from mesh_class import MeshClass
from rgd_admm import rgd_admm

# Load mesh from OFF file
vertices, faces = MeshClass.read_off('mesh.off')
mesh = MeshClass(vertices, faces)

# Compute regularized geodesic distance from vertex 0
x0 = 0  # Source vertex
alpha_hat = 0.05  # Regularization weight

u, history = rgd_admm(mesh, x0, reg='D', alpha_hat=alpha_hat)

# u contains the distance function at all vertices
```

### Running the Demo

```python
python demo.py
```

This will:
1. Create a simple test mesh (icosphere)
2. Compute geodesic distances with different regularization weights
3. Visualize the results

## API Reference

### MeshClass

Main class for mesh representation and differential geometry operations.

```python
mesh = MeshClass(vertices, faces)
```

**Attributes:**
- `vertices` - Nx3 array of vertex coordinates
- `faces` - Mx3 array of face indices (0-indexed)
- `nv` - Number of vertices
- `nf` - Number of faces
- `va` - Vertex areas
- `ta` - Face (triangle) areas
- `G` - Gradient operator (3*nf × nv sparse matrix)
- `D` - Divergence operator
- `Ww` - Cotangent Laplacian weights
- `Lap` - Normalized Laplacian

**Methods:**
- `read_off(filename)` - Static method to load mesh from OFF file
- `normalize_vf(vf)` - Normalize vector field
- `barycenter()` - Compute face barycenters

### rgd_admm

Compute regularized geodesic distances using ADMM.

```python
u, history = rgd_admm(mesh, x0, reg='D', alpha_hat=0.1, 
                      beta_hat=0.0, vf=None, max_iter=10000, 
                      quiet=True)
```

**Parameters:**
- `mesh` - MeshClass object
- `x0` - Source vertex indices (int, list, or array)
- `reg` - Regularizer type: `'D'` (Dirichlet), `'H'` (Hessian), or `'vfa'` (vector field alignment)
- `alpha_hat` - Scale-invariant regularization weight (default: 0.1)
- `beta_hat` - Vector field alignment weight (default: 0, only for `'vfa'`)
- `vf` - Vector field for alignment (nf×3 array, only for `'vfa'`)
- `max_iter` - Maximum ADMM iterations (default: 10000)
- `quiet` - Suppress iteration output (default: True)

**Returns:**
- `u` - Distance function at vertices (nv array)
- `history` - Dictionary with convergence history:
  - `r_norm` - Primal residual norms
  - `s_norm` - Dual residual norms
  - `eps_pri` - Primal tolerance thresholds
  - `eps_dual` - Dual tolerance thresholds
  - `rho` - Penalty parameter values

## Examples

### Example 1: No Regularization (Pure Geodesic)

```python
# Compute unregularized geodesic distance
u_geo, _ = rgd_admm(mesh, x0, reg='D', alpha_hat=0.0)
```

### Example 2: Dirichlet Regularization

```python
# Smooth distance function
u_smooth, _ = rgd_admm(mesh, x0, reg='D', alpha_hat=0.1)
```

### Example 3: Vector Field Alignment

```python
import numpy as np
from utils import smooth_vf

# Create vector field (e.g., radial from center)
barycenters = mesh.barycenter()
center = np.mean(barycenters, axis=0)
vf = barycenters - center
vf = mesh.normalize_vf(vf)

# Smooth the vector field
vf_smooth = smooth_vf(mesh, vf, num_iterations=5)

# Compute aligned distance
u_aligned, _ = rgd_admm(mesh, x0, reg='vfa', 
                        alpha_hat=0.05, beta_hat=50.0, vf=vf_smooth)
```

### Example 4: Multiple Source Points

```python
# Compute distance from multiple sources
sources = [0, 10, 20]  # Multiple vertex indices
u, _ = rgd_admm(mesh, sources, reg='D', alpha_hat=0.05)
```

## Regularization Parameters

### Alpha (α̂) - Smoothness Weight

Controls the amount of regularization:
- `α̂ = 0`: No regularization (pure geodesic)
- `α̂ = 0.01-0.05`: Light smoothing
- `α̂ = 0.1-0.5`: Moderate smoothing
- `α̂ > 0.5`: Heavy smoothing

The parameter is **scale-invariant**, meaning the same value works across different mesh sizes.

### Beta (β̂) - Vector Field Alignment Weight

For vector field alignment regularization:
- `β̂ = 0`: No alignment (just Dirichlet)
- `β̂ = 10-50`: Moderate alignment
- `β̂ = 100+`: Strong alignment

## Algorithm Details

The implementation uses the Alternating Direction Method of Multipliers (ADMM) to solve:

```
minimize    E_data(u) + α * E_reg(u)
subject to  |∇u| ≤ 1
```

Where:
- `E_data(u)` enforces boundary conditions at source points
- `E_reg(u)` is the regularization energy (Dirichlet, Hessian, or VFA)
- The constraint enforces the eikonal equation

**Key Features:**
- Adaptive penalty parameter for faster convergence
- Cholesky pre-factorization for efficient linear solves
- Over-relaxation for improved convergence
- Automatic convergence detection

## Performance Tips

1. **Pre-factorization**: The algorithm pre-factors the linear system, making it efficient for multiple distance computations on the same mesh.

2. **Convergence**: Typical convergence in 50-500 iterations. If not converging:
   - Try adjusting `alpha_hat` (increase for more regularization)
   - Check mesh quality (avoid degenerate triangles)
   - Increase `max_iter` if needed

3. **Memory**: Memory usage is O(nv + nf). Large meshes (>100k vertices) may require substantial RAM.

## File Format

### OFF (Object File Format)

```
OFF
nvertices nfaces nedges
x1 y1 z1
x2 y2 z2
...
3 v1 v2 v3
3 v1 v2 v3
...
```

Example:
```
OFF
4 4 0
0.0 0.0 0.0
1.0 0.0 0.0
0.5 1.0 0.0
0.5 0.5 1.0
3 0 1 2
3 0 1 3
3 0 2 3
3 1 2 3
```

## Differences from MATLAB Version

1. **Indexing**: Python uses 0-based indexing (MATLAB uses 1-based)
2. **Matrix Storage**: Uses SciPy sparse matrices (CSR format)
3. **Hessian**: Full Hessian regularization requires external dependencies
4. **Visualization**: Uses Matplotlib instead of MATLAB graphics

## Citation

If you use this code, please cite:

```bibtex
@article{edelstein2023convex,
  title={A Convex Optimization Framework for Regularized Geodesic Distances},
  author={Edelstein, Michal and Guillen, Nestor and Solomon, Justin and Ben-Chen, Mirela},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  year={2023}
}
```

## License

This implementation follows the same license as the original MATLAB code.

## Contact

For questions or issues, please open an issue on GitHub or contact the original authors.

## Acknowledgments

Original MATLAB implementation by Michal Edelstein.  
Python port maintains the core algorithm and API design.
