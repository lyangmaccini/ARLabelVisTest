# Regularized Geodesic Distances - Python Implementation

## Overview

This is a Python implementation of "A Convex Optimization Framework for Regularized Geodesic Distances" by Edelstein et al. (SIGGRAPH 2023). The code computes geodesic distances on triangle meshes with various regularization terms to achieve smoother, more controlled distance fields.

## How It Works

### Core Concept

Traditional geodesic distances on meshes can have artifacts (noise, sharp features). This framework adds regularization terms to smooth the distance field while maintaining the constraint that gradients have unit norm (distance increases by 1 per unit length).

The optimization problem is:
```
maximize ∫ u dA - α·R(u)
subject to: ||∇u|| ≤ 1 (unit gradient constraint)
            u(x₀) = 0 (boundary condition at source)
```

Where:
- `u` is the distance function
- `α` is the regularization weight
- `R(u)` is a regularization term (Dirichlet, Hessian, or Vector Field Alignment)

### Algorithm: ADMM (Alternating Direction Method of Multipliers)

The code uses ADMM to solve this constrained optimization problem by splitting it into simpler subproblems:

1. **u-step**: Update the distance function u (requires solving a linear system)
2. **z-step**: Project gradients to satisfy ||∇u|| ≤ 1
3. **y-step**: Update dual variables (Lagrange multipliers)

The algorithm alternates between these steps until convergence.

## File Structure

### Core Files

#### 1. **mesh.py** - Mesh Data Structure
The `Mesh` class stores all geometric information about a triangle mesh:

- **Vertices & Faces**: Basic mesh geometry
- **Areas**: 
  - `va`: vertex areas (1/3 of surrounding face areas)
  - `ta`: triangle/face areas
- **Differential Operators**:
  - `G`: Gradient operator (maps vertex functions to face gradients)
  - `Ww`: Cotangent Laplacian (weighted graph Laplacian)
- **Edge Information**: Edge connectivity, interior edges, boundary vertices
- **Normal Vectors**: Per-face and per-vertex normals

**Key Methods**:
- `compute_gradient_operator()`: Builds sparse matrix that computes gradients
- `compute_cotangent_laplacian()`: Builds the Laplace-Beltrami operator
- `interpolate_*()`: Convert between vertex and face data

#### 2. **rgd_admm.py** - Main ADMM Solver
Implements the ADMM algorithm for computing regularized geodesic distances.

**Parameters**:
- `mesh`: Mesh object
- `source_indices`: Starting point(s) for distance computation
- `reg`: Regularization type ('D', 'H', or 'vfa')
- `alpha_hat`: Scale-invariant regularization weight
- `beta_hat`: Vector field alignment weight (for 'vfa' mode)
- `vector_field`: Target vector field (for 'vfa' mode)

**Regularization Types**:
1. **'D' (Dirichlet)**: Penalizes ∫||∇u||² - smooths the distance field
2. **'H' (Hessian)**: Penalizes second derivatives - even smoother
3. **'vfa' (Vector Field Alignment)**: Aligns distance gradients with a given vector field

**ADMM Variables**:
- `u`: Distance function on vertices (excluding source)
- `z`: Auxiliary gradient variable (enforces unit norm constraint)
- `y`: Dual variable (Lagrange multiplier)

**Algorithm Flow**:
```
for iteration in range(max_iterations):
    1. u-step: Solve (α·L + ρ·W)u = va - div(y) + ρ·div(z)
    2. z-step: z = project_unit_norm(G·u + y/ρ)
    3. y-step: y = y + ρ·(G·u - z)
    
    Check convergence:
        - Primal residual: ||G·u - z||
        - Dual residual: ||div(z - z_old)||
```

#### 3. **rgd_allpairs.py** - All-Pairs Distances
Computes distances between all pairs of vertices simultaneously using a matrix formulation.

**Key Difference**: Instead of computing one distance function `u`, computes matrix `U` where `U[i,j]` is the distance from vertex i to vertex j.

**Variables**:
- `X`: Distance matrix (gradients along columns)
- `R`: Distance matrix (gradients along rows) 
- `U`: Consensus variable (symmetric distance matrix)
- `Z, Q`: Auxiliary gradient variables
- `Y, S, H, K`: Dual variables

**Why Two Representations (X and R)?**
- Mesh gradients aren't symmetric
- Need both column and row gradients to enforce symmetry: U = (X + R^T)/2

#### 4. **utils.py** - Utility Functions
Helper functions for mesh processing:
- `read_off()`: Load .OFF mesh files
- `compute_cotangent_laplacian()`: Standalone Laplacian computation
- `smooth_vector_field()`: Interpolate sparse vector field data across mesh
- `normalize_vectors()`: Normalize vector fields

#### 5. **visualization.py** - Visualization Tools
Functions to visualize results:
- `plot_mesh()`: Display mesh with colors
- `plot_distance_field()`: Show distance field with isolines
- `plot_vector_field()`: Overlay vector fields on mesh

### Demo Files

#### **demo_basic.py**
Shows basic usage with Dirichlet regularization:
```python
# Load mesh
mesh = Mesh.from_file('spot.off')

# Compute standard geodesic distance (no regularization)
u_standard = rgd_admm(mesh, source_vertex, alpha_hat=0)

# Compute regularized distance (Dirichlet)
u_regularized = rgd_admm(mesh, source_vertex, alpha_hat=0.05)

# Visualize
plot_distance_field(mesh, u_regularized)
```

#### **demo_vector_field.py**
Shows vector field alignment regularization:
```python
# Define sparse vector field hints
vf_faces = [4736, 2703]  # Face indices
vf_directions = [[1.6, -0.35, -0.62], [1.7, 0.32, 0.03]]

# Interpolate to full mesh
vf_full = smooth_vector_field(mesh, vf_faces, vf_directions)

# Compute distance aligned with vector field
u_vfa = rgd_admm(mesh, source_vertex, 
                 reg='vfa', 
                 alpha_hat=0.05, 
                 beta_hat=100,
                 vector_field=vf_full)
```

#### **demo_allpairs.py**
Computes all-pairs distance matrix:
```python
# Compute regularized distance matrix
U = rgd_allpairs(mesh, alpha_hat=0.03)

# U[i,j] contains distance from vertex i to vertex j
# Can extract individual distance fields:
dist_from_vertex_100 = U[100, :]
```

## Mathematical Details

### Gradient Operator G

Converts vertex function u to face gradients ∇u:
```
∇u_f = Σᵢ u_vᵢ · ∇φᵢ
```
where φᵢ are barycentric coordinate functions.

For a triangle with vertices v₁, v₂, v₃:
```
∇φ₁ = (v₃ - v₂) × n / (2·Area)
```
where n is the face normal.

### Cotangent Laplacian

Discrete Laplace-Beltrami operator:
```
(L·u)ᵢ = (1/Aᵢ) Σⱼ (cot αᵢⱼ + cot βᵢⱼ)(uⱼ - uᵢ)
```
where αᵢⱼ, βᵢⱼ are angles opposite edge (i,j).

### Regularization Terms

**Dirichlet Energy**:
```
R_D(u) = ∫||∇u||² dA = u^T·L·u
```

**Hessian Energy**:
```
R_H(u) = ∫||Hess(u)||² dA
```
Requires special curved Hessian computation (see Stein et al. 2020).

**Vector Field Alignment**:
```
R_VFA(u) = ∫||∇u||² dA + β·∫<∇u, v>² dA
```
Encourages gradients to align with vector field v.

## Usage Examples

### Basic Distance Computation
```python
import numpy as np
from mesh import Mesh
from rgd_admm import rgd_admm
from visualization import plot_distance_field

# Load mesh
mesh = Mesh.from_file('bunny.off')

# Select source vertex (center of mesh)
source = mesh.nv // 2

# Compute regularized distance
u = rgd_admm(mesh, source, alpha_hat=0.05)

# Visualize
plot_distance_field(mesh, u, source)
```

### Comparing Regularization Strengths
```python
# No regularization (standard geodesic)
u0 = rgd_admm(mesh, source, alpha_hat=0.0)

# Light regularization
u1 = rgd_admm(mesh, source, alpha_hat=0.05)

# Heavy regularization
u2 = rgd_admm(mesh, source, alpha_hat=0.15)

# Plot comparison
fig, axes = plt.subplots(1, 3)
plot_distance_field(mesh, u0, source, ax=axes[0])
plot_distance_field(mesh, u1, source, ax=axes[1])
plot_distance_field(mesh, u2, source, ax=axes[2])
```

### Advanced: Custom Vector Field
```python
# Create custom vector field
vf = np.zeros((mesh.nf, 3))

# Set direction on some faces
important_faces = [100, 200, 300]
vf[important_faces] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Smooth across mesh
vf_smooth = smooth_vector_field(mesh, vf, power=2)

# Compute aligned distance
u = rgd_admm(mesh, source, 
             reg='vfa',
             alpha_hat=0.05,
             beta_hat=100,
             vector_field=vf_smooth)
```

## Key Parameters

### alpha_hat (Scale-Invariant Regularization Weight)
- **Range**: 0 to ~0.2
- **Effect**: Controls smoothness of distance field
- **0.0**: No regularization (standard geodesic)
- **0.05**: Light smoothing (typical)
- **0.15**: Heavy smoothing

The actual regularization weight is: `α = alpha_hat * sqrt(total_surface_area)`

### beta_hat (Vector Field Alignment Weight)
- **Range**: 0 to ~200
- **Effect**: How strongly to align with vector field
- **Only used for 'vfa' regularization**

### ADMM Parameters
- **rho**: Penalty parameter (~2·sqrt(area))
- **max_iter**: Maximum iterations (default: 10000)
- **abstol, reltol**: Convergence tolerances

## Dependencies

```python
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
```

Optional for Hessian regularization:
- Custom Hessian library (see original paper)

## Performance Notes

- **Single Distance**: ~1-10 seconds for meshes with 5K-50K vertices
- **All-Pairs**: Much slower (N² problem), practical for <5K vertices
- Main bottleneck: Cholesky factorization in u-step
- Pre-factorization helps when rho is fixed

## Differences from MATLAB Version

1. **Sparse Matrices**: Uses scipy.sparse (CSR/CSC format)
2. **Linear Solver**: Uses scipy's sparse Cholesky (CHOLMOD)
3. **No CVX**: ADMM implementation doesn't need CVX
4. **Hessian**: Requires separate installation of curved Hessian library

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

- Original paper: https://mirelabc.github.io/publications/rgd.pdf
- Supplemental material: https://mirelabc.github.io/publications/rgd_sup.pdf
- MATLAB code: https://github.com/michaled/RGD
