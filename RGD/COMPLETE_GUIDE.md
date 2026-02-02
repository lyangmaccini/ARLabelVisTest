# Complete Guide: Regularized Geodesic Distances Python Implementation

## What This Code Does

This Python package computes **geodesic distances** (shortest paths) on 3D triangle meshes with optional **regularization** to make them smoother and more controlled. It's based on a 2023 SIGGRAPH paper and converted from the original MATLAB code.

### The Problem

Standard geodesic distances on meshes can be noisy and have artifacts. This code adds "smoothness penalties" while maintaining the physical constraint that distances grow at most 1 unit per unit length traveled.

---

## How the Algorithm Works

### The Math (Simplified)

The algorithm solves this optimization problem:

```
Find distance function u that:
  - Maximizes: total distance covered across the mesh
  - Minus: a smoothness penalty
  - Subject to: gradients can't exceed 1 (distances can't grow too fast)
  - And: distance is 0 at the source point
```

### The ADMM Algorithm

ADMM breaks this hard problem into 3 easier sub-problems that are solved iteratively:

1. **u-step**: Update distances (solve a linear system)
2. **z-step**: Project gradients to satisfy ||grad|| ≤ 1 constraint
3. **y-step**: Update Lagrange multipliers (dual variables)

Repeat until convergence!

---

## File-by-File Explanation

### 1. `mesh.py` - The Mesh Data Structure

**What it does**: Stores a triangle mesh and computes geometric operators needed for geodesic distances.

**Key components**:

- **Vertices & Faces**: The 3D geometry (vertices are points, faces connect them into triangles)
- **Areas**: Each vertex and triangle has an associated area
- **Gradient Operator (G)**: Converts a function on vertices into gradients on triangles
- **Laplacian (Ww)**: Measures local smoothness (discrete version of ∇²)
- **Normals**: Direction perpendicular to each triangle

**How G works**:
```python
# If you have a distance function u on vertices:
gradient = mesh.G @ u  # Gives 3D gradient on each triangle face
```

**Example usage**:
```python
# Load a mesh from file
mesh = Mesh.from_file('bunny.off')

# Or create from arrays
vertices = np.array([[0,0,0], [1,0,0], [0,1,0]])
faces = np.array([[0, 1, 2]])
mesh = Mesh(vertices, faces)

# Access properties
print(f"Number of vertices: {mesh.nv}")
print(f"Total surface area: {mesh.ta.sum()}")
```

---

### 2. `rgd_admm.py` - Main Distance Computation

**What it does**: Computes regularized geodesic distance from one source point to all other vertices.

**Key function**: `rgd_admm(mesh, source, alpha_hat, ...)`

**Parameters explained**:

- `mesh`: Your mesh object
- `source`: Index of starting vertex (or array of multiple sources)
- `reg`: Regularization type:
  - `'D'` (Dirichlet): Smooth by penalizing squared gradients
  - `'vfa'` (Vector Field Alignment): Guide distances along preferred directions
  - `'H'` (Hessian): Extra smooth by penalizing curvature
- `alpha_hat`: How much smoothing (0 = none, 0.15 = heavy)
- `beta_hat`: Alignment strength (for 'vfa' mode)

**How it works internally**:

1. Set up the problem: eliminate source vertices (they're fixed at distance 0)
2. Initialize variables: u (distances), z (auxiliary gradients), y (dual variables)
3. **Loop** until convergence:
   - Solve linear system for u
   - Project z to unit norm
   - Update y
   - Check if residuals are small enough
4. Return distance function on all vertices

**Example**:
```python
# Compute different regularization levels
u_standard = rgd_admm(mesh, source=0, alpha_hat=0.0)    # No smoothing
u_smooth = rgd_admm(mesh, source=0, alpha_hat=0.05)     # Some smoothing
u_very_smooth = rgd_admm(mesh, source=0, alpha_hat=0.15) # Lots of smoothing
```

---

### 3. `rgd_allpairs.py` - Distance Matrix Computation

**What it does**: Computes distances between **every pair** of vertices simultaneously.

**Output**: Matrix U where `U[i,j]` = distance from vertex i to vertex j

**Why it's different**: Instead of one distance function, computes a full matrix. Uses a more complex ADMM with additional consensus constraints to ensure symmetry: `U[i,j] = U[j,i]`.

**Variables**:
- `X`: Distance matrix (optimizing over column directions)
- `R`: Distance matrix (optimizing over row directions)  
- `U`: Final symmetric consensus distance
- `Z, Q`: Gradient constraints for X and R
- `Y, S, H, K`: Dual variables

**Warning**: This is O(N²) in memory and computation, so only practical for meshes with <5000 vertices.

**Example**:
```python
# Compute all-pairs distances
U = rgd_allpairs(mesh, alpha_hat=0.03)

# Extract distance from vertex 10 to all others
dist_from_10 = U[10, :]

# Find most distant pair
i, j = np.unravel_index(U.argmax(), U.shape)
max_dist = U[i, j]
```

---

### 4. `utils.py` - Helper Functions

**Functions**:

1. **`read_off(filename)`**: Load mesh from .OFF file
   ```python
   vertices, faces = read_off('bunny.off')
   ```

2. **`write_off(filename, vertices, faces)`**: Save mesh to .OFF file

3. **`smooth_vector_field(mesh, constraints, ...)`**: Interpolate sparse vector field across mesh
   - You specify a few faces and their preferred directions
   - It smoothly interpolates to the whole mesh
   - Uses "power fields" (complex number representation)

4. **`geodesic_gaussian(mesh, sources, sigma)`**: Create Gaussian falloff based on geodesic distance
   - Useful for localizing effects

**Vector field smoothing example**:
```python
# I want distances to flow in these directions on these faces
constraint_faces = [10, 50, 100]
constraint_dirs = [[1,0,0], [0,1,0], [0,0,1]]

# Smooth across entire mesh
vf = smooth_vector_field(mesh, 
                        constraint_faces=constraint_faces,
                        constraint_values=constraint_dirs)
```

---

### 5. `visualization.py` - Plotting Functions

**Functions**:

1. **`plot_mesh(mesh, vertex_colors, ...)`**: Basic 3D mesh plot
2. **`plot_distance_field(mesh, distances, source, ...)`**: Show distance with colors
3. **`plot_vector_field(mesh, vectors, ...)`**: Draw arrows showing directions
4. **`plot_comparison(mesh, [dist1, dist2, ...], labels)`**: Side-by-side comparison

**Example**:
```python
import matplotlib.pyplot as plt

# Compute distances
u = rgd_admm(mesh, source=0, alpha_hat=0.05)

# Visualize
plot_distance_field(mesh, u, source=0, 
                   cmap='jet', n_isolines=15)
plt.show()
```

---

## How to Run the Demos

### Demo 1: Basic Usage (`demo_basic.py`)

Shows Dirichlet regularization with different strengths.

```bash
python demo_basic.py
```

**What it does**:
1. Creates/loads a mesh
2. Computes 4 distance fields with varying smoothness
3. Plots them side-by-side
4. Shows statistics

---

### Demo 2: Vector Field Alignment (`demo_vector_field.py`)

Shows how to guide distances along preferred directions.

```bash
python demo_vector_field.py
```

**What it does**:
1. Creates a mesh (cylinder)
2. Defines a sparse vector field (a few faces with directions)
3. Smooths it across the mesh
4. Computes distances that align with the vector field
5. Visualizes both the field and resulting distances

---

### Demo 3: All-Pairs Distances (`demo_allpairs.py`)

Computes the full distance matrix.

```bash
python demo_allpairs.py
```

**What it does**:
1. Creates a small mesh
2. Computes U[i,j] for all vertex pairs
3. Shows sample distance fields
4. Analyzes the distance matrix (histogram, diameter, etc.)
5. Plots convergence

---

## Common Use Cases

### 1. Smooth Geodesic Distances
```python
mesh = Mesh.from_file('model.off')
u = rgd_admm(mesh, source=0, alpha_hat=0.05)
plot_distance_field(mesh, u, source=0)
```

### 2. Anisotropic Distances
```python
# Create vector field pointing in preferred direction
vf = smooth_vector_field(mesh, 
                        constraint_faces=[10, 20],
                        constraint_values=[[1,0,0], [1,0,0]])

# Compute distances aligned with field
u = rgd_admm(mesh, source=0, reg='vfa', 
            alpha_hat=0.05, beta_hat=100, vector_field=vf)
```

### 3. Compare Regularization Levels
```python
distances = [
    rgd_admm(mesh, 0, alpha_hat=0.0),
    rgd_admm(mesh, 0, alpha_hat=0.05),
    rgd_admm(mesh, 0, alpha_hat=0.10),
]
labels = ['None', 'Light', 'Medium']
plot_comparison(mesh, distances, labels, source_vertex=0)
```

### 4. Distance Between Specific Points
```python
# Compute from source A to all points
u = rgd_admm(mesh, source=vertex_A, alpha_hat=0.05)

# Distance from A to B is:
dist_AB = u[vertex_B]
```

---

## Understanding the Parameters

### `alpha_hat` (Smoothing Strength)

- **0.0**: No smoothing (standard geodesic, may have noise)
- **0.01-0.03**: Very light smoothing
- **0.05**: Good default for moderate smoothing
- **0.10**: Heavy smoothing
- **0.15+**: Very heavy smoothing (distance field becomes very flat)

The actual regularization weight is `α = alpha_hat * sqrt(total_area)`, making it scale-invariant (same alpha_hat works regardless of mesh size).

### `beta_hat` (Alignment Strength, for vfa mode)

- **0**: No alignment
- **50**: Moderate alignment  
- **100**: Strong alignment (typical)
- **200+**: Very strong alignment

### ADMM Convergence Parameters

- `max_iter`: Stop after this many iterations (default: 10000)
- `abs_tol`: Absolute residual tolerance (default: 5e-6)
- `rel_tol`: Relative residual tolerance (default: 0.01)

The algorithm stops when:
```
primal_residual < abs_tol + rel_tol * scale
dual_residual < abs_tol + rel_tol * scale
```

---

## Tips & Tricks

### For Better Performance

1. **Pre-normalize meshes**: Center and scale to unit bounding box
   ```python
   mesh.normalize_mesh()
   ```

2. **Increase tolerance for faster (but less accurate) results**:
   ```python
   u = rgd_admm(mesh, source, abs_tol=1e-4, rel_tol=0.1)
   ```

3. **Use sparse linear solvers**: Install `scikit-sparse` for faster Cholesky

### For Better Results

1. **Start with small alpha_hat** and increase gradually
2. **Check gradient constraints**: Ensure max gradient ≈ 1
   ```python
   grad_norms = compute_gradient_norm(mesh, u)
   print(f"Max gradient: {grad_norms.max()}")  # Should be ≈ 1
   ```

3. **For vector field alignment**: Localize with geodesic Gaussian to avoid global effects

---

## Troubleshooting

### "Convergence slow"
- Mesh may have issues (degenerate triangles, bad aspect ratios)
- Try different `alpha_hat`
- Increase `max_iter`

### "Max gradient >> 1"
- Constraint not satisfied - algorithm didn't converge
- Increase `max_iter`
- Decrease tolerances

### "Out of memory (all-pairs)"
- Matrix is NxN - reduce mesh resolution
- Process in batches
- Use sparse storage

### "Vector field empty"
- Check that constraint faces have non-zero directions
- Normalize constraint directions

---

## Key Differences from MATLAB Code

1. **Sparse matrices**: MATLAB uses built-in sparse, Python uses `scipy.sparse`
2. **Array indexing**: MATLAB is 1-indexed, Python is 0-indexed
3. **Linear algebra**: MATLAB's `\` operator → Python's `spsolve()` or `factorized()`
4. **Matrix operations**: Careful with `.T` (transpose) vs MATLAB's `'`
5. **Visualization**: MATLAB patches → Python's matplotlib Poly3DCollection

---

## What Each MATLAB File Became

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| `MeshClass.m` | `mesh.py` | Mesh data structure |
| `rdg_ADMM.m` | `rgd_admm.py` | Single-source solver |
| `rdg_allpairs_admm.m` | `rgd_allpairs.py` | All-pairs solver |
| `cotLaplacian.m` | `utils.py` (function) | Laplacian computation |
| `readOff.m` | `utils.py` (function) | OFF file reader |
| `smooth_vf.m` | `utils.py` (function) | Vector field smoothing |
| `demo.m` | `demo_basic.py` | Basic demo |
| `cvx_demo.m` | *(not needed)* | CVX not required in Python |
| `Hessian_demo.m` | *(documented)* | Needs external library |
| `allpairs_demo.m` | `demo_allpairs.py` | All-pairs demo |

---

## Quick Reference

### Load a mesh
```python
mesh = Mesh.from_file('model.off')
```

### Compute regularized distance
```python
u = rgd_admm(mesh, source, alpha_hat=0.05)
```

### Visualize
```python
plot_distance_field(mesh, u, source)
plt.show()
```

### All-pairs
```python
U = rgd_allpairs(mesh, alpha_hat=0.03)
```

### Vector field alignment
```python
vf = smooth_vector_field(mesh, constraint_faces, constraint_values)
u = rgd_admm(mesh, source, reg='vfa', beta_hat=100, vector_field=vf)
```

---

## Summary

This Python package provides a complete implementation of regularized geodesic distances on triangle meshes. The key innovation is adding smoothness penalties while maintaining the physical constraint that gradients can't exceed 1. The ADMM algorithm efficiently solves this constrained optimization problem through iterative refinement of distance functions, auxiliary variables, and dual multipliers.

The code is organized into modular components (mesh geometry, ADMM solvers, utilities, visualization) with comprehensive demos showing basic usage, vector field alignment, and all-pairs computation.
