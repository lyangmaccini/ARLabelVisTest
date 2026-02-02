# MATLAB to Python Translation Guide

This document shows how the original MATLAB code maps to the Python implementation.

## Code Structure Mapping

### MATLAB Classes → Python Classes

```matlab
% MATLAB
classdef MeshClass < handle
    properties
        vertices, faces, nv, nf, ...
    end
    methods
        function obj = MeshClass(filename)
        ...
end
```

```python
# Python
class Mesh:
    def __init__(self, vertices, faces, name="mesh"):
        self.vertices = vertices
        self.faces = faces
        self.nv = len(vertices)
        self.nf = len(faces)
        ...
```

---

## Key Function Translations

### 1. Gradient Operator

**MATLAB (`MeshClass.m` lines 225-242)**:
```matlab
function grad_op = GG(mesh)            
    I = repmat(1:mesh.nf,3,1);
    II = [I(:); I(:)+mesh.nf; I(:)+2*mesh.nf];
    J = double( mesh.faces' );
    JJ = [J(:); J(:); J(:)];
    RE1 = mesh.rotate_vf( mesh.E1 );
    RE2 = mesh.rotate_vf( mesh.E2 );
    RE3 = mesh.rotate_vf( mesh.E3 );
    S = [(RE1./mesh.ta2)' ; (RE2./mesh.ta2)' ; (RE3./mesh.ta2)'];
    SS = [S(:,1) ; S(:,2) ; S(:,3)];
    grad_op = sparse(II,JJ,SS,3*mesh.nf,mesh.nv);
```

**Python (`mesh.py` lines 177-240)**:
```python
def _compute_gradient_operator(self) -> csr_matrix:
    nf = self.nf
    v1 = self.vertices[self.faces[:, 0]]
    v2 = self.vertices[self.faces[:, 1]]
    v3 = self.vertices[self.faces[:, 2]]
    
    e1 = v2 - v3
    e2 = v3 - v1
    e3 = v1 - v2
    
    re1 = np.cross(self.Nf, e1)
    re2 = np.cross(self.Nf, e2)
    re3 = np.cross(self.Nf, e3)
    
    grad1 = re1 / (2 * self.ta[:, np.newaxis])
    grad2 = re2 / (2 * self.ta[:, np.newaxis])
    grad3 = re3 / (2 * self.ta[:, np.newaxis])
    
    # Build sparse matrix...
    G = csr_matrix((values, (row_idx, col_idx)), 
                   shape=(3 * nf, self.nv))
    return G
```

**Key differences**:
- MATLAB uses `repmat()`, Python uses broadcasting
- MATLAB sparse: `sparse(I,J,V)`, Python: `csr_matrix((V,(I,J)))`
- MATLAB `:` operator for flattening, Python uses `.flatten()` or `.reshape()`

---

### 2. Main ADMM Algorithm

**MATLAB (`rdg_ADMM.m` core loop lines 108-142)**:
```matlab
for ii = 1:niter
    % step 1 - u-minimization
    b = va_p-div_y+rho*div_z;
    u_p = P * (L'  \ (L \ (P' * b ))) /(alpha+rho);
    Gx = G_p*u_p;

    % step 2 - z-minimization
    zold = z;
    z = (1/rho)*y + Gx;
    z = reshape(z,nf,3)'; 
    norm_z = sqrt(sum(z.^2));
    norm_z(norm_z<1) = 1;
    z = z./norm_z;
    z = z'; z = z(:);
    
    % step 3 - dual variable update
    y = y + rho*(alphak*Gx+(1-alphak)*zold-z);
    ...
end
```

**Python (`rgd_admm.py` lines 149-195)**:
```python
for iteration in range(max_iter):
    # Step 1: u-minimization
    b = va_p - div_y + rho * div_z
    u_p = A_fact(b) / (alpha + rho)
    Gx = G_p @ u_p
    
    # Step 2: z-minimization
    z_old = z.copy()
    z_temp = (1.0 / rho) * y + Gx
    z_temp = z_temp.reshape(nf, 3)
    
    z_norms = np.linalg.norm(z_temp, axis=1)
    z_norms[z_norms < 1.0] = 1.0
    z = (z_temp / z_norms[:, np.newaxis]).flatten()
    
    # Step 3: dual variable update
    y = y + rho * (alpha_k * Gx + (1 - alpha_k) * z_old - z)
    ...
```

**Key differences**:
- MATLAB `\` (backslash) → Python `factorized()` or `spsolve()`
- MATLAB `*` (matrix multiply) → Python `@` operator
- MATLAB `sum(z.^2)` → Python `np.linalg.norm(z, axis=1)`
- MATLAB `z(:)` (column vector) → Python `z.flatten()`

---

### 3. Cotangent Laplacian

**MATLAB (`cotLaplacian.m` lines 1-35)**:
```matlab
function [W, A] = cotLaplacian(mesh, L23, L13, L12)
    X = mesh.vertices;
    T = mesh.faces;
    nv = size(X,1);
    
    L1 = normv(X(T(:,2),:)-X(T(:,3),:));
    L2 = normv(X(T(:,1),:)-X(T(:,3),:));
    L3 = normv(X(T(:,1),:)-X(T(:,2),:));
    
    A1 = (L2.^2 + L3.^2 - L1.^2) ./ (2.*L2.*L3);
    A2 = (L1.^2 + L3.^2 - L2.^2) ./ (2.*L1.*L3);
    A3 = (L1.^2 + L2.^2 - L3.^2) ./ (2.*L1.*L2);
    A = [A1,A2,A3];
    A = acos(A);
    
    I = [T(:,1);T(:,2);T(:,3)];
    J = [T(:,2);T(:,3);T(:,1)];
    S = 0.5*cot([A(:,3);A(:,1);A(:,2)]);
    In = [I;J;I;J];
    Jn = [J;I;I;J];
    Sn = [-S;-S;S;S];
    W = sparse(double(In),double(Jn),Sn,nv,nv);
```

**Python (`mesh.py` lines 242-288)**:
```python
def _compute_cotangent_laplacian(self) -> csr_matrix:
    v1 = self.vertices[self.faces[:, 0]]
    v2 = self.vertices[self.faces[:, 1]]
    v3 = self.vertices[self.faces[:, 2]]
    
    L1 = np.linalg.norm(v2 - v3, axis=1)
    L2 = np.linalg.norm(v3 - v1, axis=1)
    L3 = np.linalg.norm(v1 - v2, axis=1)
    
    A1 = np.arccos((L2**2 + L3**2 - L1**2) / (2 * L2 * L3 + 1e-10))
    A2 = np.arccos((L1**2 + L3**2 - L2**2) / (2 * L1 * L3 + 1e-10))
    A3 = np.arccos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2 + 1e-10))
    
    cot_A1 = 1.0 / np.tan(A1)
    cot_A2 = 1.0 / np.tan(A2)
    cot_A3 = 1.0 / np.tan(A3)
    
    # Build sparse matrix
    I_list = []
    J_list = []
    V_list = []
    
    I_list.extend(self.faces[:, 0])
    J_list.extend(self.faces[:, 1])
    V_list.extend(0.5 * cot_A3)
    # ... similar for other edges
    
    W = csr_matrix((V_sym, (I_sym, J_sym)), shape=(self.nv, self.nv))
```

**Key differences**:
- MATLAB `normv()` custom function → Python `np.linalg.norm()`
- MATLAB `cot()` → Python `1.0 / np.tan()`
- MATLAB matrix concatenation `[A;B]` → Python `list.extend()`

---

### 4. All-Pairs ADMM

**MATLAB (`rdg_allpairs_admm.m` lines 73-100)**:
```matlab
for ii = 1:niter
    % step 1 - X,R-minimization
    bx = (0.5*vavatMat.*vainv' - div_Y + rho*div_Z - va.*H + rho2*va.*U );
    br = (0.5*vavatMat.*vainv' - div_S + rho*div_Q - va.*K + rho2*va.*U' );
    
    X = P * (L'  \ (L \ (P' * bx )));
    R = P * (L'  \ (L \ (P' * br )));
    
    Gx = G*X;
    Gr = G*R;
    
    % step 2 - Z,Q,U-minimization:
    Z = (1/rho)*Y + Gx;
    Z = reshape(Z,nf,3,nv); 
    Z = Z./max(1,sqrt(sum(Z.^2,2)));
    Z = reshape(Z,3*nf,nv);
    
    U1 = 0.5*((1/rho2)*(H+K') + X+R');
    U = U1-diag(diag(U1));
    U(U<0) = 0;
```

**Python (`rgd_allpairs.py` lines 91-150)**:
```python
for iteration in range(max_iter):
    # Step 1: X and R minimization
    va_mat_full = va[:, np.newaxis] * np.ones((nv, nv))
    bx = (0.5 * va_mat_full * va_inv - div_Y + rho1 * div_Z 
          - va[:, np.newaxis] * H + rho2 * va[:, np.newaxis] * U)
    br = (0.5 * va_mat_full.T * va_inv - div_S + rho1 * div_Q 
          - va[:, np.newaxis] * K + rho2 * va[:, np.newaxis] * U.T)
    
    X = np.column_stack([A_fact(bx[:, i]) for i in range(nv)])
    R = np.column_stack([A_fact(br[:, i]) for i in range(nv)])
    
    Gx = G @ X
    Gr = G @ R
    
    # Step 2: Z, Q, U minimization
    Z = (1.0 / rho1) * Y + Gx
    Z = Z.reshape(nf, 3, nv)
    Z_norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_norms = np.maximum(Z_norms, 1.0)
    Z = Z / Z_norms
    Z = Z.reshape(3 * nf, nv)
    
    U1 = 0.5 * ((1.0 / rho2) * (H + K.T) + X + R.T)
    U = U1 - np.diag(np.diag(U1))
    U = np.maximum(U, 0)
```

**Key differences**:
- MATLAB `.* ` (elementwise) → Python `*` (always elementwise for arrays)
- MATLAB `max(1,...)` → Python `np.maximum(1.0, ...)`
- MATLAB `diag(diag(U))` → Python `np.diag(np.diag(U))`
- MATLAB `U(U<0) = 0` → Python `U = np.maximum(U, 0)`

---

## Common MATLAB → Python Patterns

### Matrix Operations

| MATLAB | Python (NumPy) |
|--------|----------------|
| `A * B` | `A @ B` (matrix mult) or `np.dot(A, B)` |
| `A .* B` | `A * B` (elementwise, always in NumPy) |
| `A'` | `A.T` (transpose) |
| `A(:)` | `A.flatten()` or `A.ravel()` |
| `A(:, i)` | `A[:, i]` |
| `A \ b` | `np.linalg.solve(A, b)` or `spsolve(A, b)` |
| `norm(v)` | `np.linalg.norm(v)` |
| `size(A, 1)` | `A.shape[0]` |
| `length(v)` | `len(v)` |

### Sparse Matrices

| MATLAB | Python (SciPy) |
|--------|----------------|
| `sparse(I,J,V)` | `csr_matrix((V, (I,J)))` |
| `spdiags(V,0,n,n)` | `diags(V)` |
| `speye(n)` | `eye(n)` or `identity(n)` |
| `A'` | `A.T` |

### Array Creation

| MATLAB | Python (NumPy) |
|--------|----------------|
| `zeros(n,m)` | `np.zeros((n, m))` |
| `ones(n,m)` | `np.ones((n, m))` |
| `repmat(A,n,m)` | `np.tile(A, (n, m))` or broadcasting |
| `1:n` | `np.arange(n)` or `range(n)` |
| `linspace(a,b,n)` | `np.linspace(a, b, n)` |

### Indexing

| MATLAB (1-indexed) | Python (0-indexed) |
|--------------------|-------------------|
| `A(1)` | `A[0]` |
| `A(1:n)` | `A[0:n]` or `A[:n]` |
| `A(end)` | `A[-1]` |
| `A([1 3 5])` | `A[[0, 2, 4]]` or `A[indices]` |

### Linear Algebra

| MATLAB | Python |
|--------|--------|
| `[L,U,P] = lu(A)` | `P, L, U = scipy.linalg.lu(A)` |
| `[L,flag,P] = chol(A,'lower')` | `from sksparse.cholmod import cholesky` |
| `x = L' \ (L \ b)` | `x = L.T \ (L \ b)` or use `factorized()` |

---

## File I/O

### Reading OFF Files

**MATLAB (`readOff.m`)**:
```matlab
function [vertices,faces] = readOff(filename)
    fid = fopen(filename,'r');
    firstLine = fgets(fid);
    [N,cnt] = fscanf(fid,'%d %d %d', 3);
    nv = N(1); nf = N(2);
    [vertices,cnt] = fscanf(fid,'%f %f %f', 3*nv);
    vertices = reshape(vertices, 3, nv)';
    [faces,cnt] = fscanf(fid,'%d %d %d %d', 4*nf);
    faces = reshape(faces, 4, nf)';
    faces = faces(:,2:end)+1;
    fclose(fid);
end
```

**Python (`utils.py`)**:
```python
def read_off(filename: str):
    with open(filename, 'r') as f:
        line = f.readline().strip()
        if not line.startswith('OFF'):
            raise ValueError("Not a valid OFF file")
        
        line = f.readline().strip()
        nv, nf, ne = map(int, line.split()[:3])
        
        vertices = np.zeros((nv, 3))
        for i in range(nv):
            line = f.readline().strip()
            vertices[i] = list(map(float, line.split()[:3]))
        
        faces = np.zeros((nf, 3), dtype=int)
        for i in range(nf):
            line = f.readline().strip()
            parts = list(map(int, line.split()))
            faces[i] = parts[1:4]
    
    return vertices, faces
```

**Key differences**:
- MATLAB uses `fopen/fscanf/fclose`, Python uses `with open()` context manager
- MATLAB 1-indexed → Python 0-indexed (no need for `+1` on faces)
- Python reads line-by-line explicitly

---

## Visualization

### MATLAB
```matlab
p = patch('Faces',mesh.faces,'Vertices',mesh.vertices);
p.FaceVertexCData = u;
p.FaceColor = 'interp';
colorbar;
axis equal; axis off;
```

### Python
```python
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

verts = mesh.vertices[mesh.faces]
collection = Poly3DCollection(verts, facecolors=colors)
ax.add_collection3d(collection)

ax.set_aspect('equal')
plt.colorbar(mappable, ax=ax)
```

---

## Performance Tips

### MATLAB
- Pre-allocate arrays
- Vectorize operations
- Use sparse matrices for large systems

### Python
- Use NumPy vectorized operations (avoid loops)
- Use `scipy.sparse` for sparse matrices
- Consider `numba.jit` for critical loops
- Use `factorized()` to cache Cholesky factorizations
- Install `scikit-sparse` for faster sparse linear algebra

---

## Summary of Main Changes

1. **Indexing**: MATLAB starts at 1, Python starts at 0
2. **Matrix multiply**: MATLAB `*`, Python `@`
3. **Transpose**: MATLAB `'`, Python `.T`
4. **Sparse**: MATLAB `sparse()`, Python `csr_matrix()`
5. **Linear solve**: MATLAB `\`, Python `spsolve()` or `factorized()`
6. **Elementwise**: MATLAB `.*`, Python `*` (always elementwise for arrays)
7. **Array creation**: MATLAB `zeros(n,m)`, Python `np.zeros((n,m))`
8. **File I/O**: MATLAB `fscanf`, Python line-by-line reading
9. **Visualization**: MATLAB `patch`, Python `Poly3DCollection`

The overall algorithm and mathematical operations remain identical - these are just syntax translations!
