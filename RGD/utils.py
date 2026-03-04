"""
Utility Functions for Mesh Processing and Regularized Geodesic Distances
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional


def read_off(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a mesh from an OFF file.
    
    Args:
        filename: Path to .OFF file
        
    Returns:
        vertices: Vertex positions (nv x 3)
        faces: Face connectivity (nf x 3), 0-indexed
    """
    with open(filename, 'r') as f:
        # Read header
        line = f.readline().strip()
        if not line.startswith('OFF'):
            raise ValueError(f"Not a valid OFF file: {filename}")
        
        # Read counts
        line = f.readline().strip()
        while line.startswith('#') or not line:
            line = f.readline().strip()
        
        nv, nf, ne = map(int, line.split()[:3])
        
        # Read vertices
        vertices = np.zeros((nv, 3))
        for i in range(nv):
            line = f.readline().strip()
            vertices[i] = list(map(float, line.split()[:3]))
        
        # Read faces
        faces = np.zeros((nf, 3), dtype=int)
        for i in range(nf):
            line = f.readline().strip()
            parts = list(map(int, line.split()))
            n_verts = parts[0]
            if n_verts != 3:
                raise ValueError(f"Only triangle meshes supported, face {i} has {n_verts} vertices")
            faces[i] = parts[1:4]
    
    return vertices, faces


def write_off(filename: str, vertices: np.ndarray, faces: np.ndarray):
    """
    Write a mesh to an OFF file.
    
    Args:
        filename: Output file path
        vertices: Vertex positions (nv x 3)
        faces: Face connectivity (nf x 3), 0-indexed
    """
    nv = len(vertices)
    nf = len(faces)
    
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write(f"{nv} {nf} 0\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def compute_cotangent_laplacian(vertices: np.ndarray, 
                                faces: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    """
    Compute cotangent Laplacian for a triangle mesh.
    
    Args:
        vertices: Vertex positions (nv x 3)
        faces: Face connectivity (nf x 3)
        
    Returns:
        W: Cotangent Laplacian (nv x nv)
        A: Vertex areas (nv,)
    """
    nv = len(vertices)
    nf = len(faces)
    
    # Get triangle vertices
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    
    # Edge lengths
    L1 = np.linalg.norm(v2 - v3, axis=1)
    L2 = np.linalg.norm(v3 - v1, axis=1)
    L3 = np.linalg.norm(v1 - v2, axis=1)
    
    # Angles
    A1 = np.arccos(np.clip((L2**2 + L3**2 - L1**2) / (2 * L2 * L3 + 1e-10), -1, 1))
    A2 = np.arccos(np.clip((L1**2 + L3**2 - L2**2) / (2 * L1 * L3 + 1e-10), -1, 1))
    A3 = np.arccos(np.clip((L1**2 + L2**2 - L3**2) / (2 * L1 * L2 + 1e-10), -1, 1))
    
    # Triangle areas
    s = (L1 + L2 + L3) / 2
    ta = np.sqrt(s * (s - L1) * (s - L2) * (s - L3))
    
    # Cotangent weights
    I_list = []
    J_list = []
    V_list = []
    
    for i, j, cot_angle in [(0, 1, A3), (1, 2, A1), (2, 0, A2)]:
        cot_vals = 0.5 / np.tan(cot_angle)
        
        I_list.extend(faces[:, i])
        J_list.extend(faces[:, j])
        V_list.extend(cot_vals)
    
    # Symmetrize
    I_sym = I_list + J_list
    J_sym = J_list + I_list
    V_sym = V_list + V_list
    
    W_off = csr_matrix((V_sym, (I_sym, J_sym)), shape=(nv, nv))
    W_diag = -W_off.sum(axis=1).A1
    W = W_off + diags(W_diag)
    
    # Vertex areas (barycentric)
    A = np.zeros(nv)
    for i in range(3):
        np.add.at(A, faces[:, i], ta / 3)
    
    return W.tocsr(), A


def smooth_vector_field(mesh, 
                       sparse_vf: np.ndarray,
                       power: int = 2,
                       constraint_faces: Optional[np.ndarray] = None,
                       constraint_values: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Smooth/interpolate a sparse vector field across the mesh using power fields.
    
    This is useful for vector field alignment regularization where you have
    a few direction constraints and want to smoothly interpolate across the mesh.
    
    Based on "PH-CPF: Planar Hexagonal Meshing using Coordinate Power Fields" 
    by Pluta et al., 2021.
    
    Args:
        mesh: Mesh object
        sparse_vf: Sparse vector field on faces (nf x 3), most entries are zero
        power: Power for the field (2 for line fields)
        constraint_faces: Optional face indices with constraints
        constraint_values: Optional constraint values (nc x 3)
        
    Returns:
        smooth_vf: Smoothed vector field on all faces (nf x 3)
    """
    nf = mesh.nf
    
    # If constraint faces/values provided explicitly, use those
    if constraint_faces is not None and constraint_values is not None:
        vf_sparse = np.zeros((nf, 3))
        vf_sparse[constraint_faces] = constraint_values
    else:
        vf_sparse = sparse_vf
    
    # Find non-zero locations (constraints)
    vf_norms = np.linalg.norm(vf_sparse, axis=1)
    locs = np.where(vf_norms > 1e-5)[0]
    
    if len(locs) == 0:
        # No constraints, return zero field
        return np.zeros((nf, 3))
    
    # Project to local edge basis and raise to power n
    # This makes the field rotation-invariant modulo n rotations
    vf_2d = mesh.EB @ vf_sparse.flatten()  # Project to 2D local coords
    vf_2d = vf_2d.reshape(2, nf).T
    
    # Convert to complex numbers and raise to power
    vf_complex = vf_2d[:, 0] + 1j * vf_2d[:, 1]
    vf_powered = vf_complex ** power
    
    # Convert back to 2D
    vf_2d_powered = np.column_stack([vf_powered.real, vf_powered.imag])
    
    # Set up constraints
    nl = len(locs)
    Aeq = sparse.lil_matrix((2 * nl, 2 * nf))
    beq = np.zeros(2 * nl)
    
    for i, loc in enumerate(locs):
        Aeq[i, loc] = 1.0
        Aeq[i + nl, loc + nf] = 1.0
        beq[i] = vf_2d_powered[loc, 0]
        beq[i + nl] = vf_2d_powered[loc, 1]
    
    Aeq = Aeq.tocsr()
    
    # Get smoothness operator (based on connection Laplacian)
    C = _compute_connection_laplacian(mesh, power)
    
    # Solve constrained least squares: min ||C*x||^2 s.t. Aeq*x = beq
    # Use normal equations: (C^T*C + lambda*Aeq^T*Aeq)*x = lambda*Aeq^T*beq
    lambda_constraint = 1e6
    
    A = C.T @ C + lambda_constraint * (Aeq.T @ Aeq)
    b = lambda_constraint * (Aeq.T @ beq)
    
    x = spsolve(A, b)
    
    # Take nth root to get back original field
    vf_2d_result = x.reshape(2, nf).T
    vf_complex_result = vf_2d_result[:, 0] + 1j * vf_2d_result[:, 1]
    
    # Compute nth root
    magnitude = np.abs(vf_complex_result)
    angle = np.angle(vf_complex_result)
    vf_complex_final = magnitude ** (1/power) * np.exp(1j * angle / power)
    
    vf_2d_final = np.column_stack([vf_complex_final.real, vf_complex_final.imag])
    
    # Project back to 3D using edge basis
    vf_flat = mesh.EBI @ vf_2d_final.flatten()
    vf_3d = vf_flat.reshape(nf, 3)
    
    # Normalize
    vf_norms = np.linalg.norm(vf_3d, axis=1, keepdims=True)
    vf_norms[vf_norms < 1e-15] = 1.0
    vf_3d = vf_3d / vf_norms
    
    return vf_3d


def _compute_connection_laplacian(mesh, n: int) -> csr_matrix:
    """
    Compute connection Laplacian for vector field smoothing.
    
    Args:
        mesh: Mesh object
        n: Power (2 for line fields)
        
    Returns:
        C: Connection Laplacian operator (2*nie x 2*nf)
    """
    inner = mesh.inner_edges
    nie = len(inner)
    nf = mesh.nf
    
    # Get edges
    edges = mesh.edges[inner]
    
    # Find triangles adjacent to each interior edge
    # This is a simplified version - full implementation would need edge-to-triangle map
    # For now, we'll use a simplified smoothness operator
    
    # Build simple smoothness operator based on face adjacency
    # Penalize differences between adjacent faces
    
    # Build face adjacency
    face_adj = _build_face_adjacency(mesh)
    
    # For each adjacent pair, add smoothness constraint
    I_list = []
    J_list = []
    V_list = []
    
    row = 0
    for f1, f2 in face_adj:
        # Add constraint that vf[f1] ≈ vf[f2] (rotated appropriately)
        I_list.extend([row, row])
        J_list.extend([f1, f2])
        V_list.extend([1, -1])
        
        I_list.extend([row + nie, row + nie])
        J_list.extend([f1 + nf, f2 + nf])
        V_list.extend([1, -1])
        
        row += 1
    
    C = csr_matrix((V_list, (I_list, J_list)), shape=(2 * nie, 2 * nf))
    
    # Weight by edge areas
    ea = mesh.va[edges].mean(axis=1)  # Approximate edge areas
    W = diags(np.concatenate([np.sqrt(ea), np.sqrt(ea)]))
    
    return W @ C


def _build_face_adjacency(mesh) -> list:
    """Build list of adjacent face pairs."""
    adjacency = []
    
    # Use edges to find adjacent faces
    edge_to_faces = {}
    
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    
    # Find pairs
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            adjacency.append(tuple(faces))
    
    return adjacency


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length.
    
    Args:
        vectors: Array of vectors (n x d)
        
    Returns:
        Unit vectors (n x d)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return vectors / norms


def geodesic_gaussian(mesh, 
                     source_vertices: np.ndarray,
                     sigma_squared: Optional[float] = None) -> np.ndarray:
    """
    Compute a geodesic Gaussian centered at source vertices.
    
    Useful for localizing vector fields or other operations.
    
    Args:
        mesh: Mesh object
        source_vertices: Vertex indices to center at
        sigma_squared: Variance (default: total_area / 100)
        
    Returns:
        gaussian: Geodesic Gaussian values on vertices (nv,)
    """
    from .rgd_admm import rgd_admm
    
    if sigma_squared is None:
        sigma_squared = mesh.ta.sum() / 100
    
    # Compute geodesic distance from sources
    dist = rgd_admm(mesh, source_vertices, alpha_hat=0.0)
    
    # Apply Gaussian
    gaussian = np.exp(-dist**2 / (2 * sigma_squared))
    
    return gaussian
