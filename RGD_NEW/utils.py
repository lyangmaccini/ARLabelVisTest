"""
Utility functions for RGD
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def smooth_vf(mesh, vf, num_iterations=2):
    """
    Smooth a vector field on the mesh using diffusion.
    
    Args:
        mesh: MeshClass object
        vf: Vector field on faces (nf x 3)
        num_iterations: Number of smoothing iterations
        
    Returns:
        Smoothed vector field
    """
    if vf.shape[0] != mesh.nf or vf.shape[1] != 3:
        raise ValueError(f"Vector field must be shape (nf, 3), got {vf.shape}")
    
    # Create face-to-face diffusion operator
    # This is a simplified version - proper implementation would need face adjacency
    
    # For now, use vertex-based smoothing as approximation:
    # 1. Interpolate to vertices
    # 2. Smooth on vertices
    # 3. Interpolate back to faces
    
    vf_smooth = vf.copy()
    
    for _ in range(num_iterations):
        # Convert to vertex field
        vf_v = np.zeros((mesh.nv, 3))
        vf_count = np.zeros(mesh.nv)
        
        for i in range(mesh.nf):
            for j in range(3):
                v_idx = mesh.faces[i, j]
                vf_v[v_idx] += vf_smooth[i] * mesh.ta[i]
                vf_count[v_idx] += mesh.ta[i]
        
        # Average
        vf_count[vf_count < 1e-10] = 1.0
        vf_v = vf_v / vf_count[:, np.newaxis]
        
        # Smooth on vertices using Laplacian
        dt = 0.01
        for dim in range(3):
            vf_v[:, dim] = vf_v[:, dim] - dt * mesh.Lap @ vf_v[:, dim]
        
        # Convert back to faces
        vf_smooth = np.zeros((mesh.nf, 3))
        for i in range(mesh.nf):
            for j in range(3):
                v_idx = mesh.faces[i, j]
                vf_smooth[i] += vf_v[v_idx] / 3.0
    
    return vf_smooth


def interpolate_vertices_to_faces(mesh, vertex_values):
    """
    Interpolate vertex values to face values.
    
    Args:
        mesh: MeshClass object
        vertex_values: Values at vertices (nv,) or (nv, k)
        
    Returns:
        Face values (nf,) or (nf, k)
    """
    if vertex_values.ndim == 1:
        vertex_values = vertex_values.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    nv, k = vertex_values.shape
    face_values = np.zeros((mesh.nf, k))
    
    for i in range(mesh.nf):
        face_values[i] = np.mean(vertex_values[mesh.faces[i]], axis=0)
    
    if squeeze:
        face_values = face_values.squeeze()
    
    return face_values


def interpolate_faces_to_vertices(mesh, face_values):
    """
    Interpolate face values to vertex values using area weighting.
    
    Args:
        mesh: MeshClass object
        face_values: Values at faces (nf,) or (nf, k)
        
    Returns:
        Vertex values (nv,) or (nv, k)
    """
    if face_values.ndim == 1:
        face_values = face_values.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    nf, k = face_values.shape
    vertex_values = np.zeros((mesh.nv, k))
    
    # Area-weighted accumulation
    for i in range(mesh.nf):
        for j in range(3):
            v_idx = mesh.faces[i, j]
            vertex_values[v_idx] += face_values[i] * mesh.ta[i] / 3.0
    
    # Normalize by vertex area
    vertex_values = vertex_values / mesh.va[:, np.newaxis]
    
    if squeeze:
        vertex_values = vertex_values.squeeze()
    
    return vertex_values


def compute_geodesic_gaussian(mesh, source_vertices, sigma_squared):
    """
    Compute a geodesic Gaussian centered at source vertices.
    
    Args:
        mesh: MeshClass object
        source_vertices: Source vertex indices
        sigma_squared: Variance of Gaussian
        
    Returns:
        Gaussian values at all vertices
    """
    from rgd_admm import rgd_admm
    
    # Compute geodesic distance from sources
    dist = rgd_admm(mesh, source_vertices, reg='D', alpha_hat=0.0, quiet=True)[0]
    
    # Apply Gaussian kernel
    gaussian = np.exp(-dist**2 / (2 * sigma_squared))
    
    return gaussian


def normalize_mesh(vertices, target_scale=1.0):
    """
    Normalize mesh to have specified bounding box diagonal.
    
    Args:
        vertices: Nx3 vertex array
        target_scale: Target bounding box diagonal
        
    Returns:
        Normalized vertices
    """
    vertices = vertices - np.mean(vertices, axis=0)
    bbox_diag = np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))
    vertices = vertices / bbox_diag * target_scale
    return vertices
