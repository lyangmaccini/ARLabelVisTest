"""
Mesh Class - Triangle Mesh Data Structure with Differential Geometry Operators

This module provides the Mesh class which stores triangle mesh geometry and
computes necessary differential operators for geodesic distance computation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, diags
from typing import Tuple, Optional, Union
import os


class Mesh:
    """
    Triangle mesh with differential geometry operators.
    
    Attributes:
        vertices (np.ndarray): Vertex positions (nv x 3)
        faces (np.ndarray): Face connectivity (nf x 3), 0-indexed
        nv (int): Number of vertices
        nf (int): Number of faces
        ne (int): Number of edges
        
        va (np.ndarray): Vertex areas (nv,)
        ta (np.ndarray): Triangle/face areas (nf,)
        
        Nf (np.ndarray): Face normals (nf x 3)
        Nv (np.ndarray): Vertex normals (nv x 3)
        
        G (csr_matrix): Gradient operator (3*nf x nv)
        Ww (csr_matrix): Cotangent Laplacian weights (nv x nv)
        Lap (csr_matrix): Normalized Laplacian (nv x nv)
        
        edges (np.ndarray): Unique edges (ne x 2)
        inner_edges (np.ndarray): Interior edge indices
        boundary_vertices (np.ndarray): Boundary vertex indices
    """
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, name: str = "mesh"):
        """
        Initialize mesh from vertices and faces.
        
        Args:
            vertices: Vertex positions (nv x 3)
            faces: Face connectivity (nf x 3), 0-indexed
            name: Mesh name
        """
        self.name = name
        self.vertices = vertices.astype(float)
        self.faces = faces.astype(int)
        
        # Basic counts
        self.nv = len(vertices)
        self.nf = len(faces)
        
        # Compute all geometric quantities
        self._compute_all()
    
    @classmethod
    def from_file(cls, filename: str) -> 'Mesh':
        """
        Load mesh from .OFF file.
        
        Args:
            filename: Path to .OFF file (with or without extension)
            
        Returns:
            Mesh object
        """
        from .utils import read_off
        
        if not filename.endswith('.off'):
            filename = filename + '.off'
        
        vertices, faces = read_off(filename)
        name = os.path.basename(filename).replace('.off', '')
        return cls(vertices, faces, name)
    
    def _compute_all(self):
        """Compute all geometric quantities."""
        # Face normals and areas
        v1 = self.vertices[self.faces[:, 0]]
        v2 = self.vertices[self.faces[:, 1]]
        v3 = self.vertices[self.faces[:, 2]]
        
        cross_prod = np.cross(v1 - v2, v1 - v3)
        self.ta = np.linalg.norm(cross_prod, axis=1) / 2  # Face areas
        self.Nf = cross_prod / (2 * self.ta[:, np.newaxis])  # Face normals
        
        # Vertex normals (area-weighted average of adjacent face normals)
        self.Nv = self._compute_vertex_normals()
        
        # Vertex areas (1/3 of adjacent face areas)
        self.va = self._compute_vertex_areas()
        
        # Edge information
        self.edges, self.inner_edges, self.boundary_vertices = self._compute_edges()
        self.ne = len(self.edges)
        
        # Differential operators
        self.G = self._compute_gradient_operator()
        self.Ww = self._compute_cotangent_laplacian()
        self.Lap = diags(1.0 / self.va) @ self.Ww
        
        # Edge basis for vector fields (used in some regularizations)
        self._compute_edge_basis()
    
    def _compute_vertex_normals(self) -> np.ndarray:
        """Compute area-weighted vertex normals."""
        normals = np.zeros((self.nv, 3))
        
        for i in range(3):
            np.add.at(normals, self.faces[:, i], 
                     self.Nf * self.ta[:, np.newaxis])
        
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return normals / norms
    
    def _compute_vertex_areas(self) -> np.ndarray:
        """Compute vertex areas (barycentric dual areas)."""
        va = np.zeros(self.nv)
        
        for i in range(3):
            np.add.at(va, self.faces[:, i], self.ta / 3)
        
        return va
    
    def _compute_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute edge information.
        
        Returns:
            edges: Unique edges (ne x 2)
            inner_edges: Indices of interior edges
            boundary_vertices: Indices of boundary vertices
        """
        # Collect all edges
        all_edges = np.vstack([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]
        ])
        
        # Sort each edge to make unique
        all_edges_sorted = np.sort(all_edges, axis=1)
        
        # Find unique edges and their counts
        unique_edges, counts = np.unique(all_edges_sorted, axis=0, 
                                         return_counts=True)
        
        # Interior edges appear in 2 faces, boundary edges in 1
        inner_mask = counts == 2
        inner_edges = np.where(inner_mask)[0]
        
        # Boundary vertices are endpoints of boundary edges
        boundary_edges = unique_edges[~inner_mask]
        boundary_vertices = np.unique(boundary_edges.flatten())
        
        return unique_edges, inner_edges, boundary_vertices
    
    def _compute_gradient_operator(self) -> csr_matrix:
        """
        Compute gradient operator G that maps vertex functions to face gradients.
        
        For each triangle, computes ∇φᵢ for each vertex i, where φᵢ is the
        barycentric coordinate function.
        
        Returns:
            G: Sparse gradient operator (3*nf x nv)
        """
        nf = self.nf
        
        # Get triangle vertices
        v1 = self.vertices[self.faces[:, 0]]
        v2 = self.vertices[self.faces[:, 1]]
        v3 = self.vertices[self.faces[:, 2]]
        
        # Edge vectors
        e1 = v2 - v3  # Opposite to vertex 1
        e2 = v3 - v1  # Opposite to vertex 2
        e3 = v1 - v2  # Opposite to vertex 3
        
        # Rotate edges by 90° in face plane (cross with normal)
        re1 = np.cross(self.Nf, e1)
        re2 = np.cross(self.Nf, e2)
        re3 = np.cross(self.Nf, e3)
        
        # Gradient = rotated_edge / (2 * area)
        grad1 = re1 / (2 * self.ta[:, np.newaxis])
        grad2 = re2 / (2 * self.ta[:, np.newaxis])
        grad3 = re3 / (2 * self.ta[:, np.newaxis])
        
        # Build sparse matrix
        # G has shape (3*nf, nv) where rows are [x-grads, y-grads, z-grads]
        row_idx = []
        col_idx = []
        values = []
        
        for dim in range(3):
            face_offset = dim * nf
            
            # Vertex 1 contribution
            row_idx.extend(range(face_offset, face_offset + nf))
            col_idx.extend(self.faces[:, 0])
            values.extend(grad1[:, dim])
            
            # Vertex 2 contribution
            row_idx.extend(range(face_offset, face_offset + nf))
            col_idx.extend(self.faces[:, 1])
            values.extend(grad2[:, dim])
            
            # Vertex 3 contribution
            row_idx.extend(range(face_offset, face_offset + nf))
            col_idx.extend(self.faces[:, 2])
            values.extend(grad3[:, dim])
        
        G = csr_matrix((values, (row_idx, col_idx)), 
                       shape=(3 * nf, self.nv))
        
        return G
    
    def _compute_cotangent_laplacian(self) -> csr_matrix:
        """
        Compute cotangent Laplacian (symmetric weights only).
        
        Returns:
            Ww: Cotangent Laplacian weights (nv x nv)
        """
        # Get edge lengths
        v1 = self.vertices[self.faces[:, 0]]
        v2 = self.vertices[self.faces[:, 1]]
        v3 = self.vertices[self.faces[:, 2]]
        
        L1 = np.linalg.norm(v2 - v3, axis=1)  # Opposite vertex 1
        L2 = np.linalg.norm(v3 - v1, axis=1)  # Opposite vertex 2
        L3 = np.linalg.norm(v1 - v2, axis=1)  # Opposite vertex 3
        
        # Compute angles using law of cosines
        A1 = np.arccos((L2**2 + L3**2 - L1**2) / (2 * L2 * L3 + 1e-10))
        A2 = np.arccos((L1**2 + L3**2 - L2**2) / (2 * L1 * L3 + 1e-10))
        A3 = np.arccos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2 + 1e-10))
        
        # Cotangent weights
        cot_A1 = 1.0 / np.tan(A1)
        cot_A2 = 1.0 / np.tan(A2)
        cot_A3 = 1.0 / np.tan(A3)
        
        # Build sparse matrix
        I_list = []
        J_list = []
        V_list = []
        
        # Edge (v1, v2) has weight cot(A3)
        I_list.extend(self.faces[:, 0])
        J_list.extend(self.faces[:, 1])
        V_list.extend(0.5 * cot_A3)
        
        # Edge (v2, v3) has weight cot(A1)
        I_list.extend(self.faces[:, 1])
        J_list.extend(self.faces[:, 2])
        V_list.extend(0.5 * cot_A1)
        
        # Edge (v3, v1) has weight cot(A2)
        I_list.extend(self.faces[:, 2])
        J_list.extend(self.faces[:, 0])
        V_list.extend(0.5 * cot_A2)
        
        # Make symmetric (add transpose)
        I_sym = I_list + J_list
        J_sym = J_list + I_list
        V_sym = V_list + V_list
        
        # Create off-diagonal part
        W_off = csr_matrix((V_sym, (I_sym, J_sym)), 
                           shape=(self.nv, self.nv))
        
        # Diagonal is negative sum of off-diagonals
        W_diag = -W_off.sum(axis=1).A1
        W = W_off + diags(W_diag)
        
        return W.tocsr()
    
    def _compute_edge_basis(self):
        """Compute edge-based orthonormal basis for vector fields on faces."""
        # E1 is edge from v2 to v3
        v2 = self.vertices[self.faces[:, 1]]
        v3 = self.vertices[self.faces[:, 2]]
        E1 = v2 - v3
        
        # Normalize to get first basis vector
        E1_norm = np.linalg.norm(E1, axis=1, keepdims=True)
        E1_norm[E1_norm < 1e-15] = 1.0
        self.F1 = E1 / E1_norm
        
        # Second basis vector is normal cross first
        self.F2 = np.cross(self.Nf, self.F1)
        
        # Build projection matrices
        nf = self.nf
        I = np.repeat(np.arange(nf), 3)
        J = np.concatenate([np.arange(nf), 
                           np.arange(nf, 2*nf), 
                           np.arange(2*nf, 3*nf)])
        
        vals_F1 = np.concatenate([self.F1[:, 0], self.F1[:, 1], self.F1[:, 2]])
        vals_F2 = np.concatenate([self.F2[:, 0], self.F2[:, 1], self.F2[:, 2]])
        
        B1 = csr_matrix((vals_F1, (I, J)), shape=(nf, 3*nf))
        B2 = csr_matrix((vals_F2, (I, J)), shape=(nf, 3*nf))
        
        self.EB = sparse.vstack([B1, B2])  # 2*nf x 3*nf
        self.EBI = self.EB.T  # 3*nf x 2*nf
    
    def interpolate_face_to_vertex(self, face_data: np.ndarray) -> np.ndarray:
        """
        Interpolate data from faces to vertices (area-weighted average).
        
        Args:
            face_data: Data on faces (nf,) or (nf, d)
            
        Returns:
            vertex_data: Data on vertices (nv,) or (nv, d)
        """
        if face_data.ndim == 1:
            face_data = face_data[:, np.newaxis]
        
        vertex_data = np.zeros((self.nv, face_data.shape[1]))
        
        for i in range(3):
            weighted = face_data * (self.ta[:, np.newaxis] / 3)
            np.add.at(vertex_data, self.faces[:, i], weighted)
        
        vertex_data = vertex_data / self.va[:, np.newaxis]
        
        return vertex_data.squeeze()
    
    def interpolate_vertex_to_face(self, vertex_data: np.ndarray) -> np.ndarray:
        """
        Interpolate data from vertices to faces (barycentric average).
        
        Args:
            vertex_data: Data on vertices (nv,) or (nv, d)
            
        Returns:
            face_data: Data on faces (nf,) or (nf, d)
        """
        if vertex_data.ndim == 1:
            vertex_data = vertex_data[:, np.newaxis]
        
        v1_data = vertex_data[self.faces[:, 0]]
        v2_data = vertex_data[self.faces[:, 1]]
        v3_data = vertex_data[self.faces[:, 2]]
        
        face_data = (v1_data + v2_data + v3_data) / 3
        
        return face_data.squeeze()
    
    def barycenters(self) -> np.ndarray:
        """
        Compute face barycenters.
        
        Returns:
            centers: Barycenter positions (nf x 3)
        """
        v1 = self.vertices[self.faces[:, 0]]
        v2 = self.vertices[self.faces[:, 1]]
        v3 = self.vertices[self.faces[:, 2]]
        
        return (v1 + v2 + v3) / 3
    
    def normalize_mesh(self, target_bbox_size: float = 1.0):
        """
        Center and normalize mesh to fit in unit bounding box.
        
        Args:
            target_bbox_size: Target bounding box diagonal
        """
        # Center
        center = self.vertices.mean(axis=0)
        self.vertices = self.vertices - center
        
        # Scale to target size
        bbox_min = self.vertices.min(axis=0)
        bbox_max = self.vertices.max(axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        
        if bbox_diag > 1e-10:
            self.vertices = self.vertices * (target_bbox_size / bbox_diag)
        
        # Recompute geometry
        self._compute_all()
    
    def center_mesh(self):
        """Center mesh at origin."""
        center = self.vertices.mean(axis=0)
        self.vertices = self.vertices - center
        self._compute_all()
    
    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length.
        
        Args:
            vectors: Array of vectors (n x d)
            
        Returns:
            normalized: Unit vectors (n x d)
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return vectors / norms
    
    def __repr__(self) -> str:
        return (f"Mesh('{self.name}', vertices={self.nv}, faces={self.nf}, "
                f"edges={self.ne}, area={self.ta.sum():.3f})")
