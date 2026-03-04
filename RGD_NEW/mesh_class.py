"""
Mesh class for computing regularized geodesic distances
Python implementation of the MATLAB MeshClass
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import warnings


class MeshClass:
    """
    A class for triangle mesh processing and computation.
    
    Attributes:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of face indices (0-indexed)
        nv: number of vertices
        nf: number of faces
        va: vertex areas
        ta: face (triangle) areas
        Nf: face normals
        Nv: vertex normals
        G: gradient operator
        D: divergence operator
        Ww: cotangent Laplacian weights
        Lap: cotangent Laplacian operator
    """
    
    def __init__(self, vertices, faces):
        """
        Initialize mesh from vertices and faces.
        
        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices (0-indexed)
        """
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int32)
        self.compute_all()
    
    def compute_all(self):
        """Compute all mesh properties."""
        self.nv = self.vertices.shape[0]
        self.nf = self.faces.shape[0]
        
        # Compute face normals and areas
        self._compute_normals_and_areas()
        
        # Compute vertex areas
        self._compute_vertex_areas()
        
        # Compute triangle edges
        self._compute_edges()
        
        # Compute Laplacian
        self._compute_laplacian()
        
        # Compute gradient operator
        self._compute_gradient()
        
        # Compute divergence operator
        self._compute_divergence()
    
    def _compute_normals_and_areas(self):
        """Compute face normals and areas."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        # Face normals (unnormalized)
        cross_prod = np.cross(v1 - v0, v2 - v0)
        
        # Face areas
        self.ta = np.sqrt(np.sum(cross_prod**2, axis=1)) / 2.0
        
        # Normalized face normals
        norms = np.linalg.norm(cross_prod, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0  # Avoid division by zero
        self.Nf = cross_prod / norms
        
        # Compute vertex normals (area-weighted)
        self._compute_vertex_normals()
    
    def _compute_vertex_normals(self):
        """Compute vertex normals as area-weighted average of face normals."""
        self.Nv = np.zeros((self.nv, 3))
        for i in range(3):
            # Accumulate area-weighted face normals to vertices
            for j in range(3):
                np.add.at(self.Nv[:, i], self.faces[:, j], 
                         self.Nf[:, i] * self.ta)
        
        # Normalize
        norms = np.linalg.norm(self.Nv, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        self.Nv = self.Nv / norms
    
    def _compute_vertex_areas(self):
        """Compute vertex areas using barycentric area."""
        self.va = np.zeros(self.nv)
        for i in range(3):
            np.add.at(self.va, self.faces[:, i], self.ta / 3.0)
    
    def _compute_edges(self):
        """Compute edge information."""
        # Triangle edges
        self.E1 = self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 2]]
        self.E2 = self.vertices[self.faces[:, 0]] - self.vertices[self.faces[:, 2]]
        self.E3 = self.vertices[self.faces[:, 0]] - self.vertices[self.faces[:, 1]]
    
    def _compute_laplacian(self):
        """Compute cotangent Laplacian."""
        # Edge lengths
        L1 = np.linalg.norm(self.E1, axis=1)
        L2 = np.linalg.norm(self.E2, axis=1)
        L3 = np.linalg.norm(self.E3, axis=1)
        
        # Angles using law of cosines
        A1 = (L2**2 + L3**2 - L1**2) / (2 * L2 * L3)
        A2 = (L1**2 + L3**2 - L2**2) / (2 * L1 * L3)
        A3 = (L1**2 + L2**2 - L3**2) / (2 * L1 * L2)
        
        # Clip to avoid numerical issues with arccos
        A1 = np.clip(A1, -1.0, 1.0)
        A2 = np.clip(A2, -1.0, 1.0)
        A3 = np.clip(A3, -1.0, 1.0)
        
        A1 = np.arccos(A1)
        A2 = np.arccos(A2)
        A3 = np.arccos(A3)
        
        # Cotangent weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C1 = 0.5 * np.cos(A3) / np.sin(A3)  # cot(A3)
            C2 = 0.5 * np.cos(A1) / np.sin(A1)  # cot(A1)
            C3 = 0.5 * np.cos(A2) / np.sin(A2)  # cot(A2)
        
        # Handle infinities and NaNs
        C1 = np.nan_to_num(C1, nan=0.0, posinf=0.0, neginf=0.0)
        C2 = np.nan_to_num(C2, nan=0.0, posinf=0.0, neginf=0.0)
        C3 = np.nan_to_num(C3, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Build sparse Laplacian
        I = np.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
        J = np.concatenate([self.faces[:, 1], self.faces[:, 2], self.faces[:, 0]])
        S = np.concatenate([C3, C1, C2])
        
        # Create sparse matrix
        In = np.concatenate([I, J, I, J])
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([-S, -S, S, S])
        
        self.Ww = sparse.csr_matrix((Sn, (In, Jn)), shape=(self.nv, self.nv))
        
        # Normalized Laplacian
        va_inv = sparse.diags(1.0 / self.va, 0)
        self.Lap = va_inv @ self.Ww
        
        # Area matrix
        self.Aa = sparse.diags(self.va, 0)
    
    def _compute_gradient(self):
        """Compute gradient operator on vertex functions."""
        # Rotated edges for gradient computation
        RE1 = self.rotate_vf(self.E1)
        RE2 = self.rotate_vf(self.E2)
        RE3 = self.rotate_vf(self.E3)
        
        # Build gradient operator (3*nf x nv)
        I = np.tile(np.arange(self.nf), 3)
        II = np.concatenate([I, I + self.nf, I + 2*self.nf])
        
        J = self.faces.T.flatten()
        JJ = np.tile(J, 3)
        
        # Values for gradient operator
        ta_inv = 1.0 / (2.0 * self.ta)
        S = np.concatenate([
            RE2[:, 0] * ta_inv, RE3[:, 0] * ta_inv, RE1[:, 0] * ta_inv,
            RE2[:, 1] * ta_inv, RE3[:, 1] * ta_inv, RE1[:, 1] * ta_inv,
            RE2[:, 2] * ta_inv, RE3[:, 2] * ta_inv, RE1[:, 2] * ta_inv
        ])
        
        self.G = sparse.csr_matrix((S, (II, JJ)), shape=(3*self.nf, self.nv))
    
    def _compute_divergence(self):
        """Compute divergence operator on face vector fields."""
        # Divergence is computed as div(v) = -G^T * (ta .* v) for face fields
        # where ta are the face areas (repeated 3 times for x,y,z components)
        # This is the L2 adjoint of the gradient
        
        # Note: We'll compute divergence in the ADMM algorithm directly
        # since it needs the ta weighting applied to the vector field first
        self.D = None  # Placeholder - computed on-the-fly in ADMM
    
    def rotate_vf(self, vf):
        """
        Rotate vector field by 90 degrees in the tangent plane.
        
        Args:
            vf: Nx3 array of vectors
            
        Returns:
            Rotated vectors (cross product with face normal)
        """
        if len(vf.shape) == 1:
            vf = vf.reshape(-1, 3)
        return np.cross(self.Nf, vf)
    
    def normalize_vf(self, vf):
        """
        Normalize vector field.
        
        Args:
            vf: Nx3 array of vectors
            
        Returns:
            Normalized vectors
        """
        if len(vf.shape) == 1:
            vf = vf.reshape(-1, 3)
        
        norms = np.linalg.norm(vf, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return vf / norms
    
    def normv(self, vf):
        """
        Compute norms of vector field.
        
        Args:
            vf: Nx3 array of vectors
            
        Returns:
            Array of norms
        """
        if len(vf.shape) == 1:
            vf = vf.reshape(-1, 3)
        return np.linalg.norm(vf, axis=1)
    
    def barycenter(self):
        """Compute face barycenters."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return (v0 + v1 + v2) / 3.0
    
    @staticmethod
    def read_off(filename):
        """
        Read mesh from OFF file.
        
        Args:
            filename: Path to OFF file
            
        Returns:
            vertices, faces arrays
        """
        with open(filename, 'r') as f:
            # Read header
            line = f.readline().strip()
            if line != 'OFF':
                raise ValueError('Not a valid OFF file')
            
            # Read counts
            line = f.readline().strip()
            while line.startswith('#') or len(line) == 0:
                line = f.readline().strip()
            
            counts = [int(x) for x in line.split()]
            nv, nf = counts[0], counts[1]
            
            # Read vertices
            vertices = []
            for i in range(nv):
                line = f.readline().strip()
                vertices.append([float(x) for x in line.split()])
            
            # Read faces
            faces = []
            for i in range(nf):
                line = f.readline().strip()
                parts = [int(x) for x in line.split()]
                if parts[0] != 3:
                    raise ValueError('Only triangle meshes supported')
                faces.append(parts[1:4])
        
        return np.array(vertices), np.array(faces)
