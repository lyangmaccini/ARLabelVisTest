import numpy as np
import alphashape
import trimesh
import torch
from tqdm import tqdm
from scipy.optimize import minimize
import logging
import sys
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_edge_loss, mesh_normal_consistency
from scipy.spatial import cKDTree

def pointsToMesh(allLABs):
    # Converts given points to a trimesh
    # allLABs: RGB points at each step size converted into LAB points; size nx3

    alpha = 0.05 # hyperparameter
    shape = alphashape.alphashape(allLABs, alpha=alpha)
    mesh = trimesh.Trimesh(vertices=np.array(shape.vertices), faces=shape.faces)

    if not mesh.is_watertight:
        print("WARNING WARNING: LAB mesh is not watertight")
    # do we want to remesh here at all? will that affect the optimization?
    return mesh

def insideMesh(point: np.array, mesh: trimesh.Trimesh):
    # point should be a 1x3 numpy array
    # Returns whether point is on or within the mesh
    # print(point)
    containment = mesh.contains([point])
    # still need to check for points on the mesh surface 
    # print(containment.shape)
    return containment[0]


def meshVolume(vertices: torch.Tensor, faces: torch.Tensor):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = torch.cross(v1, v2, dim=1)
    volume = torch.sum(v0 * cross) / 6.0
    return torch.abs(volume)

class ColorSpaceOptimizer:
    def __init__(self, trimesh: trimesh.Trimesh):
        self.mesh = trimesh
        self.original_mesh = trimesh.copy()
        # print(len(self.mesh.vertices))
        self.iterations = 5

        # Containment:
        self.containment_weight = 1.0

        # Curvature:
        self.curvature_weight = 1.0
        self.elliptic_penalty = 2.0
        self.hyperbolic_penalty = 1.0
        self.elliptic_threshold = 1.0
        self.hyperbolic_threshold = 1.0
        self.threshold_penalty = 1.0

        # Volume:
        self.volume_weight = 1.0

    def containment(self, mesh: trimesh.Trimesh):
        # loss = 0
        all_containment = self.original_mesh.contains(mesh.vertices).astype(np.float64)
        all_containment = 1.0 - all_containment
        # for vert in mesh.vertices:
            # loss += 1.0 - float(insideMesh(vert, self.original_mesh))

        # why is this immediately high 
        return np.sum(all_containment)
    
    def curvature(self, mesh: trimesh.Trimesh):
        # elliptic_penalty and hyperbolic_penalty are hyperparameters; elliptic_penalty should be larger than hyperbolic_penalty
        # threshold is a hyperparameter representing the maximum curvature to allow
        # should threshold be imposed before or after penalty? could incorporate relative difference here
        # could also add a separate average penalty outside curvature weight
        curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0)
        elliptic_max = -1 * np.min(curvatures)
        hyperbolic_max = np.max(curvatures)
        curvatures[curvatures < 0] *= self.elliptic_penalty
        curvatures[curvatures > 0] *= self.hyperbolic_penalty
        total_curvature = np.sum(curvatures ** 2)
        if elliptic_max > self.elliptic_threshold or hyperbolic_max > self.hyperbolic_threshold:
            total_curvature += self.threshold_penalty
        return total_curvature 
            
    
    def volume(self, mesh: trimesh.Trimesh):
        return mesh.volume
    
    def loss_fn(self, vertices):
        # print(self.containment(mesh))
        # print(self.curvature(mesh))
        # print(self.volume(mesh))
        mesh = trimesh.Trimesh(vertices=vertices.reshape(-1, 3), faces=self.mesh.faces)
        loss = (self.containment_weight * self.containment(mesh) + self.curvature_weight * self.curvature(mesh) - self.volume_weight * self.volume(mesh))
        # print(loss)
        return loss
    
    def optimizeMesh(self):
        res = minimize(fun=self.loss_fn, x0=self.mesh.vertices.flatten(), method="Powell", options={"maxiter": self.iterations, "disp": True})

        final_mesh = trimesh.Trimesh(vertices=res.x.reshape(-1, 3), faces=self.mesh.faces)
        self.original_mesh.show()
        final_mesh.show()
        return final_mesh
    
class ColorSpaceTorchOptimizer:
    def __init__(self, mesh: trimesh.Trimesh, device="cpu"):
        self.device = device
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
        self.mesh = Meshes(verts=[vertices], faces=[faces])
        self.original_vertices = vertices.clone().detach()
        self.iterations = 100
        self.deform_verts = torch.full(self.mesh.verts_packed().shape, 0.0, device="cpu", requires_grad=True) # start at 0s?
        self.optimizer = torch.optim.SGD([self.deform_verts], lr=1e-3, momentum=0.9)
        self.original_mesh = mesh.copy()
        print(self.original_mesh.volume)

        print(torch.cuda.is_available())

        # self._prepare_sdf()
        
        # Containment:
        self.containment_weight = 1.0

        # Curvature:
        self.curvature_weight = 20.0

        # Volume:
        self.volume_weight = 0.1

    # def _prepare_sdf(self, n_surface_samples=10000):
    #     surface_points, _ = trimesh.sample.sample_surface(self.original_mesh, n_surface_samples)
    #     self.surface_tree = cKDTree(surface_points)

    def containment(self, vertices: torch.Tensor):
        inside = torch.tensor(self.original_mesh.contains(vertices.detach().numpy()).astype(np.float32), device=self.device)
        signed_dist =  (1.0 - inside)  
        
        loss = torch.mean(torch.relu(signed_dist) ** 2)
        # print("contain")
        # print(loss)
        return loss
    
    def curvature(self, mesh: Meshes):
        laplacian_loss = mesh_laplacian_smoothing(mesh, method="uniform")
        edge_loss = mesh_edge_loss(mesh)
        normal_loss = mesh_normal_consistency(mesh)
        curve = 0.1 * laplacian_loss + edge_loss + 0.01 * normal_loss
        # print("curve")
        # print(curve)
        return curve
            
    def volume(self, mesh: Meshes):
        vol = meshVolume(mesh.verts_packed(), mesh.faces_packed())
        # print("vol")
        # print(vol)
        return -vol
    
    def loss_fn(self, mesh: Meshes):
        return self.containment_weight * self.containment(mesh.verts_packed()) + self.curvature_weight * self.curvature(mesh) - self.volume_weight * self.volume(mesh)
    
    def optimizeMesh(self):
        for i in tqdm(range(self.iterations)):
            self.optimizer.zero_grad()
            new_mesh = self.mesh.offset_verts(self.deform_verts)
            loss = self.loss_fn(new_mesh)
            loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                print("iteration:")
                print(i)
                print("loss:")
                print(loss.item())

        final_mesh = self.mesh.offset_verts(self.deform_verts).cpu()
        return final_mesh

    

