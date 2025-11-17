import numpy as np
import alphashape
import trimesh
import torch
from tqdm import tqdm
from scipy.optimize import minimize
import kaolin as kal

# torch version: torch-2.5.1 + cu118

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
    volume = torch.sum(v0 * torch.cross(v1, v2, dim=1)) / 6.0
    return torch.abs(volume)

def triangleAngles(opposites, left, right):
    cos = (left**2 + right**2 - opposites**2) / (2 * left * right)
    return torch.acos(cos.clamp(-1.0, 1.0))



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
    def __init__(self, mesh: trimesh.Trimesh, device="cuda"):
        self.device = device

        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

        self.original_vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        self.iterations = 100000

        self.deform_verts = torch.zeros_like(self.vertices, device=device, requires_grad=True) 
        self.optimizer = torch.optim.SGD([self.deform_verts], lr=1e-4, momentum=0.85)
        # self.original_mesh = mesh.copy()
        print("original volume:")
        # print(self.original_mesh.volume)
        print(meshVolume(self.vertices, self.faces))
        print("cuda:")
        print(torch.cuda.is_available())
        
        # Containment:
        self.containment_weight = 10000.0

        # Curvature:
        self.curvature_weight = 1200.0

        # Volume:
        self.volume_weight = 0.0000001

    def gaussian_curvature(self, vertices: torch.Tensor, faces: torch.Tensor):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e1 = (v1 - v2).norm(dim=1)
        e2 = (v2 - v0).norm(dim=1)
        e3 = (v0 - v1).norm(dim=1)

        mesh_angles = torch.stack([triangleAngles(e1, e2, e3), triangleAngles(e2, e3, e1), triangleAngles(e3, e1, e2)], dim=1)
        vertex_angles = torch.full((vertices.shape[0],), 2*torch.pi, device=self.device)

        for i in range(3):
            vertex_angles = vertex_angles.index_add(0, faces[:, i], -mesh_angles[:, i])
        
        return vertex_angles

    def containment(self, vertices, faces):
        # print(self.original_vertices.shape)
        # print(vertices.shape)
        verts = torch.unsqueeze(vertices, dim=0)
        original_verts = torch.unsqueeze(self.original_vertices, dim=0)
        distances, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(verts, kal.ops.mesh.index_vertices_by_faces(original_verts, faces))
        containment = kal.ops.mesh.check_sign(original_verts, faces, verts)[0].float()
        c = torch.mean(distances * (1.0 - containment))
        # print("con")
        # print(c)
        return c
    
    def curvature(self, vertices, faces):
        # laplacian_loss = mesh_laplacian_smoothing(mesh, method="uniform")
        # edge_loss = mesh_edge_loss(mesh)
        # normal_loss = mesh_normal_consistency(mesh)
        # curve = 0.1 * laplacian_loss + edge_loss + 0.01 * normal_loss
        # print("curve")
        # print(curve)


        c = torch.max(self.gaussian_curvature(vertices, faces) ** 2)
        # print("cur")
        # print(c)
        # smoothed_verts = kal.metrics.trianglemesh.uniform_laplacian_smoothing(torch.unsqueeze(vertices, dim=1), faces)[0]
        # distances = torch.norm(vertices - smoothed_verts, dim=1)
        # c = distances.sum()
        # print(c)
        return c
            
    def volume(self, vertices, faces):
        vol = meshVolume(vertices,faces)
        # print("vol")
        # vol = mesh.volume
        # print(vol)
        return -vol
    
    def loss_fn(self, vertices, faces):
        # return self.containment_weight * self.containment(vertices, faces) + self.curvature_weight * self.curvature(vertices, faces) + self.volume_weight * self.volume(vertices, faces)
        # return self.containment(vertices, faces)
        return self.curvature_weight * self.curvature(vertices, faces) + self.containment_weight * self.containment(vertices, faces)
    
    def optimizeMesh(self):
        for i in tqdm(range(self.iterations)):
            
            self.optimizer.zero_grad()
            # new_mesh = kal.rep.SurfaceMesh(vertices=self.vertices + self.deform_verts, faces=self.faces)
            new_vertices = self.vertices + self.deform_verts
            loss = self.loss_fn(new_vertices, self.faces)
            loss.backward()
            self.optimizer.step()

            if i % 1000 == 0:
                print("iteration:")
                print(i)
                print("loss:")
                print(loss.item())

        final_mesh = trimesh.Trimesh(vertices=(self.vertices + self.deform_verts).detach().cpu().numpy(), faces=self.faces.detach().cpu().numpy())
        return final_mesh

    

