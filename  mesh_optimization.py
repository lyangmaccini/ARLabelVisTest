import numpy as np
import alphashape
import trimesh

def pointsToMesh(allLABs):
    # Converts given points to a trimesh
    # allLABs: RGB points at each step size converted into LAB points; size nx3

    alpha = 0.05 # hyperparameter
    shape = alphashape.alphashape(allLABs, alpha=alpha)
    mesh = trimesh.Trimesh(vertices=np.array(shape.vertices), faces=shape.faces)

    if not mesh.is_watertight:
        print("WARNING: LAB mesh is not watertight")
    # do we want to remesh here at all? will that affect the optimization?
    return mesh

def insideMesh(point: np.array, mesh: trimesh.Trimesh):
    # point should be a 1x3 numpy array
    # Returns whether point is on or within the mesh
    containment = mesh.contains(point)
    # still need to check for points on the mesh surface 
    print(containment.shape)
    return containment[0]

class ColorSpaceOptimizer:
    def __init__(self, trimesh: trimesh.Trimesh):
        self.trimesh = trimesh
        self.original_mesh = trimesh.copy()

    def optimizeMesh(self, iterations):
        return 0
    
    def containment(self, mesh: trimesh.Trimesh):
        loss = 0
        for vert in mesh.vertices:
            loss += 1.0 - float(insideMesh(vert, self.original_mesh))
        return loss
    
    def curvature(self, mesh: trimesh.Trimesh, elliptic_penalty, hyperbolic_penalty, elliptic_threshold, hyperbolic_threshold, threshold_penalty):
        # elliptic_penalty and hyperbolic_penalty are hyperparameters; elliptic_penalty should be larger than hyperbolic_penalty
        # threshold is a hyperparameter representing the maximum curvature to allow
        # should threshold be imposed before or after penalty? could incorporate relative difference here
        # could also add a separate average penalty outside curvature weight

        curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0)
        elliptic_max = -1 * np.min(curvatures)
        hyperbolic_max = np.max(curvatures)
        curvatures[curvatures < 0] *= elliptic_penalty
        curvatures[curvatures > 0] *= hyperbolic_penalty
        total_curvature = np.sum(curvatures ** 2)
        if elliptic_max > elliptic_threshold or hyperbolic_max > hyperbolic_threshold:
            total_curvature += threshold_penalty
        return total_curvature 
            
    
    def volume(self, mesh: trimesh.Trimesh):
        return mesh.volume
    
    def loss(self, mesh: trimesh.Trimesh, containment_weight, curvature_weight, volume_weight):
        return containment_weight * self.containment(mesh) + curvature_weight * self.curvature(trimesh) - volume_weight * self.volume(mesh)

    

