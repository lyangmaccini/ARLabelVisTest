from mesh_optimization import ColorSpaceTorchOptimizer
import os
import numpy as np
from parallel_compute_binding import RGBToLAB, pointsToMesh, get_mesh_vertex_colors
import trimesh

def get_LAB_mesh():
    num_cpus = os.cpu_count() 
    n_processes = num_cpus - 4 # Change this to use more/less CPUs. 
    print("Number of CPUs:", num_cpus, "Number of CPUs we are using:", n_processes)

    # Generating LABs:
    stepSize = 16
    # Sample: 0, 15, 31, 45, ... 255
    allRGB = np.array([[r-1, g-1, b-1] for r in range(0, 257, stepSize)
                               for g in range(0, 257, stepSize)
                               for b in range(0, 257, stepSize)])
    allRGB = np.where(allRGB < 0, 0, allRGB)
    allRGB = np.where(allRGB > 255, 255, allRGB)
    allLABs = RGBToLAB(allRGB)

    mesh = pointsToMesh(allLABs)
    mesh.visual.vertex_colors = get_mesh_vertex_colors(mesh, allLABs, allRGB)
    return mesh

def test_containment(optimizer: ColorSpaceTorchOptimizer, mesh: trimesh.Trimesh):
    optimized_mesh = optimizer.optimizeMesh()

def test_curvature(optimizer: ColorSpaceTorchOptimizer, mesh: trimesh.Trimesh):
    optimized_mesh = optimizer.optimizeMesh()

def test_volume(optimizer: ColorSpaceTorchOptimizer, mesh: trimesh.Trimesh):
    optimized_mesh = optimizer.optimizeMesh()


def main():
    LAB_mesh = get_LAB_mesh()

    contaiment_optimizer = ColorSpaceTorchOptimizer(LAB_mesh, curvature_weight=0.0, volume_weight=0.0)
    curvature_optimizer = ColorSpaceTorchOptimizer(LAB_mesh, containment_weight=0.0, volume_weight=0.0)
    volume_optimizer = ColorSpaceTorchOptimizer(LAB_mesh, containment_weight=0.0, curvature_weight=0.0)

if __name__ == "__main__":
    main()