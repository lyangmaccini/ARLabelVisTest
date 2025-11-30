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

def test_containment():
    print("Testing containment:")
    # original_mesh = get_LAB_mesh()
    # original_mesh.apply_translation(np.array([-50, 0, 0]))
    
    LAB_mesh = get_LAB_mesh()
    offset_mesh = LAB_mesh.copy()

    offset_mesh.apply_scale(1.1)
    print(LAB_mesh.vertices)
    print(offset_mesh.vertices)
    # saved_mesh = offset_mesh.copy()
    offset_colors = np.zeros_like(offset_mesh.visual.vertex_colors)
    print(offset_colors)
    offset_colors[:,3] = 1.0
    print(offset_colors)
    # offset_mesh.visual.vertex_colors = offset_colors
    
    
    contaiment_optimizer = ColorSpaceTorchOptimizer(LAB_mesh, curvature_weight=0.0, volume_weight=0.0, iterations=1300, save_intermediate_meshes=False)
    optimized_mesh = contaiment_optimizer.optimizeMesh(offset_mesh)
    print(optimized_mesh.vertices)
    diff = abs(np.sum(optimized_mesh.vertices - LAB_mesh.vertices))
    print(LAB_mesh.vertices)
    print(diff)
    scene = trimesh.Scene()
    scene.add_geometry(optimized_mesh)
    # scene.add_geometry(offset_mesh)
    scene.show()
    assert(diff < 100)
    print("Passed containment test.")

def test_curvature():
    print("Testing curvature:")
    
    curvature_optimizer = ColorSpaceTorchOptimizer(mesh, containment_weight=0.0, volume_weight=0.0)
    optimized_mesh = curvature_optimizer.optimizeMesh()

    print("Passed curvature test.")

def test_volume():
    print("Testing volume:")
    mesh = get_LAB_mesh()
    volume_optimizer = ColorSpaceTorchOptimizer(mesh, containment_weight=0.0, curvature_weight=0.0)
    optimized_mesh = volume_optimizer.optimizeMesh()

    print("Passed volume test.")

def main():
    test_containment()
    # test_curvature()    
    # test_volume()

if __name__ == "__main__":
    main()