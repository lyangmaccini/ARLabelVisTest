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

def test_containment(LAB_mesh):
    print("Testing containment:")  

    # LAB_mesh = get_LAB_mesh()

    offset_mesh = LAB_mesh.copy()
    offset_mesh.apply_scale(1.1)
    
    contaiment_optimizer = ColorSpaceTorchOptimizer(LAB_mesh, curvature_weight=0.0, volume_weight=0.0, iterations=4000, save_intermediate_meshes=False)
    optimized_mesh = contaiment_optimizer.optimizeMesh(offset_mesh)

    scene = trimesh.Scene()
    scene.add_geometry(optimized_mesh)
    scene.show()

    diff = abs(np.sum(optimized_mesh.vertices - LAB_mesh.vertices))
    assert(diff < 100) # should do a quick check to see if everything's inside the mesh-- verts could be not exactly where they started but still within the mesh
    print("Shifted difference after optimization: " + str(diff))
    print("Passed containment test.")

def test_curvature(LAB_mesh):
    print("Testing curvature:")

    # mesh = get_LAB_mesh()
    
    curvature_optimizer = ColorSpaceTorchOptimizer(LAB_mesh, containment_weight=0.0, volume_weight=0.0, iterations=75000)
    optimized_mesh = curvature_optimizer.optimizeMesh(LAB_mesh)

    scene = trimesh.Scene()
    scene.add_geometry(optimized_mesh)
    scene.show()

    final_curvature = np.sum(curvature_optimizer.getFinalCurvature())
    initial_curvature = np.sum(curvature_optimizer.getOriginalCurvature())
    print("Final curvature: " + str(final_curvature))
    print("Initial curvature: " + str(initial_curvature))
    assert(final_curvature < initial_curvature)
    print("Passed curvature test.")

def test_volume(LAB_mesh):
    print("Testing volume:")

    initial_volume = LAB_mesh.volume
    
    volume_optimizer = ColorSpaceTorchOptimizer(LAB_mesh, containment_weight=0.0, curvature_weight=0.0, volume_weight=1.0, iterations=5000, print_updates=False)
    optimized_mesh = volume_optimizer.optimizeMesh(LAB_mesh)

    scene = trimesh.Scene()
    scene.add_geometry(optimized_mesh)
    scene.show()

    final_volume = optimized_mesh.volume
    print("Final volume: " + str(final_volume))
    print("Initial volume: " + str(initial_volume))
    assert(final_volume > initial_volume)
    print("Passed curvature test.")

def main():
    LAB_mesh = get_LAB_mesh()

    # test_containment(LAB_mesh)
    test_curvature(LAB_mesh)    
    # test_volume(LAB_mesh)

if __name__ == "__main__":
    main()