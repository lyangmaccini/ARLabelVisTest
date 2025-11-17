import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
from skimage.color import deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, lab2rgb 
# from tqdm import tqdm
import os
from multiprocessing import Pool
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
# from scipy.optimize import curve_fit
# from skimage import measure
import matplotlib.pyplot as plt
import pyvista as pv 
# from binding import bindToMeshBinding
# from voxels import convertToVoxels
from mesh_optimization import pointsToMesh, ColorSpaceOptimizer, ColorSpaceTorchOptimizer
from binding import bindToOptimizedMeshBinding
import trimesh
import math
from trimesh.viewer import SceneViewer
import time
import imageio
from PIL import Image
import io

def RGBToLAB(RGB):
    RGB = np.array(RGB) / 255.0
    mask = RGB > 0.04045

    RGB[mask] = ((RGB[mask] + 0.055) / 1.055) ** 2.4
    RGB[~mask] /= 12.92
    RGB *= 100


    XYZ = np.dot(RGB, np.array([[0.4124, 0.3576, 0.1805],
                                [0.2126, 0.7152, 0.0722],
                                [0.0193, 0.1192, 0.9505]]))

    XYZ /= np.array([95.047, 100.0, 108.883])
    mask = XYZ > 0.008856
    XYZ[mask] = XYZ[mask] ** (1/3)
    XYZ[~mask] = (7.787 * XYZ[~mask]) + (16/116)

    L = (116 * XYZ[:, 1]) - 16
    a = 500 * (XYZ[:, 0] - XYZ[:, 1])
    b = 200 * (XYZ[:, 1] - XYZ[:, 2])

    return np.stack([L, a, b], axis=1)
    
def findPointAtMaxDistance(inputRGB, allLABPoints):
    inputLAB = RGBToLAB([inputRGB])[0]
    allLABPoints = np.array(allLABPoints)

    distances = deltaE_cie76(np.tile(inputLAB, (len(allLABPoints), 1)), allLABPoints)
    max_distance_index = np.argmax(distances)

    return allLABPoints[max_distance_index]

def process_colors(rgb_range, allLABs):
    CandidateLABs = []
    CandidateRGBs = []
    
    CandidateRGBs = rgb_range.tolist()
    print('finished converting RGB to a list')
    CandidateLABs = np.apply_along_axis(lambda a: findPointAtMaxDistance(a, allLABs), axis=1, arr=rgb_range)
    CandidateLABs = CandidateLABs.tolist()
    print('finished generating all candidateLABs')
    return CandidateLABs, CandidateRGBs

def main():
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
    print(allLABs.shape)
    print("labs ready")

    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2 + (p1[2]-p2[2]) ** 2)

    mesh = pointsToMesh(allLABs)
    colors = []
    for vert in mesh.vertices:
        best_idx = 0
        best_distance = dist(vert, allLABs[0])
        for i, lab in enumerate(allLABs):
            distance = dist(vert, lab)
            if distance < best_distance:
                best_idx = i
                best_distance = distance
        c = allRGB[best_idx]/255.0
        color = [c[0], c[1], c[2], 1.0]
        colors.append(color)
    colors = np.array(colors)
    print(colors.shape)
    mesh.visual.vertex_colors = colors
    print(mesh.visual.vertex_colors.shape)
    print(allRGB.shape)
    # mesh.show()
    optimizer = ColorSpaceTorchOptimizer(mesh)
    print("hello")
    final_mesh = optimizer.optimizeMesh()
    colors = []
    for vert in final_mesh.vertices:
        best_idx = 0
        best_distance = dist(vert, allLABs[0])
        for i, lab in enumerate(allLABs):
            distance = dist(vert, lab)
            if distance < best_distance:
                best_idx = i
                best_distance = distance
        c = allRGB[best_idx]/255.0
        color = [c[0], c[1], c[2], 1.0]
        colors.append(color)
    colors = np.array(colors)
    final_mesh.visual.vertex_colors = colors 

    scene = trimesh.Scene()
    # scene.add_geometry(mesh)
    # scene.add_geometry(final_mesh)
    # scene.show()

    intermediate_meshes = optimizer.getIntermediateMeshes()
    scene.add_geometry(intermediate_meshes[0])
    # scene.show()
    viewer = scene.show()
    # print(viewer)
    # while not viewer.is_active:
        # time.sleep(0.1)
    # time.sleep(2)
    # for mesh in intermediate_meshes[1:]:
    #     scene.geometry[list(scene.geometry.keys())[0]].vertices = mesh.vertices
    #     viewer._redraw()
    #     time.sleep(0.1)
    #     print("hello")
        # scene.show()
    frames = []
    r = trimesh.transformations.rotation_matrix(np.pi/2.0, [0, 1, 0])
    for mesh in intermediate_meshes:
        s = trimesh.Scene(mesh)
        s.apply_transform(r)
        png = s.save_image(resolution=[800,800], visible=True)
        # Image.open(io.BytesIO(png)).show()
        frames.append(np.array(Image.open(io.BytesIO(png))))
    imageio.mimsave("energy.gif", frames, duration=0.2)
    print(frames)
    print(np.min(frames[0]))
    
    # optimized_trimesh = trimesh.Trimesh(vertices=final_mesh.verts_packed().detach().numpy(), faces=final_mesh.faces_packed().detach().numpy())
    # mesh.show()
    # final_mesh.show()
    # Voxelizing mesh: neural binding requires a voxelized point cloud. When we run the neural binding code, it automatically
    # saves a .binvox file as testNeuralBounding_<dims>.binvox to finish the mapping on.
    # dim = 100
    # write = False
    # if write:
    #     voxels = convertToVoxels(allLABs, dim)
    #     v = binvox_rw.Voxels(voxels, [dim, dim, dim], [0.0, 0.0, 0.0], 1.0, 'xyz')
    #     filepath = "allLABs" + str(dim) + ".binvox"
    #     with open(filepath, 'w', encoding="latin-1") as fp:
    #         binvox_rw.write(v, fp)
    #     print("Saved to file " + filepath)

    # Processing post-neural-bounding voxels of color space:
    # read = True
    # if read:
    #     print("opening file")
    #     filepath = "testNeuralBounding_" + str(dim) + ".binvox"
    #     voxels = np.zeros((dim, dim, dim))
    #     with open(filepath, "rb") as fp:
    #         voxels = binvox_rw.read_as_3d_array(fp).data

    #     # More smoothing; convert to triangle mesh
    #     print("meshing")
    #     verts, faces, _, _ = measure.marching_cubes(voxels, 0.0)
    #     new_faces = []
    #     for face in faces:
    #         arr = np.insert(face, 0, 3)
    #         new_faces.append(arr)
    #     faces = np.hstack(new_faces) 
    # vertices = final_mesh.vertices
    # faces = final_mesh.faces
    # mesh = pv.PolyData(vertices, faces)
    # print("subdividing")
    # mesh.subdivide(1, subfilter='loop', inplace=True)
    # verts = mesh.points # in voxel space

    # x_max, y_max, z_max = np.max(allLABs, axis=0)
    # x_min, y_min, z_min = np.min(allLABs, axis=0)
    # max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

    # for i in range(len(verts)):
    #     new_x = max_range * verts[i][0] / dim
    #     new_y = max_range * verts[i][1] / dim
    #     new_z = max_range * verts[i][2] / dim

    #     new_x += x_min
    #     new_y += y_min
    #     new_z += z_min

    #     verts[i] = [new_x, new_y, new_z]
    
    #     # for visualization (mesh):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(verts[:,0], verts[:,1], verts[:,2], c = "blue", alpha=0.5)
    #     ax.set_xlim([-2, 35])   # Set x-axis limits
    #     ax.set_ylim([-2, 35])   # Set y-axis limits
    #     ax.set_zlim([-2, 35])   # Set z-axis limits
    #     ax.set_box_aspect([1.0, 1.0, 1.0])
    #     ax.set_xlabel("L")
    #     ax.set_ylabel("a")
    #     ax.set_zlabel("b")
    #     ax.set_title("Mesh Surface Plot")
    #     plt.show()

    #     # Map all points to smoothed mesh:
    #     print("binding")


    # boundedLABs = bindToOptimizedMeshBinding(final_mesh, allLABs)
    # # boundedLABs = allLABs

    # # For visualization (bounded LABs):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(boundedLABs[:,2], boundedLABs[:,1], boundedLABs[:,0], c = allRGB/255.0, alpha=0.15)
    # ax.set_xlim([-100, 100]) 
    # ax.set_ylim([-100, 100])  
    # ax.set_zlim([-100, 100])  
    # ax.set_box_aspect([1.0, 1.0, 1.0])
    # ax.set_xlabel("b")
    # ax.set_ylabel("a")
    # ax.set_zlabel("L")
    # # ax.set_title("Bounded LABs Surface Plot")
    # plt.show()

    #     print("number allLAB", len(allLABs))
    #     print("number boundedLABs", len(boundedLABs))

    # print("finished preparing all LABs")
    # rgb_ranges = np.array_split(allRGB, n_processes)

    # with Pool(n_processes) as pool:
    #     results = pool.starmap(process_colors, [(rgb_range, boundedLABs) for rgb_range in rgb_ranges])

    # CandidateLABs = []
    # CandidateRGBs = []
    # for lab, rgb in results:
    #     CandidateLABs += lab
    #     CandidateRGBs += rgb

    # with open("CandidateLABvals_step16_76_ellipsoid.txt", "w") as f, open("CorrespondingRGBVals_step16_76_ellipsoid.txt", "w") as f2:
    #     for lab, rgb in zip(CandidateLABs, CandidateRGBs):
    #         f.write(f"{lab[0]},{lab[1]},{lab[2]}\n")
    #         f2.write(f"{rgb[0]},{rgb[1]},{rgb[2]}\n")

    # print("Number of unique LABs:", len(CandidateLABs))


if __name__ == "__main__":
    main()