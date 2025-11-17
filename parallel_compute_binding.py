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

def save_single_view(mesh, rotation_matrix, filename):
    s = trimesh.Scene(trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy()))
    s.apply_transform(rotation_matrix)
    png2 = s.save_image(resolution=[800,800], visible=True)
    Image.open(io.BytesIO(png2)).save(filename + ".png")

def save_views(mesh: trimesh.Trimesh):
    r_quarter = trimesh.transformations.rotation_matrix(np.pi/2.0, [0, 1, 0])
    r_half = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    r_three_quarter = trimesh.transformations.rotation_matrix(3.0*np.pi/2.0, [0, 1, 0])

    save_single_view(mesh, r_quarter, "quarter_view")
    save_single_view(mesh, r_half, "half_view")
    save_single_view(mesh, r_three_quarter, "three_quarter_view")

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2 + (p1[2]-p2[2]) ** 2)

def get_mesh_vertex_colors(mesh, allLABs, allRGB):
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
    return np.array(colors)

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

    mesh = pointsToMesh(allLABs)
    mesh.visual.vertex_colors = get_mesh_vertex_colors(mesh, allLABs, allRGB)
    save_views(mesh)
    # mesh.show()

    optimizer = ColorSpaceTorchOptimizer(mesh)
    final_mesh = optimizer.optimizeMesh()
    final_mesh.visual.vertex_colors = get_mesh_vertex_colors(final_mesh, allLABs, allRGB) 

    scene = trimesh.Scene()
    # scene.add_geometry(mesh)
    # scene.add_geometry(final_mesh)
    # scene.show()

    intermediate_meshes = optimizer.getIntermediateMeshes()
    scene.add_geometry(intermediate_meshes[0])
    # scene.show()

    frames = []
    quarter_frames = []
    half_frames = []
    three_quarter_frames = []
    r_quarter = trimesh.transformations.rotation_matrix(np.pi/2.0, [0, 1, 0])
    r_half = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    r_three_quarter = trimesh.transformations.rotation_matrix(3.0*np.pi/2.0, [0, 1, 0])
    for mesh in intermediate_meshes:
        s1 = trimesh.Scene(trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy()))
        png1 = s1.save_image(resolution=[800,800], visible=True)
        frames.append(np.array(Image.open(io.BytesIO(png1))))

        s2 = trimesh.Scene(trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy()))
        s2.apply_transform(r_quarter)
        png2 = s2.save_image(resolution=[800,800], visible=True)
        quarter_frames.append(np.array(Image.open(io.BytesIO(png2))))

        s3 = trimesh.Scene(trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy()))
        s3.apply_transform(r_half)
        png3 = s3.save_image(resolution=[800,800], visible=True)
        half_frames.append(np.array(Image.open(io.BytesIO(png3))))

        s4 = trimesh.Scene(trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy()))
        s4.apply_transform(r_three_quarter)
        png4 = s4.save_image(resolution=[800,800], visible=True)
        three_quarter_frames.append(np.array(Image.open(io.BytesIO(png4))))

    imageio.mimsave("energy_original.gif", frames, duration=0.2)
    imageio.mimsave("energy_quarter.gif", quarter_frames, duration=0.2)
    imageio.mimsave("energy_half.gif", half_frames, duration=0.2)
    imageio.mimsave("energy_three_quarters.gif", three_quarter_frames, duration=0.2)


if __name__ == "__main__":
    main()