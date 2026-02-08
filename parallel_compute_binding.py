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
from utils.mesh_optimization import pointsToMesh, ColorSpaceOptimizer, ColorSpaceTorchOptimizer
from utils.binding import bindToOptimizedMeshBinding
import trimesh
import math
from trimesh.viewer import SceneViewer
import time
import imageio
from PIL import Image
import io
from RGD.mesh import Mesh
from RGD.rgd_admm import rgd_admm
import alphashape

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
    mesh_copy = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    mesh_copy.visual.vertex_colors = mesh.visual.vertex_colors
    s = trimesh.Scene(mesh)
    s.apply_transform(rotation_matrix)
    png = s.save_image(resolution=[800,800], visible=True)
    Image.open(io.BytesIO(png)).save(filename + ".png")

def save_views(mesh: trimesh.Trimesh):
    r_quarter = trimesh.transformations.rotation_matrix(np.pi/2.0, [0, 1, 0])
    r_half = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    r_three_quarter = trimesh.transformations.rotation_matrix(3.0*np.pi/2.0, [0, 1, 0])

    save_single_view(mesh, r_quarter, "quarter_view")
    save_single_view(mesh, r_half, "half_view")
    save_single_view(mesh, r_three_quarter, "three_quarter_view")

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2 + (p1[2]-p2[2]) ** 2)

def get_mesh_vertex_colors(mesh, allLABs, allRGBs):
    colors = []
    for vert in mesh.vertices:
        best_idx = 0
        best_distance = dist(vert, allLABs[0])
        for i, lab in enumerate(allLABs):
            distance = dist(vert, lab)
            if distance < best_distance:
                best_idx = i
                best_distance = distance
        c = allRGBs[best_idx]/255.0
        color = [c[0], c[1], c[2], 1.0]
        colors.append(color)
    return np.array(colors)

def generate_LABs(stepSize = 16):
    # Sample: 0, 15, 31, 45, ... 255
    allRGBs = np.array([[r-1, g-1, b-1] for r in range(0, 257, stepSize)
                               for g in range(0, 257, stepSize)
                               for b in range(0, 257, stepSize)])
    allRGBs = np.where(allRGBs < 0, 0, allRGBs)
    allRGBs = np.where(allRGBs > 255, 255, allRGBs)
    allLABs = RGBToLAB(allRGBs)
    print(allLABs.shape)
    print("labs ready")
    return allRGBs, allLABs

def save_off_file(filename, mesh):
    print(filename)
    filename = "data/" + filename
    off_data = trimesh.exchange.off.export_off(mesh)
    # print("hello")
    with open(filename, "w") as f:
        f.write(off_data)
    print("Saved to " + filename)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def furthest_rgd(mesh:Mesh, allLABS, allRGBs, num=1):
    def closest_vertex(lab, num=1):
        # would have to modify for higher number of points for distance
        closest_vert = 0
        closest_dist = 1000000
        for i, mesh_vertex in enumerate(mesh.vertices):
            # print(mesh_vertex)
            # print(lab)
            euclidean_dist = euclidean_distance(mesh_vertex, lab)
            if euclidean_dist < closest_dist:
                closest_dist = euclidean_dist
                closest_vert = i
        return closest_vert
    
    furthest = []

    LABtoVertices = [] # closest vertex (index) to each LAB point
    for lab in allLABS:
        LABtoVertices.append(closest_vertex(lab))
    LABtoVertices = np.array(LABtoVertices)
    print("closest vertices found")

    # for i in range(len(mesh.vertices)):
    #     print(i)
    #     distances = rgd_admm(mesh, source_indices=i, quiet=True)
    #     furthest_distance_idx = np.argmax(distances)
    #     c = allRGBs[furthest_distance_idx]/255.0
    #     furthest.append([c[0], c[1], c[2], 1.0])

        
    for i, lab in enumerate(LABtoVertices):
        print(i)
        distances = rgd_admm(mesh, source_indices=lab, quiet=True)
        furthest_distance_idx = np.argmax(distances)
        c = allRGBs[furthest_distance_idx]
        furthest.append(c)
    furthest = np.array(furthest)
    return furthest

def plot_lab_points_3d(allLABs, furthestRGBs=None, subsample=1):
    """
    allLABs: (N,3) LAB points
    allRGBs: (N,3) RGB points in [0,255]
    furthestRGBs: (M,3) optional RGB points to highlight
    subsample: plot every k-th point for speed
    """

    allLABs = np.asarray(allLABs)
    # allRGBs = np.asarray(allRGBs) / 255.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if furthestRGBs is not None:
        ax.scatter(
            allLABs[::subsample, 0],
            allLABs[::subsample, 1],
            allLABs[::subsample, 2],
            c=furthestRGBs[::subsample]/255.0,
            s=120,
            # alpha=0.4,
            linewidth=1.5
        )
    else:
        ax.scatter(
            allLABs[::subsample, 0],
            allLABs[::subsample, 1],
            allLABs[::subsample, 2],
            s=120,
            # alpha=0.4,
            linewidth=1.5
        )

    ax.set_xlabel("L*")
    ax.set_ylabel("a*")
    ax.set_zlabel("b*")

    ax.set_title("CIELAB space (furthest colors highlighted)")
    ax.legend()
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.show()


def furthest_euclidean(mesh:Mesh, allLABS, allRGBs, num=1):
    def closest_vertex(lab, num=1):
        # would have to modify for higher number of points for distance
        closest_vert = 0
        closest_dist = 1000000
        for i, mesh_vertex in enumerate(mesh.vertices):
            # print(mesh_vertex)
            # print(lab)
            euclidean_dist = euclidean_distance(mesh_vertex, lab)
            if euclidean_dist < closest_dist:
                closest_dist = euclidean_dist
                closest_vert = i
        return closest_vert
    
    furthest = []

    LABtoVertices = [] # closest vertex (index) to each LAB point
    for lab in allLABS:
        LABtoVertices.append(closest_vertex(lab))
    LABtoVertices = np.array(LABtoVertices)
    print("closest vertices found")
        
    for lab in LABtoVertices:
        print(lab)
        distances = rgd_admm(mesh, source_indices=lab, quiet=True)
        furthest_distance_idx = np.argmax(distances)
        c = allRGBs[furthest_distance_idx]
        furthest.append(c)
    furthest = np.array(furthest)
    return furthest

def assign_vertex_colors(mesh:trimesh.Trimesh, allLABs, furthest):
    colors = []
    for vertex in mesh.vertices:
        closest = 0
        closest_dist = 100000
        for i, lab in enumerate(allLABs):
            dist2 = euclidean_distance(lab, vertex)
            if dist2 < closest_dist:
                closest = i
                closest_dist = dist2
        c = furthest[closest]/255.0
        colors.append([c[0], c[1], c[2], 1.0])
    return np.array(colors)

def resample(filename):
    if filename.split("/")[0] != "data":
        filename = "data/" + filename
    vertices, faces = Mesh.verts_from_file(filename)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    count = 2500
    sampled_vertices, face_indices = trimesh.sample.sample_surface_even(mesh, count) # consider adding a seed for consistentcy?
    shape = alphashape.alphashape(sampled_vertices, alpha=0.01)
    print(np.array(sampled_vertices).shape)
    print(shape.vertices.shape)
    mesh = trimesh.Trimesh(vertices=shape.vertices, faces=shape.faces)
    return mesh


def main():
    num_cpus = os.cpu_count() 
    n_processes = num_cpus - 4 # Change this to use more/less CPUs. 
    print("Number of CPUs:", num_cpus, "Number of CPUs we are using:", n_processes)
    print("checl")

    rgb2cielab_16_resampled = "data/RGB2CIELAB_16_resampled.off"
    rgb2cielab_32_resampled = "data/RGB2CIELAB_32.off"

    allRGBS, allLABs = generate_LABs(stepSize=32)
    # subsample = 50
    # plot_lab_points_3d(allLABs, subsample=50)

    # Further subsample LABs and RGBs
    # interval = 75
    # allLABs = allLABs[::interval]
    # allRGBS = allRGBS[::interval]

    # Save data filesw
    # save_off_file("RGB2CIELAB_32.off", pointsToMesh(allLABs))

    # Calculate farthest distances 
    furthest = furthest_rgd(Mesh.from_file(rgb2cielab_32_resampled), allLABs, allRGBS)  
    print(furthest.shape)
    print(allLABs.shape)


    # Saving resampled files
    # mesh = resample("RGB2CIELAB_16.off")
    # save_off_file("RGB2CIELAB_16_resampled.off", mesh)

    # Plotting results with matplotlib
    # plot_lab_points_3d(mesh.vertices, subsample=5)
    plot_lab_points_3d(allLABs, furthestRGBs=furthest)

    print(furthest)
    # with open("data/Furthest_RGB_32_resampled.txt", "w") as f:
        # f.write(furthest)
    # print("Saved furthest colors")
    np.savetxt('data/Furthest_RGB_32_resampled.txt', furthest, fmt='%d')


    # Assign proper colors to vertices
    # mesh = pointsToMesh(allLABs)
    # mesh.visual.vertex_colors = assign_vertex_colors(mesh, allLABs, furthest)
    # print(mesh.visual.vertex_colors)
    # mesh.show()


if __name__ == "__main__":
    main()