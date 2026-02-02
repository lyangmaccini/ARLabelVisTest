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
from RGD.mesh import Mesh
from RGD.rgd_admm import rgd_admm

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
    off_data = trimesh.exchange.off.export_off(mesh)
    with open(filename, "w") as f:
        f.write(off_data)

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

        
    for lab in LABtoVertices:
        print(lab)
        distances = rgd_admm(mesh, source_indices=lab, quiet=True)
        furthest_distance_idx = np.argmax(distances)
        c = allRGBs[furthest_distance_idx]
        furthest.append(c)
    furthest = np.array(furthest)
    return furthest

def plot_lab_points_3d(allLABs, allRGBs, furthestRGBs=None, subsample=1):
    """
    allLABs: (N,3) LAB points
    allRGBs: (N,3) RGB points in [0,255]
    furthestRGBs: (M,3) optional RGB points to highlight
    subsample: plot every k-th point for speed
    """

    allLABs = np.asarray(allLABs)
    allRGBs = np.asarray(allRGBs) / 255.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot full LAB cloud
    ax.scatter(
        allLABs[::subsample, 0],
        allLABs[::subsample, 1],
        allLABs[::subsample, 2],
        c=allRGBs[::subsample],
        s=8,
        alpha=0.4,
        linewidth=0
    )

    # Highlight furthest points (if provided)
    if furthestRGBs is not None:
        furthestRGBs = np.asarray(furthestRGBs)
        furthestLABs = RGBToLAB(furthestRGBs)

        ax.scatter(
            furthestLABs[:, 0],
            furthestLABs[:, 1],
            furthestLABs[:, 2],
            c=furthestRGBs / 255.0,
            s=120,
            edgecolors="black",
            linewidth=1.5,
            label="Furthest colors"
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

def main():
    num_cpus = os.cpu_count() 
    n_processes = num_cpus - 4 # Change this to use more/less CPUs. 
    print("Number of CPUs:", num_cpus, "Number of CPUs we are using:", n_processes)
    print("checl")

    allRGBS, allLABs = generate_LABs()
    interval = 10
    allLABs = allLABs[::interval]
    allRGBS = allRGBS[::interval]

    # save_off_file(f"CIELAB_{interval}.off", pointsToMesh(allLABs))

    furthest = furthest_rgd(Mesh.from_file(f"CIELAB_{interval}.off"), allLABs, allRGBS)  
    print(furthest) 
    print(furthest.shape)
    print(allLABs.shape)

    plot_lab_points_3d(
    allLABs,
    allRGBS,
    furthestRGBs=furthest,
    subsample=5  # increase if it’s slow
    )

    mesh = pointsToMesh(allLABs)
    mesh.visual.vertex_colors = assign_vertex_colors(mesh, allLABs, furthest)
    print(mesh.visual.vertex_colors)
    mesh.show()


    # mesh = pointsToMesh(allLABs)
    # mesh.visual.vertex_colors = get_mesh_vertex_colors(mesh, allLABs, allRGB)
    # save_views(mesh)
    # mesh.show()

    # optimizer = ColorSpaceTorchOptimizer(mesh)
    # final_mesh = optimizer.optimizeMesh(mesh)
    # final_mesh.visual.vertex_colors = get_mesh_vertex_colors(final_mesh, allLABs, allRGB) 

    # scene = trimesh.Scene()
    # # scene.add_geometry(mesh)
    # scene.add_geometry(final_mesh)
    # scene.show()

    # intermediate_meshes = optimizer.getIntermediateMeshes()
    # scene.add_geometry(intermediate_meshes[0])
    # # scene.show()

    # frames = []
    # quarter_frames = []
    # half_frames = []
    # three_quarter_frames = []
    # r_quarter = trimesh.transformations.rotation_matrix(np.pi/2.0, [0, 1, 0])
    # r_half = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    # r_three_quarter = trimesh.transformations.rotation_matrix(3.0*np.pi/2.0, [0, 1, 0])
    # for mesh in intermediate_meshes:
    #     try:
    #         t1 = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    #         t1.visual.vertex_colors = mesh.visual.vertex_colors
    #         s1 = trimesh.Scene(t1)
    #         png1 = s1.save_image(resolution=[800,800], visible=True)
    #         frames.append(np.array(Image.open(io.BytesIO(png1))))

    #         t2 = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    #         t2.visual.vertex_colors = mesh.visual.vertex_colors
    #         s2 = trimesh.Scene(t2)
    #         s2.apply_transform(r_quarter)
    #         png2 = s2.save_image(resolution=[800,800], visible=True)
    #         quarter_frames.append(np.array(Image.open(io.BytesIO(png2))))

    #         t3 = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    #         t3.visual.vertex_colors = mesh.visual.vertex_colors
    #         s3 = trimesh.Scene(t3)
    #         s3.apply_transform(r_half)
    #         png3 = s3.save_image(resolution=[800,800], visible=True)
    #         half_frames.append(np.array(Image.open(io.BytesIO(png3))))

    #         t4 = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    #         t4.visual.vertex_colors = mesh.visual.vertex_colors
    #         s4 = trimesh.Scene(t4)
    #         s4.apply_transform(r_three_quarter)
    #         png4 = s4.save_image(resolution=[800,800], visible=True)
    #         three_quarter_frames.append(np.array(Image.open(io.BytesIO(png4))))
    #     except ZeroDivisionError:
    #         print("zero divide")

    # imageio.mimsave("energy_0.gif", frames, duration=0.2)
    # imageio.mimsave("energy_90.gif", quarter_frames, duration=0.2)
    # imageio.mimsave("energy_180.gif", half_frames, duration=0.2)
    # imageio.mimsave("energy_270.gif", three_quarter_frames, duration=0.2)
    # print("saved all views")


if __name__ == "__main__":
    main()