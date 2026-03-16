import numpy as np
from skimage.color import deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, lab2rgb 
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pyvista as pv 
from utils.mesh_optimization import pointsToMesh
import trimesh
import math
from PIL import Image
import io
from RGD.mesh import Mesh
import alphashape
from RGD_CUDA.furthest_rgd_fast import plot_geodesic_field_pyvista

import matplotlib.pyplot as plt
from enum import Enum
from RGD_NEW.mesh_class import MeshClass
from utils.color_spaces import RGBtoOKLAB, RGBtoOKLCH

from RGD_CUDA.furthest_rgd_fast import furthest_rgd

from interpolate import interpolate_files
import pyvista as pv

class Mode(Enum):
    RGD = 1
    RESAMPLE = 2
    SHOW_PLOT = 3
    SHOW_MESH = 4
    TO_FILE = 5
    MESH_FILE = 6,
    TEST_MATLAB = 7,
    OKLAB_FILE = 8,
    OKLCH_FILE = 9,
    INTERPOLATE = 10,
    FULL = 11

class ColorSpace(Enum):
    CIELAB = 1,
    RGB = 2,
    OKLAB = 3,
    OKLCH = 4

class DistanceMeasure(Enum):
    EUCLIDEAN = 1,
    RGD = 2

class SmoothingMethod(Enum):
    ELLIPSE = 1,
    NEURAL_BOUNDING = 2,
    PYTORCH = 3

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
    print(len(range(0, 257, stepSize)))
    allRGBs = np.where(allRGBs < 0, 0, allRGBs)
    allRGBs = np.where(allRGBs > 255, 255, allRGBs)
    allLABs = RGBToLAB(allRGBs)
    print("LABs shape: " + str(allLABs.shape))
    return allRGBs, allLABs

def save_off_file(filename, mesh):
    if filename.split("/")[0] != "data":
        filename = "data/" + filename
    off_data = trimesh.exchange.off.export_off(mesh)
    # print("hello")
    with open(filename, "w") as f:
        f.write(off_data)
    print("Saved to " + filename)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def plot_lab_points_3d(allLABs, furthestRGBs=None, furthest_matlab=None, subsample=1):
    allLABs = np.asarray(allLABs)

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
    elif furthest_matlab is not None:
        ax.scatter(
            allLABs[::subsample, 0],
            allLABs[::subsample, 1],
            allLABs[::subsample, 2],
            c=allLABs[furthest_matlab[::subsample]]/255.0,
            s=120,
            linewidth=1.5
        )  
    else:
        ax.scatter(
            allLABs[::subsample, 0],
            allLABs[::subsample, 1],
            allLABs[::subsample, 2],
            s=120,
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

def furthest_euclidean_old(mesh:Mesh, allLABS, allRGBs, num=1):
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

def resample(filename, count=500):
    if filename.split("/")[0] != "data":
        filename = "data/" + filename
    vertices, faces = Mesh.verts_from_file(filename)
    # plot_lab_points_3d(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    sampled_vertices, face_indices = trimesh.sample.sample_surface_even(mesh, count) # consider adding a seed for consistentcy?

    # resampled = mesh.simplify_quadric_decimation(count)
    shape = alphashape.alphashape(sampled_vertices, alpha=0.01)
    # print(resampled.vertices.shape)
    # print(shape.vertices.shape)
    mesh = trimesh.Trimesh(vertices=shape.vertices, faces=shape.faces)
    # mesh.show()
    plot_lab_points_3d(mesh.vertices)
    print(mesh.vertices.shape)
    return mesh

def show_original_mesh_pyvista(mesh,
                               show_edges=True,
                               show_normals=False):

    V = mesh.vertices
    F = mesh.faces

    # Convert faces to PyVista format
    faces_pv = np.hstack(
        [np.full((F.shape[0], 1), 3), F]
    ).astype(np.int32).ravel()

    poly = pv.PolyData(V, faces_pv)

    # Clean mesh (optional but useful for diagnostics)
    poly_clean = poly.clean(tolerance=1e-12)

    print("---- Mesh Diagnostics ----")
    print("Vertices:", poly.n_points)
    print("Faces:", poly.n_cells)
    print("Is manifold:", poly.is_manifold)
    print("Is all triangles:", poly.is_all_triangles)
    print("Has open edges:", poly.n_open_edges > 0)
    print("--------------------------")

    plotter = pv.Plotter()

    # Add mesh with edge overlay
    plotter.add_mesh(
        poly_clean,
        color="lightgray",
        show_edges=show_edges,
        edge_color="black",
        smooth_shading=True,
        backface_params=dict(color="orange"),
    )

    # Optional: show normals
    if show_normals:
        normals = poly_clean.compute_normals(
            cell_normals=False,
            point_normals=True,
            auto_orient_normals=False
        )
        print("showing normal")

        arrows = normals.glyph(
            orient="Normals",
            scale=False,
            factor=0.05 * np.linalg.norm(V.max(0) - V.min(0))
        )

        plotter.add_mesh(arrows, color="blue")

    plotter.add_axes()
    plotter.show()

def write_point_cloud_ply(points, filename):
    """Write point cloud as PLY file"""
    with open(filename, 'w') as f:
        # Header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        
        # Data
        for point in points:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')

def main():
    num_cpus = os.cpu_count() 
    n_processes = num_cpus - 4 # Change this to use more/less CPUs. 
    print("Number of CPUs:", num_cpus, "Number of CPUs we are using:", n_processes)

    interval = 4
    space = "CIELAB"

    original_path = f"data/RGB2{space}_{interval}.off"

    original_LAB_path = f"data/RGB2CIELAB_{interval}.off"
    original_OKLAB_path = f"data/RGB2OKLAB_{interval}.off"
    original_OKLCH_path = f"data/RGB2OKLCH_{interval}.off"

    saved_furthest_path = f"data/MATLAB_FurthestRGB_From{space}_{interval}_resampled.txt"
    matlab_path = f"data/max_indices_{space}_matlab.txt"

    allRGBs, allLABs = generate_LABs(stepSize=interval)

    if space == "OKLAB":
        allPoints = RGBtoOKLAB(allRGBs)
    elif space == "OKLCH":
        allPoints = RGBtoOKLCH(allRGBs)
    else:
        allPoints = allLABs

    mode = Mode.MESH_FILE

    if mode is Mode.RGD:
        furthest = furthest_rgd(Mesh.from_file(original_path), allPoints, allRGBs)
        print(furthest)
        print(furthest.shape)
        plot_lab_points_3d(allLABs, furthestRGBs=furthest)
        np.savetxt(saved_furthest_path, furthest, fmt='%d')
        print("Saved furthest RGB values at interval: " + str(interval))

    elif mode is Mode.TO_FILE:
        furthest = np.loadtxt(saved_furthest_path, dtype=int)
        LAB_file = f"CandidateLABvals_MATLAB_{space}_{interval}.txt"
        RGB_file = f"CorrespondingRGBVals_MATLAB_{space}_{interval}.txt"
        with open(LAB_file, "w") as f, open(RGB_file, "w") as f2:
            for lab, rgb in zip(furthest.tolist(), allRGBs.tolist()):
                f.write(f"{lab[0]},{lab[1]},{lab[2]}\n")
                f2.write(f"{rgb[0]},{rgb[1]},{rgb[2]}\n")
        print("Saved files: " + LAB_file + ", " + RGB_file)

    elif mode is Mode.SHOW_MESH:
        furthest = np.loadtxt(saved_furthest_path, dtype=int)
        mesh = pointsToMesh(allLABs)
        mesh.visual.vertex_colors = assign_vertex_colors(mesh, allLABs, furthest)
        mesh.show()
        print("Showed furthest RGBs in mesh")

    elif mode is Mode.MESH_FILE:
        mesh = pointsToMesh(allPoints)
        # save_off_file(original_path, mesh)
        show_original_mesh_pyvista(mesh, show_edges=True, show_normals=False)

    elif mode is Mode.TEST_MATLAB:
        matlab_test_path = "data/u_D1.csv"
        point_test_path = 'data/RGB2OKLAB_1.off'
        dist = np.loadtxt(matlab_test_path)
        vertices, faces = MeshClass.read_off(point_test_path)
        mesh = MeshClass(vertices, faces)
        furthest_idx =  int(np.argmax(dist))
        plot_geodesic_field_pyvista(mesh, dist, source_idx=0, furthest_idx=furthest_idx)

if __name__ == "__main__":
    main()