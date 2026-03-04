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
from RGD.rgd_cuda import furthest_rgd_cuda_new, rgd_admm_cuda, batch_closest, furthest_lab
from scipy.spatial import cKDTree
# from RGD.temp import rgd_admm_batch
from RGD.rgd_admm_batch import furthest_rgd_batch
import alphashape
from RGD_CUDA.furthest_rgd_fast import plot_geodesic_field_pyvista

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cupy as cp
import cProfile
from tqdm import tqdm
from RGD.rgd_admm_batch_cuda import rgd_admm_batch_cuda, furthest_rgd_batch_cuda
from enum import Enum
from RGD_NEW.mesh_class import MeshClass
from oklab_test import RGBtoOKLAB, RGBtoOKLCH

from RGD_CUDA.furthest_rgd_fast import furthest_rgd

class Mode(Enum):
    RGD = 1
    RESAMPLE = 2
    SHOW_PLOT = 3
    SHOW_MESH = 4
    TO_FILE = 5
    MESH_FILE = 6,
    TEST_MATLAB = 7,
    OKLAB_FILE = 8,
    OKLCH_FILE = 9


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

def furthest_rgd_old(mesh:Mesh, allLABS, allRGBs, num=1):
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
    
    def closest_lab(vertex, num=1):
        # would have to modify for higher number of points for distance
        closest_lab = 0
        closest_dist = 1000000
        for i, lab in enumerate(allLABS):
            # print(mesh_vertex)
            # print(lab)
            euclidean_dist = euclidean_distance(vertex, lab)
            if euclidean_dist < closest_dist:
                closest_dist = euclidean_dist
                closest_lab = i
        return closest_lab
    
    furthest = []

    LABtoVertices = [] # closest vertex (index) to each LAB point
    for lab in allLABS:
        LABtoVertices.append(closest_vertex(lab))
    LABtoVertices = np.array(LABtoVertices)
    print("closest vertices found")

    VerticestoLAB = [] # closest vertex (index) to each LAB point
    for vert in mesh.vertices:
        VerticestoLAB.append(closest_lab(vert))
    VerticestoLAB = np.array(VerticestoLAB)
    print("closest vertices found")

    for i, lab in enumerate(LABtoVertices):
        print(i)
        distances = rgd_admm(mesh, source_indices=lab, quiet=True)
        furthest_distance_vertex_idx = np.argmax(distances)
        print(furthest_distance_vertex_idx)
        c = allRGBs[VerticestoLAB[furthest_distance_vertex_idx]]
        furthest.append(c)
    furthest = np.array(furthest)
    return furthest

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
        
    for lab in LABtoVertices:
        print(lab)
        # distances = rgd_admm(mesh, source_indices=lab, quiet=True)
        # furthest_distance_idx = np.argmax(distances)
        # c = allRGBs[furthest_distance_idx]
        # furthest.append(c)
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

def furthest_rgd_fast(mesh, allLABS, allRGBs, alpha_hat=0.05, batch_size=32):
    allLABS = np.asarray(allLABS)
    verts   = np.asarray(mesh.vertices)

    # fast vectorised nearest-neighbour (GPU if available)
    LABtoVertices = batch_closest(allLABS, verts)
    VerticestoLAB = batch_closest(verts, allLABS)

    # solve all sources in batches — one factorisation per batch
    furthest = []
    batches = range(0, len(LABtoVertices), batch_size)

    for start in tqdm(batches, desc="Batches"):
        end      = min(start + batch_size, len(LABtoVertices))
        sources  = LABtoVertices[start:end]

        U = rgd_admm(mesh, sources, alpha_hat=alpha_hat, quiet=True, abs_tol=1e-3, rel_tol=0.05, max_iter=500)
        # U is (batch_size, nv) — one row per source

        for row in U:
            far_vert = np.argmax(row)
            furthest.append(allRGBs[VerticestoLAB[far_vert]])

    return np.array(furthest)

from scipy.spatial import cKDTree
import numpy as np

def build_maps(mesh, allLABS):
    V = mesh.vertices

    vert_tree = cKDTree(V)
    _, LAB_to_V = vert_tree.query(allLABS)

    lab_tree = cKDTree(allLABS)
    _, V_to_LAB = lab_tree.query(V)

    return LAB_to_V, V_to_LAB

from scipy.sparse.linalg import factorized

def rgd_precompute(mesh, alpha_hat=0.05):

    nv = mesh.nv
    ta = mesh.ta
    Ww = mesh.Ww

    total_area = ta.sum()
    alpha = alpha_hat * np.sqrt(total_area)
    rho   = 2.0 * np.sqrt(total_area)

    A = (alpha + rho) * Ww

    return {
        "A_fact": factorized(A.tocsc()),
        "alpha": alpha,
        "rho": rho
    }

def rgd_single_source(mesh, source, cache, max_iter=500):

    nv = mesh.nv
    G  = mesh.G
    ta = mesh.ta
    va = mesh.va

    A_fact = cache["A_fact"]
    rho    = cache["rho"]

    mask = np.ones(nv, dtype=bool)
    mask[source] = False

    u  = np.zeros(nv)
    y  = np.zeros(3*mesh.nf)
    z  = np.zeros(3*mesh.nf)

    for _ in range(max_iter):

        b = va - G.T @ (ta.repeat(3) * y) + rho * G.T @ (ta.repeat(3) * z)
        u[mask] = A_fact(b[mask])

        Gx = G @ u

        z_temp = (1/rho)*y + Gx
        z_temp = z_temp.reshape(mesh.nf,3)

        norms = np.linalg.norm(z_temp,axis=1)
        norms[norms<1] = 1
        z = (z_temp / norms[:,None]).flatten()

        y += rho*(Gx - z)

    return u

def furthest_rgd_new(mesh, allLABS, allRGBs, num=1):

    LAB_to_V, V_to_LAB = build_maps(mesh, allLABS)

    cache = rgd_precompute(mesh)

    furthest = np.zeros((len(allLABS),3))

    for i, v in enumerate(LAB_to_V):

        dist = rgd_single_source(mesh, v, cache)

        idx  = np.argmax(dist)

        furthest[i] = allRGBs[V_to_LAB[idx]]

    return furthest

import pyvista as pv


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

    rgb2cielab_4_resampled = "data/RGB2CIELAB_4_resampled.off"
    rgb2cielab_16_resampled = "data/RGB2CIELAB_16_resampled.off"
    rgb2cielab_32_resampled = "data/RGB2CIELAB_32_resampled.off"
    rgb2cielab_64_resampled = "data/RGB2CIELAB_64_resampled.off"

    rgb2cielab_4 = "data/RGB2CIELAB_4.off"
    rgb2cielab_16 = "data/RGB2CIELAB_16.off"
    rgb2cielab_32 = "data/RGB2CIELAB_32.off" # nan? i think too few verts
    rgb2cielab_64 = "data/RGB2CIELAB_64.off"

    interval = 4

    original_LAB_path = f"data/RGB2CIELAB_{interval}_17.off"
    original_OKLAB_path = f"data/RGB2OKLAB_{interval}_17.off"
    original_OKLCH_path = f"data/RGB2OKLCH_{interval}_17.off"
    resampled_LAB_path = f"data/RGB2CIELAB_{interval}_resample_3000.off"
    saved_furthest_path = f"data/MATLAB_Furthest_RGB_{interval}_resampled.txt"
    matlab_path = "data/max_indices_matlab.txt"

    allRGBs, allLABs = generate_LABs(stepSize=interval)

    mode = Mode.OKLCH_FILE

    if mode is Mode.RESAMPLE:
        count = 3000
        # mesh = resample(f"RGB2CIELAB_{interval}.off", count=count)
        # save_off_file(f"RGB2CIELAB_{interval}_resampled.off", mesh)
        mesh = resample(original_LAB_path, count=count)
        save_off_file(resampled_LAB_path, mesh)
        print("Saved resampled mesh at interval: " + str(interval))
    elif mode is Mode.RGD:
        # furthest = furthest_rgd(Mesh.from_file(original_LAB_path), allLABs, allRGBs, alpha_hat=0.1)
        furthest = furthest_rgd(Mesh.from_file(original_LAB_path), original_LAB_path, allLABs, allRGBs, alpha_hat=0.1)
        
        print(furthest)
        print(furthest.shape)
        np.savetxt(saved_furthest_path, furthest, fmt='%d')
        plot_lab_points_3d(allLABs, furthestRGBs=furthest)

        print("Saved furthest RGB values at interval: " + str(interval))
    elif mode is Mode.SHOW_PLOT:
        # furthest = np.loadtxt(saved_furthest_path, dtype=int)
        max_indices = np.loadtxt(matlab_path, delimiter=',')
        furthest = []
        print(allLABs.shape)
        print(max_indices.shape)
        for i in range(allLABs.shape[0]):
            if i < max_indices.shape[0]:
                # print(max_indices[i])
                furthest.append(allRGBs[int(max_indices[i])])
            else:
                furthest.append(np.array([0.0, 0.0, 0.0]))

        # NEED TO MAP LABS TO VERTICES NEAREST THEN BROADCAST TO THE 
        furthest = np.array(furthest)
        # furthest = np.pad(furthest, (0, len(allLABs) - len(furthest)), constant_values=0)
        plot_lab_points_3d(allLABs, furthestRGBs=furthest)
        print("Showed furthest RGBs at interval: " + str(interval))
    elif mode is Mode.TO_FILE:
        furthest = np.loadtxt(saved_furthest_path, dtype=int)
        LAB_file = f"CandidateLABvals_MATLAB_030_{interval}.txt"
        RGB_file = f"CorrespondingRGBVals_MATLAB_030_{interval}.txt"
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
        # plot_lab_points_3d(allLABs)
        # write_point_cloud_ply(allLABs, "allLABs.ply")
        mesh = pointsToMesh(allLABs)
        save_off_file(original_LAB_path, mesh)
        show_original_mesh_pyvista(mesh, show_edges=True, show_normals=True)
    elif mode is Mode.TEST_MATLAB:
        matlab_test_path = "data/u_D1 (6).csv"
        point_test_path = 'data/RGB2CIELAB_4_15.off'
        dist = np.loadtxt(matlab_test_path)
        vertices, faces = MeshClass.read_off(point_test_path)
        mesh = MeshClass(vertices, faces)
        furthest_idx =  int(np.argmax(dist))

        #     print(dist.min(), dist.max())
        #     print(dist)
        #     print(np.percentile(dist, 95))
        #     print(dist[furthest_idx])
        plot_geodesic_field_pyvista(mesh, dist, source_idx=0, furthest_idx=furthest_idx)
    elif mode is Mode.OKLAB_FILE:
        allOKLABs = RGBtoOKLAB(allRGBs)
        mesh = pointsToMesh(allOKLABs)
        save_off_file(original_OKLAB_path, mesh)
        show_original_mesh_pyvista(mesh, show_edges=True, show_normals=True)
    elif mode is Mode.OKLCH_FILE:
        allOKLCHs = RGBtoOKLCH(allRGBs)
        mesh = pointsToMesh(allOKLCHs)
        save_off_file(original_OKLCH_path, mesh)
        show_original_mesh_pyvista(mesh, show_edges=True, show_normals=True)


    # Further subsample LABs and RGBs
    # interval = 75
    # allLABs = allLABs[::interval]
    # allRGBS = allRGBS[::interval]

    # Save data files
    # save_off_file("RGB2CIELAB_4.off", pointsToMesh(allLABs))


    # Calculate farthest distances 
    # furthest = furthest_rgd(Mesh.from_file(rgb2cielab_16_resampled), allLABs, allRGBs)  
    # cProfile.run('furthest_rgd_batch(Mesh.from_file("data/RGB2CIELAB_16_resampled.off"), allLABS, allRGBs)', sort='cumulative')
    # furthest = furthest_rgd_batch(Mesh.from_file(resampled_LAB), allLABs, allRGBs, alpha_hat=0.05)
    # furthest = furthest_rgd_batch_cuda(Mesh.from_file(resampled_LAB), allLABs, allRGBs, alpha_hat=0.1)



if __name__ == "__main__":
    main()