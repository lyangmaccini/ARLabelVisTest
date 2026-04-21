import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv 
from utils.points_to_mesh import pointsToMesh
import trimesh
from utils.distances import plot_geodesic_field_pyvista
import matplotlib.pyplot as plt
from enum import Enum
from utils.color_spaces import RGBtoOKLAB
from utils.distances import furthest_rgd, furthest_euclidean, furthest_delta_e76_points, furthest_euclidean_lab_points
from utils.interpolate import interpolate_interval, interpolate_from_files
from utils.files import read_off, save_off_file
from utils.color_spaces import RGBtoLAB
from utils.distances import euclidean_distance
from utils.binding import bindLABtoSphere, bindToNeuralBounding, bindToOptimizedMeshBinding
from utils.voxels import writeVoxels
from utils.metrics import run_metrics, process_final_colors
from utils.mesh_optimization import optimize_mesh

class Mode(Enum):
    RGD = 1, 
    SHOW_MESH = 2, 
    TO_FILE = 3, 
    MESH_FILE = 4, 
    TEST_MATLAB = 5, 
    INTERPOLATE = 6, 
    FULL = 7, 
    SMOOTH_MESH = 8, 
    TO_VOXELS = 9, 
    METRICS = 10

def get_mesh_vertex_colors(mesh, allLABs, allRGBs):
    colors = []
    for vert in mesh.vertices:
        best_idx = 0
        best_distance = euclidean_distance(vert, allLABs[0])
        for i, lab in enumerate(allLABs):
            distance = euclidean_distance(vert, lab)
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
    allLABs = RGBtoLAB(allRGBs)
    print("LABs shape: " + str(allLABs.shape))
    return allRGBs, allLABs

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

def show_original_mesh_pyvista(mesh,
                               show_edges=True,
                               show_normals=False):

    V = mesh.vertices
    F = mesh.faces

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

def main():
    interval = 1
    space = "CIELAB"
    distance_measure = "RGD"
    alpha = "75"
    voxel_dim = 256 # neural binding
    sigma = 0.0
    vox = 256 # meshing
    str_sigma = str(sigma).replace(".", "o")

    # neural, sphere, pytorch, none (gaussian?)
    smoothing_mode = "neural"

    if (smoothing_mode == "none"):
        # print("NO SMOOTHING/GAUSSIAN")
        final_LAB_filepath = f"AllCandidateLABvals_{space}_{interval}_{distance_measure}_{alpha}_sigma_{str_sigma}_vox_{vox}.txt"
        # print("goal: " + final_LAB_filepath)
        original_path = f"RGD_MATLAB/RGB2{space}_{interval}_sigma_{str_sigma}_vox_{vox}.off"
        matlab_path = f"RGD_MATLAB/max_indices_{space}_{interval}_{distance_measure}_{alpha}_sigma_{str_sigma}_vox_{vox}.txt"
    else:
        print("SMOOTHING")
        original_path = f"RGD_MATLAB/RGB2{space}_{smoothing_mode}_{interval}.off"
        matlab_path = f"RGD_MATLAB/max_indices_{space}_{interval}_{distance_measure}_{alpha}_{smoothing_mode}_{voxel_dim}.txt"
        final_LAB_filepath = f"AllCandidateLABvals_{space}_{interval}_{distance_measure}_{alpha}_{smoothing_mode}_{voxel_dim}.txt"

    LAB_file = f"CandidateLABvals_MATLAB_{space}_{interval}_{distance_measure}_{alpha}.txt"
    RGB_file = f"CorrespondingRGBVals_MATLAB_{space}_{interval}_{distance_measure}_{alpha}.txt"

    saved_furthest_path = f"data/MATLAB_FurthestRGB_From{space}_{interval}_{distance_measure}_{alpha}.txt"

    smoothed_path = f"RGD_MATLAB/RGB2{space}_{smoothing_mode}_{interval}_sigma_{sigma}.off"
    binvox_filepath = f"neural_bounding/data/3D/{space}_{interval}_{str(voxel_dim)}.binvox"
    neural_bounded_filepath = f"data/neural_bounding_{space}_{voxel_dim}.binvox"

    # mode = Mode.FULL
    # mode = Mode.MESH_FILE
    # mode = Mode.SMOOTH_MESH
    mode = Mode.METRICS

    if mode is not Mode.METRICS:
        allRGBs, allLABs = generate_LABs(stepSize=interval)

        if space == "OKLAB":
            allPoints = RGBtoOKLAB(allRGBs)
        elif space == "RGB":
            allPoints = allRGBs
        else:
            allPoints = allLABs


    if mode is Mode.RGD:
        vertices, faces = read_off(original_path)
        furthest = furthest_rgd(vertices, allPoints, allRGBs, matlab_path)
        print(furthest)
        print(furthest.shape)
        np.savetxt(saved_furthest_path, furthest, fmt='%d')
        print("Saved furthest RGB values at interval: " + str(interval))

    elif mode is Mode.TO_FILE:
        furthest = np.loadtxt(saved_furthest_path, dtype=int)
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
        mesh = pointsToMesh(allPoints, sigma=sigma, vox=vox)
        show_original_mesh_pyvista(mesh, show_edges=True, show_normals=False)
        print(f"Saved mesh with sigma {sigma}, voxel {vox}, color space {space}")

    elif mode is Mode.TEST_MATLAB:
        matlab_test_path = "RGD_MATLAB/u_D1.csv"
        point_test_path = f'RGD_MATLAB/RGB2{space}_{interval}.off'
        dist = np.loadtxt(matlab_test_path)
        vertices, faces = read_off(point_test_path)
        furthest_idx =  int(np.argmax(dist))
        plot_geodesic_field_pyvista(vertices, faces, dist, source_idx=0, furthest_idx=furthest_idx)

    elif mode is Mode.SMOOTH_MESH:
        if smoothing_mode == "sphere":
            allBoundPoints = bindLABtoSphere(allPoints, allRGBs)
        elif smoothing_mode == "neural":
            print("neural")
            allBoundPoints = bindToNeuralBounding(neural_bounded_filepath, voxel_dim, allPoints, allRGBs)
        elif smoothing_mode == "pytorch":
            mesh = pointsToMesh(allPoints, sigma=sigma, vox=vox)
            show_original_mesh_pyvista(mesh, show_edges=True, show_normals=False)
            optimized_mesh = optimize_mesh(mesh, n_iters=20000, lr=5e-4, w_smooth=1.0, w_inside=200.0, w_volume=0.5, sdf_resolution=64)
            optimized_mesh.show()
            show_original_mesh_pyvista(optimized_mesh, show_edges=True, show_normals=False)

            allBoundPoints = bindToOptimizedMeshBinding(optimized_mesh, allPoints)
        else:
            allBoundPoints = allPoints
        mesh = pointsToMesh(allBoundPoints)
        save_off_file(smoothed_path, mesh)
        show_original_mesh_pyvista(mesh, show_edges=True, show_normals=False)

    elif mode is Mode.FULL:
        if space == "RGB":
            vertices = allRGBs
        else:
            vertices, faces = read_off(original_path)
        if distance_measure == "RGD":
            furthest = furthest_rgd(vertices, allPoints, allRGBs, matlab_path)
        else:
            print("euclidean")
            furthest = furthest_euclidean_lab_points(allPoints)
        interpolate_interval(allRGBs, furthest, final_LAB_filepath, interval)

    elif mode is Mode.INTERPOLATE:
        interpolate_from_files(RGB_file, LAB_file, final_LAB_filepath)

    elif mode is Mode.TO_VOXELS:
        writeVoxels(allPoints, voxel_dim, binvox_filepath)

    elif mode is Mode.METRICS:
        run_metrics()
        # original_csv = "data/label_colors_export_og.csv"
        # new_csv = "data/label_colors_export_new.csv"
        # smoothed_csv = "data/label_colors_export.csv"
        # process_final_colors(original_csv, "original_gradients.png")
        # process_final_colors(new_csv, "new_gradients.png")
        # process_final_colors(smoothed_csv, "smoothed_gradients.png")


if __name__ == "__main__":
    main()