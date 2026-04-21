import numpy as np
from scipy.spatial import KDTree, ConvexHull
import math
from utils.color_spaces import RGBtoLAB
from skimage.color import deltaE_cie76, deltaE_ciede2000, lab2rgb
import pyvista as pv

def closest_vertices_batch(lab_points: np.ndarray, mesh_tree: KDTree) -> np.ndarray:
    """For each LAB point, return the index of the closest mesh vertex."""
    _, indices = mesh_tree.query(lab_points)
    return indices.astype(int)

def closest_labs_batch(mesh_vertices: np.ndarray, lab_tree: KDTree) -> np.ndarray:
    """For each mesh vertex, return the index of the closest LAB point."""
    _, indices = lab_tree.query(mesh_vertices)
    return indices.astype(int)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def furthest_rgd(vertices, allPoints, allRGBs, matlab_path):
    allPoints = np.asarray(allPoints, dtype=float)
    allRGBs = np.asarray(allRGBs)

    mesh_tree = KDTree(vertices)
    lab_tree = KDTree(allPoints)
    LABtoVertices = closest_vertices_batch(allPoints, mesh_tree)
    VerticestoLAB = closest_labs_batch(vertices, lab_tree)
    print(f"{len(allPoints)} LAB points <-> {len(vertices)} mesh vertices mapped")

    unique_sources = np.unique(LABtoVertices)
    n_unique = len(unique_sources)

    print(f"Finding furthest for {n_unique} unique sources ({len(allPoints)} LAB points)")

    furthestRGBs = []
    max_indices = np.loadtxt(matlab_path, delimiter=',', dtype='int')
    print(f"From matlab file: {matlab_path}")

    for i in range(allPoints.shape[0]):
        vert = LABtoVertices[i]
        if vert < max_indices.shape[0]:
            furthest_vert = max_indices[vert] - 1
        else:
            furthest_vert = 0
            print("WARNING: out of bounds. Vertices are likely not what RGD was run on.")
        furthest_lab = VerticestoLAB[int(furthest_vert)]
        furthestRGBs.append(allRGBs[int(furthest_lab)])

    return np.array(furthestRGBs)

def furthest_delta_e76_points(inputRGB, allLABPoints):
    inputLAB = RGBtoLAB([inputRGB])[0]
    allLABPoints = np.array(allLABPoints)

    distances = deltaE_cie76(np.tile(inputLAB, (len(allLABPoints), 1)), allLABPoints)
    max_distance_index = np.argmax(distances)

    return allLABPoints[max_distance_index]

def furthest_euclidean_lab_points(allLABs, chunk_size=10_000):
    lab = allLABs.astype(np.float32)

    hull = ConvexHull(lab)
    hull_verts = lab[hull.vertices].astype(np.float32)
    print(f"Hull vertices: {len(hull_verts)}")

    N = len(lab)
    furthest = np.empty((N, 3), dtype=np.float32)
    for i in range(0, N, chunk_size):
        print(i)
        chunk = lab[i:i+chunk_size]
        dists_sq = np.sum((chunk[:, None, :] - hull_verts[None, :, :]) ** 2, axis=-1)
        furthest[i:i+chunk_size] = 255.0 * lab2rgb(hull_verts[np.argmax(dists_sq, axis=1)])
    return furthest

def furthest_euclidean_rgb(allRGBs):
    furthest = np.where(allRGBs < 128, 255, 0).astype(np.uint8) 
    return furthest

def plot_geodesic_field_pyvista(V, F,
                                distances,
                                source_idx=None,
                                furthest_idx=None,
                                cmap="plasma"):
    # Convert faces to PyVista format
    faces_pv = np.hstack(
        [np.full((F.shape[0], 1), 3), F]
    ).astype(np.int32)
    faces_pv = faces_pv.ravel()

    poly = pv.PolyData(V, faces_pv)

    # Attach scalar field
    poly["Geodesic Distance"] = distances

    # Normalize for consistent coloring
    clim = [float(distances.min()), float(distances.max())]

    plotter = pv.Plotter()
    plotter.add_mesh(
        poly,
        scalars="Geodesic Distance",
        cmap=cmap,
        clim=clim,
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={
            "title": "Regularized Geodesic Distance",
        }
    )

    if source_idx is not None:
        center = V[source_idx]
        radius = 0.02 * np.linalg.norm(V.max(0) - V.min(0))

        source_sphere = pv.Sphere(radius=radius, center=center)
        plotter.add_mesh(
            source_sphere,
            color="lime",
            specular=1.0,
            smooth_shading=True,
        )

    # ---- Add furthest sphere ----
    if furthest_idx is not None:
        center = V[furthest_idx]
        radius = 0.02 * np.linalg.norm(V.max(0) - V.min(0))

        furthest_sphere = pv.Sphere(radius=radius, center=center)
        plotter.add_mesh(
            furthest_sphere,
            color="red",
            specular=1.0,
            smooth_shading=True,
        )

    plotter.add_axes()
    plotter.show()


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