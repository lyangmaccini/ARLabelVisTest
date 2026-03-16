import numpy as np
from scipy.spatial import KDTree
import math

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

def furthest_rgd(vertices, allPoints, allRGBs):
    allPoints = np.asarray(allPoints, dtype=float)
    allRGBs = np.asarray(allRGBs)

    print("Building KDTrees...")
    mesh_tree = KDTree(vertices)
    lab_tree = KDTree(allPoints)
    LABtoVertices = closest_vertices_batch(allPoints, mesh_tree)
    VerticestoLAB = closest_labs_batch(vertices, lab_tree)
    print(f"  {len(allPoints)} LAB points <-> {len(vertices)} mesh vertices mapped")

    unique_sources = np.unique(LABtoVertices)
    n_unique = len(unique_sources)

    print(f"Finding furthest for {n_unique} unique sources ({len(allPoints)} LAB points)...")

    furthestRGBs = []
    matlab_path = "data/max_indices_cielab_05.txt"
    max_indices = np.loadtxt(matlab_path, delimiter=',', dtype='int')
    print("from matlab file")
    print(max_indices)
    print(max_indices.shape)

    for i in range(allPoints.shape[0]):
        vert = LABtoVertices[i]
        if vert < max_indices.shape[0]:
            furthest_vert = max_indices[vert] - 1
        else:
            furthest_vert = 0
            print("out of bounds")
        furthest_lab = VerticestoLAB[int(furthest_vert)]
        furthestRGBs.append(allRGBs[int(furthest_lab)])

    return np.array(furthestRGBs)


def furthest_euclidean_old(vertices, allPoints, allRGBS):
    allPoints = np.asarray(allPoints, dtype=float)
    allRGBs = np.asarray(allRGBs)

    print("Building KDTrees...")
    mesh_tree = KDTree(vertices)
    lab_tree = KDTree(allPoints)
    LABtoVertices = closest_vertices_batch(allPoints, mesh_tree)
    VerticestoLAB = closest_labs_batch(vertices, lab_tree)
    print(f"  {len(allPoints)} LAB points <-> {len(vertices)} mesh vertices mapped")

    unique_sources = np.unique(LABtoVertices)
    n_unique = len(unique_sources)

    print(f"Finding furthest for {n_unique} unique sources ({len(allPoints)} LAB points)...")

    furthestRGBs = []

    for i in range(allPoints.shape[0]):
        vert = LABtoVertices[i]
        furthest_vert = 0
        furthest_dist = -1
        for i, mesh_vertex in enumerate(vertices):
            euclidean_dist = euclidean_distance(mesh_vertex, vert)
            if euclidean_dist > furthest_dist:
                furthest_dist = euclidean_dist
                furthest_vert = i

        furthest_lab = VerticestoLAB[int(furthest_vert)]
        furthestRGBs.append(allRGBs[int(furthest_lab)])

    return np.array(furthestRGBs)

def plot_geodesic_field(mesh, distances,
                        source_idx=None,
                        furthest_idx=None,
                        cmap='viridis'):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    V = mesh.vertices
    F = mesh.faces

    d_min = distances.min()
    d_max = distances.max()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Surface colored by scalar field
    surf = ax.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        cmap=cmap,
        linewidth=0.1,
        antialiased=True,
        shade=True,
        array=distances
    )
    surf.set_clim(d_min, d_max)

    # ---- Compute vertex normals (simple area-weighted average) ----
    normals = np.zeros_like(V)
    tri_pts = V[F]
    tri_normals = np.cross(tri_pts[:,1] - tri_pts[:,0],
                           tri_pts[:,2] - tri_pts[:,0])

    for i in range(3):
        normals[F[:, i]] += tri_normals

    norms = np.linalg.norm(normals, axis=1)
    normals[norms > 0] /= norms[norms > 0][:, None]

    # small offset scale relative to mesh size
    bbox_size = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
    offset_scale = 0.02 * bbox_size

    def draw_marker(idx, color, label):
        p = V[idx]
        n = normals[idx]
        p_offset = p + offset_scale * n  # lift above surface

        # outline
        ax.scatter(*p_offset,
                   color='black',
                   s=600,
                   depthshade=False)

        # filled center
        ax.scatter(*p_offset,
                   color=color,
                   s=300,
                   depthshade=False,
                   label=label)

    if source_idx is not None:
        draw_marker(source_idx, 'lime', 'Source')

    if furthest_idx is not None:
        draw_marker(furthest_idx, 'red', 'Furthest')

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
    cbar.set_label("Regularized Geodesic Distance")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend()

    plt.tight_layout()
    plt.show()

import pyvista as pv


def plot_geodesic_field_pyvista(V, F,
                                distances,
                                source_idx=None,
                                furthest_idx=None,
                                cmap="plasma"):
    # Convert faces to PyVista format
    # PyVista expects: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    faces_pv = np.hstack(
        [np.full((F.shape[0], 1), 3), F]
    ).astype(np.int32)
    faces_pv = faces_pv.ravel()

    # Create PolyData
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

    # ---- Add source sphere ----
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