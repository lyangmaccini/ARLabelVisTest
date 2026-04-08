import numpy as np
from scipy.spatial import KDTree, ConvexHull
import math
from utils.color_spaces import RGBtoLAB
from skimage.color import deltaE_cie76

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
    # matlab_path = "data/max_indices_cielab_05.txt"
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

def get_extreme_candidates(vertices, n_directions=2048):
    """
    Sample unit directions on a sphere, find the vertex furthest in each
    direction. These candidates cover all possible furthest-point answers.
    """
    # Fibonacci sphere sampling for uniform direction coverage
    i = np.arange(n_directions, dtype=float)
    phi = np.arccos(1 - 2 * (i + 0.5) / n_directions)
    theta = np.pi * (1 + 5**0.5) * i
    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=1)  # (K, 3)

    # For each direction, find the vertex with maximum dot product
    dots = vertices @ directions.T  # (M, K)
    candidate_indices = np.unique(dots.argmax(axis=0))  # (<=K,)
    return candidate_indices

def furthest_delta_e76_points(inputRGB, allLABPoints):
    inputLAB = RGBtoLAB([inputRGB])[0]
    allLABPoints = np.array(allLABPoints)

    distances = deltaE_cie76(np.tile(inputLAB, (len(allLABPoints), 1)), allLABPoints)
    max_distance_index = np.argmax(distances)

    return allLABPoints[max_distance_index]

def furthest_euclidean_lab_points(allLABs, chunk_size=10_000):
    """
    allLABs: (N, 3) float array of CIELAB points in allRGBs order
    returns: (N, 3) float32 array of furthest LAB point for each input
    """
    lab = allLABs.astype(np.float32)

    # ── 1. Convex hull directly from the LAB points ──
    hull = ConvexHull(lab)
    hull_verts = lab[hull.vertices].astype(np.float32)
    print(f"Hull vertices: {len(hull_verts)}")

    # ── 2. Find furthest hull vertex for each point ──
    N = len(lab)
    print("len: " + str(N))
    furthest = np.empty((N, 3), dtype=np.float32)
    for i in range(0, N, chunk_size):
        print(i)
        chunk = lab[i:i+chunk_size]
        dists_sq = np.sum((chunk[:, None, :] - hull_verts[None, :, :]) ** 2, axis=-1)
        furthest[i:i+chunk_size] = hull_verts[np.argmax(dists_sq, axis=1)]

    return furthest

def furthest_euclidean(allPoints, allRGBs, batch_size=5000):
    # hull = ConvexHull(allPoints)
    # hull_idx = hull.vertices
    # print(hull_idx)
    # hull_pts = allPoints[hull_idx]
   
    # n = len(allPoints)
    # result_indices = np.empty(n, dtype=np.intp)
   
    # for start in range(0, n, batch_size):
    #     end = min(start + batch_size, n)
    #     batch = allPoints[start:end]
    #     difference = batch[:, None, :] - hull_pts[None, :, :]
    #     abs_difference = np.einsum('ijk,ijk->ij', difference, difference)
    #     result_indices[start:end] = np.argmax(abs_difference, axis=1)
   
    # return allRGBs[result_indices]
    # furthest = furthest_lab_points(allPoints)

    # For RGB:
    furthest = np.where(allRGBs < 128, 255, 0).astype(np.uint8) 
    return furthest

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