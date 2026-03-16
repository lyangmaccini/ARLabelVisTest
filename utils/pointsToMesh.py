"""
RGB Gamut Boundary Mesh in OKLab Space
---------------------------------------
Assumes you already have `oklab_points` as an (N, 3) numpy array
of all 8-bit RGB colors converted to OKLab.

Pipeline:
  1. Extract boundary points (voxel neighbour check)
  2. Poisson surface reconstruction via Open3D (with outward normal correction)
  3. QEM simplification + Loop subdivision
  4. trimesh repair (fill holes, fix normals)

Dependencies:
    pip install open3d trimesh numpy
"""

import numpy as np
import open3d as o3d
import trimesh


# ─────────────────────────────────────────────────────────────────────────────
# 1. Boundary extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundary_points(oklab_points: np.ndarray) -> np.ndarray:
    """
    Keep only OKLab points that sit on the gamut boundary — i.e. whose
    voxel has at least one empty face-neighbour in OKLab voxel space.
    Reduces ~16.7 M points to ~200 k–400 k surface points.
    """
    print("Extracting boundary points…")

    vox_res = 256
    mins  = oklab_points.min(axis=0)
    maxs  = oklab_points.max(axis=0)
    scale = (vox_res - 1) / (maxs - mins + 1e-12)
    idx   = np.floor((oklab_points - mins) * scale).astype(np.int32)
    idx   = np.clip(idx, 0, vox_res - 1)

    occupied = np.zeros((vox_res, vox_res, vox_res), dtype=bool)
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    offsets = np.array([
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
    ], dtype=np.int32)

    is_boundary = np.zeros(len(oklab_points), dtype=bool)
    for off in offsets:
        ni      = idx + off
        outside = np.any((ni < 0) | (ni >= vox_res), axis=1)
        valid   = ~outside
        inside_empty = np.zeros(len(oklab_points), dtype=bool)
        inside_empty[valid] = ~occupied[ni[valid, 0], ni[valid, 1], ni[valid, 2]]
        is_boundary |= outside | inside_empty

    boundary = oklab_points[is_boundary]
    print(f"  {len(boundary):,} boundary points (of {len(oklab_points):,} total)")
    return boundary


# ─────────────────────────────────────────────────────────────────────────────
# 2. Poisson surface reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def poisson_reconstruct(
    boundary_pts: np.ndarray,
    poisson_depth: int = 8,
    density_quantile: float = 0.005,
    target_faces: int = 10_000,
) -> o3d.geometry.TriangleMesh:
    """
    Estimate normals then run Screened Poisson Surface Reconstruction.

    poisson_depth     : 6–10; higher = more faithful, slower.
    density_quantile  : fraction of lowest-density faces to trim. Keep this
                        small (0.005–0.01) to avoid creating holes.
    target_faces      : simplify output to this many triangles via QEM.
    """
    print("Building Open3D point cloud & estimating normals…")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(boundary_pts)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )

    # Orient all normals to point away from the centroid.
    # This is the most reliable method for a closed convex-ish surface
    # and prevents Poisson from wrapping the surface around both sides.
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    normals  = np.asarray(pcd.normals).copy()
    points   = np.asarray(pcd.points)
    outward  = points - centroid                         # centroid → point
    flip     = np.sum(normals * outward, axis=1) < 0    # inward-pointing normals
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    print(f"  Flipped {flip.sum():,} inward-pointing normals")

    print(f"Running Poisson reconstruction (depth={poisson_depth})…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=1.2, linear_fit=False
    )

    # Trim the Poisson "skirt" — use a very small quantile to avoid holes
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_quantile)
    keep = densities > threshold
    mesh.remove_vertices_by_mask(~keep)

    # Simplify to a manageable face count before remeshing
    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )
    mesh.compute_vertex_normals()

    print(f"  Poisson mesh: {len(mesh.vertices):,} verts, "
          f"{len(mesh.triangles):,} faces")
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# 3. Uniform remeshing with pyacvd
# ─────────────────────────────────────────────────────────────────────────────

def uniform_remesh(
    mesh: o3d.geometry.TriangleMesh,
    target_faces: int = 10_000,
    subdivide: int = 2,
) -> trimesh.Trimesh:
    """
    Optionally subdivide for smoother triangles, then convert to trimesh
    and repair any remaining holes or flipped normals.

    target_faces : desired triangle count (applied via QEM in poisson_reconstruct).
    subdivide    : Loop subdivision passes for rounder triangles. 0 to skip.
    """
    if subdivide > 0:
        print(f"  Subdividing {subdivide}x in Open3D…")
        mesh = mesh.subdivide_loop(number_of_iterations=subdivide)
        # Decimate back down to target after subdivision
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Repair holes and fix any remaining flipped normals
    trimesh.repair.fill_holes(tm)
    print("HOLES")
    tm.is_watertight
    trimesh.repair.fix_normals(tm)

    print(f"  Final mesh: {len(tm.vertices):,} verts, {len(tm.faces):,} faces, "
          f"watertight={tm.is_watertight}")
    return tm


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_gamut_mesh(
    oklab_points: np.ndarray,
    *,
    poisson_depth: int = 8,
    density_quantile: float = 0.005,
    target_faces: int = 10_000,
    subdivide: int = 2,
) -> trimesh.Trimesh:
    """
    Full pipeline: OKLab point cloud → trimesh of gamut boundary.

    Parameters
    ----------
    oklab_points      : (N, 3) array of OKLab colours
    poisson_depth     : Poisson reconstruction depth (6–10). Higher = finer.
    density_quantile  : Fraction of low-density Poisson faces to discard.
                        Keep small (0.005) to avoid holes.
    target_faces      : Desired triangle count in the final mesh.
    subdivide         : Loop subdivision passes for rounder triangles (0–3).

    Returns
    -------
    trimesh.Trimesh
    """
    boundary = extract_boundary_points(oklab_points)
    o3d_mesh = poisson_reconstruct(boundary, poisson_depth, density_quantile)
    tm       = uniform_remesh(o3d_mesh, target_faces, subdivide)
    return tm


# ─────────────────────────────────────────────────────────────────────────────
# Example usage  (replace with your actual oklab_points)
# ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     # --- plug in your array here ---
#     # from your_module import oklab_points   # shape (16_777_216, 3)

#     # Quick smoke-test with a synthetic sphere (remove when using real data)
#     print("Generating synthetic test point cloud (replace with your oklab_points)…")
#     rng   = np.random.default_rng(0)
#     theta = rng.uniform(0, np.pi,   50_000)
#     phi   = rng.uniform(0, 2*np.pi, 50_000)
#     oklab_points = np.column_stack([
#         np.sin(theta) * np.cos(phi),
#         np.sin(theta) * np.sin(phi),
#         np.cos(theta),
#     ])
#     # --------------------------------

#     mesh = build_gamut_mesh(
#         oklab_points,
#         poisson_depth=8,
#         density_quantile=0.005,
#         target_faces=10_000,
#         subdivide=2,
#     )

#     mesh.export("oklab_gamut.glb")
#     print("Saved → oklab_gamut.glb")
#     print(mesh)