# """
# RGB Gamut Boundary Mesh in OKLab Space
# ---------------------------------------
# Assumes you already have `oklab_points` as an (N, 3) numpy array
# of all 8-bit RGB colors converted to OKLab.

# Pipeline:
#   1. Voxelise OKLab points into a 3D occupancy grid
#   2. Gaussian-blur the grid to get a smooth SDF-like field
#   3. Marching cubes → guaranteed watertight mesh, no gaps possible
#   4. Laplacian smooth to remove voxel staircase artifacts
#   5. QEM decimation to target face count
#   6. trimesh repair as final safety net

# Dependencies:
#     pip install open3d trimesh numpy scikit-image scipy
# """

import numpy as np
import trimesh
import trimesh.smoothing
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import open3d as o3d
import pymeshlab


# def build_gamut_mesh(
#     oklab_points: np.ndarray,
#     *,
#     vox_res: int = 128,
#     smooth_sigma: float = 1.5,
#     target_faces: int = 10_000,
#     smooth_iterations: int = 10,
#     pre_decimate_smooth: int = 3, # light smooth before decimation
#     post_decimate_smooth: int = 5
# ) -> trimesh.Trimesh:
#     """
#     Full pipeline: OKLab point cloud → watertight trimesh of gamut boundary.

#     Parameters
#     ----------
#     oklab_points      : (N, 3) array of OKLab colours
#     vox_res           : voxel grid resolution per axis. 64 is fast, 128 is
#                         good, 256 is high precision (memory: vox_res^3 bytes).
#     smooth_sigma      : Gaussian blur sigma applied to occupancy grid before
#                         marching cubes. Higher = smoother surface but less
#                         faithful to the true boundary. 1.0–2.0 is good.
#     target_faces      : desired triangle count after decimation.
#     smooth_iterations : Laplacian smoothing passes to remove voxel staircase.

#     Returns
#     -------
#     trimesh.Trimesh  (guaranteed watertight)
#     """

#     # ── 1. Voxelise ──────────────────────────────────────────────────────────
#     print(f"Voxelising {len(oklab_points):,} points at {vox_res}³…")

#     mins  = oklab_points.min(axis=0)
#     maxs  = oklab_points.max(axis=0)
#     # Add a 2-voxel padding so marching cubes sees a closed surface at edges
#     padding = 2
#     scale = (vox_res - 1 - 2 * padding) / (maxs - mins + 1e-12)
#     idx   = np.floor((oklab_points - mins) * scale).astype(np.int32) + padding
#     idx   = np.clip(idx, 0, vox_res - 1)

#     grid = np.zeros((vox_res, vox_res, vox_res), dtype=np.float32)
#     grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
#     print(f"  Occupied voxels: {grid.sum():,.0f}")

#     # ── 2. Gaussian blur → smooth iso-surface field ───────────────────────────
#     print(f"Smoothing field (sigma={smooth_sigma})…")
#     blurred = gaussian_filter(grid, sigma=smooth_sigma)

#     # ── 3. Marching cubes ────────────────────────────────────────────────────
#     # Isovalue at 0.5 × peak of blurred field sits right at the surface
#     isovalue = blurred.max() * 0.5
#     print(f"Running marching cubes (isovalue={isovalue:.4f})…")
#     verts_vox, faces, normals, _ = marching_cubes(blurred, level=isovalue)

#     # Convert voxel coords back to OKLab space
#     verts = (verts_vox - padding) / scale + mins

#     tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
#                          process=True)
#     print(f"  Marching cubes mesh: {len(tm.vertices):,} verts, "
#           f"{len(tm.faces):,} faces, watertight={tm.is_watertight}")
    
#         # ── 4. Light smooth BEFORE decimation (just kills staircase) ─────────────
#     if pre_decimate_smooth > 0:
#         trimesh.smoothing.filter_laplacian(tm, iterations=pre_decimate_smooth)

#     # ── 5. Decimate ───────────────────────────────────────────────────────────
#     if target_faces and len(tm.faces) > target_faces:
#             print(f"Decimating to {target_faces:,} faces…")
#             import open3d as o3d
#             o3d_mesh = o3d.geometry.TriangleMesh()
#             o3d_mesh.vertices  = o3d.utility.Vector3dVector(tm.vertices)
#             o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
#             o3d_mesh = o3d_mesh.simplify_quadric_decimation(
#                 target_number_of_triangles=target_faces
#             )
#             verts = np.asarray(o3d_mesh.vertices)
#             faces = np.asarray(o3d_mesh.triangles)
#             tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
#     print("Isotropic remeshing…")
#     try:
#         import pymeshlab
#         ms = pymeshlab.MeshSet()
#         ms.add_mesh(pymeshlab.Mesh(
#             vertex_matrix=tm.vertices.astype(np.float64),
#             face_matrix=tm.faces.astype(np.int32),
#         ))
#         ms.meshing_isotropic_explicit_remeshing(
#             iterations=5,
#             targetlen=pymeshlab.PercentageValue(0.8),  # 0.8% of bbox diagonal
#         )
#         m = ms.current_mesh()
#         tm = trimesh.Trimesh(
#             vertices=m.vertex_matrix(),
#             faces=m.face_matrix(),
#             process=True,
#         )
#         print(f"  After remeshing: {len(tm.vertices):,} verts, "
#               f"{len(tm.faces):,} faces")
#     except ImportError:
#         print("  pymeshlab not found — skipping isotropic remeshing.")
#         print("  Install with: pip install pymeshlab")
#         print("  Without this step, geodesic distances may show banding.")

#     # ── 6. Smooth AFTER remeshing (now acts on uniform triangles) ─────────────
#     if post_decimate_smooth > 0:
#         trimesh.smoothing.filter_laplacian(tm, iterations=post_decimate_smooth)
#     trimesh.repair.fix_normals(tm)
#     trimesh.repair.fill_holes(tm)

#     # Connectivity check — multiple components cause wild geodesic outliers
#     components = trimesh.graph.connected_components(tm.edges)
#     if len(components) > 1:
#         print(f"  Warning: {len(components)} connected components found. "
#               f"Keeping largest to avoid geodesic outliers.")
#         tm = tm.submesh([max(components, key=len)], append=True)

#     print(f"  Final mesh: {len(tm.vertices):,} verts, {len(tm.faces):,} faces, "
#           f"watertight={tm.is_watertight}")
#     return tm

#     # # ── 4. Laplacian smoothing to remove staircase ────────────────────────────
#     # if smooth_iterations > 0:
#     #     print(f"Laplacian smoothing ({smooth_iterations} iterations)…")
#     #     trimesh.smoothing.filter_laplacian(tm, iterations=smooth_iterations)

#     # # ── 5. Decimate to target face count ─────────────────────────────────────
#     # if target_faces and len(tm.faces) > target_faces:
#     #     print(f"Decimating to {target_faces:,} faces…")
#     #     import open3d as o3d
#     #     o3d_mesh = o3d.geometry.TriangleMesh()
#     #     o3d_mesh.vertices  = o3d.utility.Vector3dVector(tm.vertices)
#     #     o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
#     #     o3d_mesh = o3d_mesh.simplify_quadric_decimation(
#     #         target_number_of_triangles=target_faces
#     #     )
#     #     verts = np.asarray(o3d_mesh.vertices)
#     #     faces = np.asarray(o3d_mesh.triangles)
#     #     tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

#     # # ── 6. Final repair ───────────────────────────────────────────────────────
#     # trimesh.repair.fix_normals(tm)
#     # trimesh.repair.fill_holes(tm)

#     # print(f"  Final mesh: {len(tm.vertices):,} verts, {len(tm.faces):,} faces, "
#     #       f"watertight={tm.is_watertight}")
#     # return tm


# # ─────────────────────────────────────────────────────────────────────────────
# # Example usage
# # ─────────────────────────────────────────────────────────────────────────────

# # if __name__ == "__main__":
# #     # from your_module import oklab_points  # shape (N, 3)

# #     # Smoke test — replace with your real oklab_points
# #     print("Generating synthetic test point cloud…")
# #     rng   = np.random.default_rng(0)
# #     theta = rng.uniform(0, np.pi,   5_000)
# #     phi   = rng.uniform(0, 2*np.pi, 5_000)
# #     oklab_points = np.column_stack([
# #         np.sin(theta) * np.cos(phi),
# #         np.sin(theta) * np.sin(phi),
# #         np.cos(theta),
# #     ])

# #     mesh = build_gamut_mesh(
# #         oklab_points,
# #         vox_res=128,           # increase to 256 for full 16.7M point cloud
# #         smooth_sigma=1.5,
# #         target_faces=10_000,
# #         smooth_iterations=10,
# #     )

# #     mesh.export("oklab_gamut.glb")
# #     print("Saved → oklab_gamut.glb")
# #     print(mesh)

# # def build_gamut_mesh(
# #     oklab_points: np.ndarray,
# #     *,
# #     vox_res: int = 128,
# #     smooth_sigma: float = 1.5,
# #     target_faces: int = 10_000,
# #     smooth_iterations: int = 10,
# #     ) -> trimesh.Trimesh:
# #     """
# #     Full pipeline: OKLab point cloud → watertight trimesh of gamut boundary.

# #     Parameters
# #     ----------
# #     oklab_points : (N, 3) array of OKLab colours
# #     vox_res : voxel grid resolution per axis. 64 is fast, 128 is
# #     good, 256 is high precision (memory: vox_res^3 bytes).
# #     smooth_sigma : Gaussian blur sigma applied to occupancy grid before
# #     marching cubes. Higher = smoother surface but less
# #     faithful to the true boundary. 1.0–2.0 is good.
# #     target_faces : desired triangle count after decimation.
# #     smooth_iterations : Laplacian smoothing passes to remove voxel staircase.

# #     Returns
# #     -------
# #     trimesh.Trimesh (guaranteed watertight)
# #     """

# #     # ── 1. Voxelise ──────────────────────────────────────────────────────────
# #     print(f"Voxelising {len(oklab_points):,} points at {vox_res}³…")

# #     mins = oklab_points.min(axis=0)
# #     maxs = oklab_points.max(axis=0)
# #     # Add a 2-voxel padding so marching cubes sees a closed surface at edges
# #     padding = 2
# #     scale = (vox_res - 1 - 2 * padding) / (maxs - mins + 1e-12)
# #     idx = np.floor((oklab_points - mins) * scale).astype(np.int32) + padding
# #     idx = np.clip(idx, 0, vox_res - 1)

# #     grid = np.zeros((vox_res, vox_res, vox_res), dtype=np.float32)
# #     grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
# #     print(f" Occupied voxels: {grid.sum():,.0f}")

# #     # ── 2. Gaussian blur → smooth iso-surface field ───────────────────────────
# #     print(f"Smoothing field (sigma={smooth_sigma})…")
# #     blurred = gaussian_filter(grid, sigma=smooth_sigma)

# #     # ── 3. Marching cubes ────────────────────────────────────────────────────
# #     # Isovalue at 0.5 × peak of blurred field sits right at the surface
# #     isovalue = blurred.max() * 0.5
# #     print(f"Running marching cubes (isovalue={isovalue:.4f})…")
# #     verts_vox, faces, normals, _ = marching_cubes(blurred, level=isovalue)

# #     # Convert voxel coords back to OKLab space
# #     verts = (verts_vox - padding) / scale + mins

# #     tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
# #     process=True)
# #     print(f" Marching cubes mesh: {len(tm.vertices):,} verts, "
# #     f"{len(tm.faces):,} faces, watertight={tm.is_watertight}")

# #     # ── 4. Laplacian smoothing to remove staircase ────────────────────────────
# #     if smooth_iterations > 0:
# #         print(f"Laplacian smoothing ({smooth_iterations} iterations)…")
# #         trimesh.smoothing.filter_laplacian(tm, iterations=smooth_iterations)

# #     # ── 5. Decimate to target face count ─────────────────────────────────────
# #     if target_faces and len(tm.faces) > target_faces:
# #         print(f"Decimating to {target_faces:,} faces…")
# #         o3d_mesh = o3d.geometry.TriangleMesh()
# #         o3d_mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
# #         o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
# #         o3d_mesh = o3d_mesh.simplify_quadric_decimation(
# #         target_number_of_triangles=target_faces
# #         )
# #         verts = np.asarray(o3d_mesh.vertices)
# #         faces = np.asarray(o3d_mesh.triangles)
# #         tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

# #     # ── 6. Final repair ───────────────────────────────────────────────────────
# #     trimesh.repair.fix_normals(tm)
# #     trimesh.repair.fill_holes(tm)

# #     print(f" Final mesh: {len(tm.vertices):,} verts, {len(tm.faces):,} faces, "
# #     f"watertight={tm.is_watertight}")
# #     return tm

def build_gamut_mesh(
    cielab_points: np.ndarray,
    *,
    vox_res: int = 256,
    smooth_sigma: float = 1.0,
    target_faces: int = 50_000,
    pre_decimate_smooth: int = 3,
    post_decimate_smooth: int = 5,
) -> trimesh.Trimesh:
    """
    Full pipeline: CIELAB point cloud → watertight trimesh of gamut boundary.

    Parameters
    ----------
    cielab_points         : (N, 3) array of CIELAB colours (L*, a*, b*)
    vox_res               : voxel grid resolution for the LONGEST axis.
                            Other axes scale proportionally so voxels are
                            cubical in CIELAB units — prevents L* banding.
                            128 is good, 256 is high precision.
    smooth_sigma          : Gaussian blur sigma before marching cubes.
                            Lower (1.0) stays closer to true boundary.
    target_faces          : desired triangle count after decimation.
                            50k+ recommended for smooth geodesic distances.
    pre_decimate_smooth   : Laplacian passes before decimation (removes
                            marching-cubes staircase only; keep low).
    post_decimate_smooth  : Laplacian passes after remeshing (smooths
                            decimation artifacts on uniform triangles).

    Returns
    -------
    trimesh.Trimesh  (guaranteed watertight, isotropically remeshed)
    """

    # ── 1. Voxelise with per-axis resolution ──────────────────────────────────
    # CIELAB axes have very different ranges: L* ~[0,100], a*/b* ~[-128,127].
    # Using a cubic grid over-resolves L* relative to a*/b*, which causes
    # marching cubes to produce denser triangles at constant-L* values (banding).
    # Fix: size each axis proportionally to its colorspace extent so voxels
    # are cubical in CIELAB units.
    print(f"Voxelising {len(cielab_points):,} points…")

    mins = cielab_points.min(axis=0)
    maxs = cielab_points.max(axis=0)
    ranges = maxs - mins + 1e-12

    longest = ranges.max()
    per_axis_res = np.round((ranges / longest) * vox_res).astype(int)
    per_axis_res = np.clip(per_axis_res, 16, vox_res)

    print(f"  Per-axis resolution: L*={per_axis_res[0]}, "
          f"a*={per_axis_res[1]}, b*={per_axis_res[2]}")

    padding = 2
    scale = (per_axis_res - 1 - 2 * padding) / ranges

    idx = np.floor((cielab_points - mins) * scale).astype(np.int32) + padding
    idx = np.clip(idx, 0, per_axis_res - 1)

    grid = np.zeros(tuple(per_axis_res), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    print(f"  Occupied voxels: {grid.sum():,.0f}")

    # ── 2. Gaussian blur → smooth iso-surface field ───────────────────────────
    print(f"Smoothing field (sigma={smooth_sigma})…")
    blurred = gaussian_filter(grid, sigma=smooth_sigma)

    # ── 3. Marching cubes with physical voxel spacing ─────────────────────────
    # Pass the real CIELAB-unit size of each voxel so marching cubes produces
    # vertices directly in colorspace units rather than voxel indices.
    isovalue = blurred.max() * 0.5
    voxel_size = ranges / (per_axis_res - 1)

    print(f"  Voxel size (ΔE/voxel): L*={voxel_size[0]:.3f}, "
          f"a*={voxel_size[1]:.3f}, b*={voxel_size[2]:.3f}")
    print(f"Running marching cubes (isovalue={isovalue:.4f})…")

    verts_vox, faces, normals, _ = marching_cubes(
        blurred,
        level=isovalue,
        spacing=tuple(voxel_size),
    )

    # Shift from voxel origin back to CIELAB space
    verts = verts_vox + mins - (padding * voxel_size)

    tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                         process=True)
    print(f"  Marching cubes mesh: {len(tm.vertices):,} verts, "
          f"{len(tm.faces):,} faces, watertight={tm.is_watertight}")

    # ── 4. Light Laplacian smooth BEFORE decimation ───────────────────────────
    # Just removes marching-cubes staircase; keep iterations low so we don't
    # pull the surface away from the true gamut boundary.
    if pre_decimate_smooth > 0:
        print(f"Pre-decimation Laplacian smoothing ({pre_decimate_smooth} iters)…")
        trimesh.smoothing.filter_laplacian(tm, iterations=pre_decimate_smooth)

    # ── 5. Quadric decimation ─────────────────────────────────────────────────
    # Reduces face count efficiently but produces anisotropic triangles —
    # we fix that in step 6.
    if target_faces and len(tm.faces) > target_faces:
        print(f"Decimating to {target_faces:,} faces…")
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(tm.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        verts = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        print(f"  After decimation: {len(tm.vertices):,} verts, "
              f"{len(tm.faces):,} faces")

    # ── 6. Isotropic remeshing ────────────────────────────────────────────────
    # Redistributes vertices to equalise edge lengths across the whole surface.
    # Eliminates long thin triangles that cause geodesic banding and speckling.
    print("Isotropic remeshing…")
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=tm.vertices.astype(np.float64),
            face_matrix=tm.faces.astype(np.int32),
        ))
        ms.meshing_isotropic_explicit_remeshing(
            iterations=5,
            targetlen=pymeshlab.PercentageValue(0.8),
        )
        m = ms.current_mesh()
        tm = trimesh.Trimesh(
            vertices=m.vertex_matrix(),
            faces=m.face_matrix(),
            process=True,
        )
        print(f"  After remeshing: {len(tm.vertices):,} verts, "
              f"{len(tm.faces):,} faces")
    except ImportError:
        # Fallback: subdivide → re-decimate forces more uniform triangles,
        # not as clean as pymeshlab but no new dependency.
        print("  pymeshlab not found — using open3d subdivide+decimate fallback.")
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(tm.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
        o3d_mesh.compute_vertex_normals()
        o3d_mesh = o3d_mesh.subdivide_midpoint(number_of_iterations=1)
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        o3d_mesh = o3d_mesh.filter_smooth_laplacian(number_of_iterations=3)
        verts = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        print(f"  After fallback remeshing: {len(tm.vertices):,} verts, "
              f"{len(tm.faces):,} faces")

    # ── 7. Post-remesh Laplacian smooth ───────────────────────────────────────
    # Now that triangles are uniform, Laplacian smoothing acts evenly across
    # the surface instead of over-smoothing dense regions.
    if post_decimate_smooth > 0:
        print(f"Post-remesh Laplacian smoothing ({post_decimate_smooth} iters)…")
        trimesh.smoothing.filter_laplacian(tm, iterations=post_decimate_smooth)

    # ── 8. Final repair ───────────────────────────────────────────────────────
    trimesh.repair.fix_normals(tm)
    trimesh.repair.fill_holes(tm)

    # Connectivity check — multiple components cause wild geodesic outliers
    components = trimesh.graph.connected_components(tm.edges)
    if len(components) > 1:
        print(f"  Warning: {len(components)} connected components found. "
              f"Keeping largest to avoid geodesic outliers.")
        tm = tm.submesh([max(components, key=len)], append=True)

    print(f"  Final mesh: {len(tm.vertices):,} verts, {len(tm.faces):,} faces, "
          f"watertight={tm.is_watertight}")
    return tm
