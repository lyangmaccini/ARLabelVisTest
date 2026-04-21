import numpy as np
import trimesh
import trimesh.smoothing
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import open3d as o3d

def pointsToMesh(allLABs, sigma=0.25, vox=256, pre_decimate_smooth: int = 3, post_decimate_smooth: int = 5, target_faces = 50000):
    print(f"Input length: {len(allLABs):,}")

    mins = allLABs.min(axis=0)
    maxs = allLABs.max(axis=0)
    ranges = maxs - mins + 1e-12

    longest = ranges.max()
    per_axis_res = np.round((ranges / longest) * vox).astype(int)
    per_axis_res = np.clip(per_axis_res, 16, vox)

    padding = 2
    scale = (per_axis_res - 1 - 2 * padding) / ranges

    idx = np.floor((allLABs - mins) * scale).astype(np.int32) + padding
    idx = np.clip(idx, 0, per_axis_res - 1)

    grid = np.zeros(tuple(per_axis_res), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    blurred = gaussian_filter(grid, sigma=sigma)

    isovalue = blurred.max() * 0.5
    voxel_size = ranges / (per_axis_res - 1)

    verts_vox, faces, normals, _ = marching_cubes(blurred, level=isovalue, spacing=tuple(voxel_size))

    verts = verts_vox + mins - (padding * voxel_size)

    tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    if pre_decimate_smooth > 0:
        trimesh.smoothing.filter_laplacian(tm, iterations=pre_decimate_smooth)

    if target_faces and len(tm.faces) > target_faces:
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

    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=tm.vertices.astype(np.float64), face_matrix=tm.faces.astype(np.int32)))
    ms.meshing_isotropic_explicit_remeshing(iterations=5, targetlen=pymeshlab.PercentageValue(0.8))
    m = ms.current_mesh()
    tm = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=True)

    if post_decimate_smooth > 0:
        trimesh.smoothing.filter_laplacian(tm, iterations=post_decimate_smooth)

    trimesh.repair.fix_normals(tm)
    trimesh.repair.fill_holes(tm)

    components = trimesh.graph.connected_components(tm.edges)
    if len(components) > 1:
        print(f"WARNING: {len(components)} connected components found")
        tm = tm.submesh([max(components, key=len)], append=True)

    print(f"Final mesh: {len(tm.vertices):,} vertices, {len(tm.faces):,} faces, watertight={tm.is_watertight}")
    return tm
    
    
def insideMesh(point: np.array, mesh: trimesh.Trimesh):
    containment = mesh.contains([point])
    return containment[0]
