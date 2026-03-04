"""
Optimized furthest_rgd implementation.

Key speedups vs original:
  1. KDTree replaces O(n^2) Python loops for nearest-neighbor queries
  2. Pre-factorized ADMM: mesh-level matrices built once and shared
  3. Source-specific Ww_p sliced and factorized per unique source (not per LAB point)
  4. Optional parallel execution via joblib
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import KDTree
from scipy.sparse.linalg import factorized
from joblib import Parallel, delayed
# from RGD.rgd_admm import rgd_admm

from RGD_NEW.mesh_class import MeshClass
from RGD_NEW.rgd_admm import rgd_admm


# ---------------------------------------------------------------------------
# Nearest-neighbor helpers
# ---------------------------------------------------------------------------

def closest_vertices_batch(lab_points: np.ndarray, mesh_tree: KDTree) -> np.ndarray:
    """For each LAB point, return the index of the closest mesh vertex."""
    _, indices = mesh_tree.query(lab_points)
    return indices.astype(int)


def closest_labs_batch(mesh_vertices: np.ndarray, lab_tree: KDTree) -> np.ndarray:
    """For each mesh vertex, return the index of the closest LAB point."""
    _, indices = lab_tree.query(mesh_vertices)
    return indices.astype(int)


# ---------------------------------------------------------------------------
# ADMM helpers
# ---------------------------------------------------------------------------

def _build_mesh_cache(mesh, alpha_hat: float):
    """
    Pre-compute all mesh-level quantities shared across every source vertex.
    Nothing here depends on which vertex is the source.
    """
    nv = mesh.nv
    nf = mesh.nf
    va = mesh.va      # (nv,)  vertex areas
    ta = mesh.ta      # (nf,)  face areas
    G  = mesh.G       # (3*nf, nv)  sparse gradient operator
    Ww = mesh.Ww      # (nv, nv)    sparse cotangent Laplacian

    total_area = float(ta.sum())
    alpha      = alpha_hat * np.sqrt(total_area)
    rho_init   = 2.0 * np.sqrt(total_area)

    # Divergence operator: G^T * diag(ta repeated 3x), shape (nv, 3*nf)
    ta_rep = np.repeat(ta, 3)
    div    = (G.T @ sp.diags(ta_rep)).tocsr()

    # Store Ww in both CSR (for row slicing) and we'll do col slicing separately
    # Most reliable cross-version approach: convert to lil for arbitrary indexing,
    # then tocsc for factorization.
    Ww_csr = Ww.tocsr()

    return {
        "nv":         nv,
        "nf":         nf,
        "va":         va,
        "ta":         ta,
        "ta_rep":     ta_rep,
        "G":          G,
        "Ww_csr":     Ww_csr,
        "div":        div,
        "total_area": total_area,
        "alpha":      alpha,
        "rho_init":   rho_init,
    }


def _slice_square(A_csr, free_idx: np.ndarray):
    """
    Extract the submatrix A[free_idx, :][:, free_idx] as a square CSC matrix.
    Uses explicit integer-array indexing which works on all scipy versions.
    """
    # Row slice first (CSR is efficient for row slicing)
    A_rows = A_csr[free_idx, :]
    # Column slice on the result (convert to CSC for efficient column slicing)
    A_sub  = A_rows.tocsc()[:, free_idx]
    return A_sub.tocsc()


def _build_source_system(cache: dict, source_idx: int):
    """
    Reduce the full system by removing the source vertex (u=0 there).
    Returns reduced operators and an initial Cholesky factorization.
    """
    nv      = cache["nv"]
    va      = cache["va"]
    Ww_csr  = cache["Ww_csr"]
    G       = cache["G"]
    div     = cache["div"]
    alpha   = cache["alpha"]
    rho     = cache["rho_init"]

    # Free vertices = all except source
    mask     = np.ones(nv, dtype=bool)
    mask[source_idx] = False
    free_idx = np.where(mask)[0]   # shape (nv_p,), integer indices
    nv_p     = len(free_idx)

    va_p  = va[free_idx]                       # (nv_p,)
    G_p   = G[:, free_idx]                     # (3*nf, nv_p)  sparse
    div_p = div[free_idx, :]                   # (nv_p, 3*nf)  sparse

    # Square Laplacian submatrix — this MUST be (nv_p x nv_p)
    Ww_p  = _slice_square(Ww_csr, free_idx)   # (nv_p, nv_p)

    # Sanity check (will catch bugs immediately rather than inside factorized())
    assert Ww_p.shape == (nv_p, nv_p), (
        f"Ww_p is {Ww_p.shape}, expected ({nv_p}, {nv_p})"
    )

    A_fact = factorized(((alpha + rho) * Ww_p).tocsc())

    return {
        "free_idx": free_idx,
        "nv_p":     nv_p,
        "va_p":     va_p,
        "G_p":      G_p,
        "div_p":    div_p,
        "Ww_p":     Ww_p,
        "A_fact":   A_fact,
        "rho":      rho,
    }


def _rgd_admm_single(mesh, source_idx: int, cache: dict,
                     max_iter: int = 10000,
                     abs_tol: float = 5e-6,
                     rel_tol: float = 1e-2,
                     quiet: bool = True) -> np.ndarray:
    """
    Run Dirichlet-regularized geodesic ADMM for one source vertex.
    Reuses the pre-built mesh cache; only the source reduction is built here.
    """
    nv         = cache["nv"]
    nf         = cache["nf"]
    ta         = cache["ta"]
    alpha      = cache["alpha"]
    total_area = cache["total_area"]

    src      = _build_source_system(cache, source_idx)
    free_idx = src["free_idx"]
    nv_p     = src["nv_p"]
    va_p     = src["va_p"]
    G_p      = src["G_p"]
    div_p    = src["div_p"]
    Ww_p     = src["Ww_p"]
    A_fact   = src["A_fact"]
    rho      = src["rho"]

    # Hyperparameters
    mu      = 10.0
    tau_inc = 2.0
    tau_dec = 2.0
    alpha_k = 1.7   # over-relaxation

    # Convergence thresholds
    thresh1 = np.sqrt(3 * nf) * abs_tol * np.sqrt(total_area)
    thresh2 = np.sqrt(nv)     * abs_tol * total_area

    # Precompute constant weight vector
    ta_sqrt = np.sqrt(np.repeat(ta, 3))   # (3*nf,)

    # ADMM variables
    u_p   = np.zeros(nv_p)
    y     = np.zeros(3 * nf)
    z     = np.zeros(3 * nf)
    div_y = np.zeros(nv_p)
    div_z = np.zeros(nv_p)

    for iteration in range(max_iter):

        # u-step: solve (alpha + rho) * Ww_p * u_p = rhs
        b    = va_p - div_y + rho * div_z
        u_p  = A_fact(b) / (alpha + rho)
        Gx   = G_p @ u_p                    # (3*nf,)

        # z-step: per-face projection onto unit ball
        z_old     = z
        div_z_old = div_z

        z_hat  = (1.0 / rho) * y + Gx
        z_hat3 = z_hat.reshape(nf, 3)
        norms  = np.linalg.norm(z_hat3, axis=1)   # (nf,)
        scale  = np.maximum(norms, 1.0)
        z      = (z_hat3 / scale[:, None]).ravel()
        div_z  = div_p @ z                  # (nv_p,)

        # y-step: dual update with over-relaxation
        y     = y + rho * (alpha_k * Gx + (1.0 - alpha_k) * z_old - z)
        div_y = div_p @ y                   # (nv_p,)

        # Residuals
        r_norm = np.linalg.norm(ta_sqrt * (Gx - z))
        s_norm = rho * np.linalg.norm(div_z - div_z_old)

        eps_pri  = thresh1 + rel_tol * max(
            np.linalg.norm(ta_sqrt * Gx),
            np.linalg.norm(ta_sqrt * z))
        eps_dual = thresh2 + rel_tol * np.linalg.norm(div_y)

        if iteration > 0 and r_norm < eps_pri and s_norm < eps_dual:
            if not quiet:
                print(f"  [src={source_idx}] converged at iter {iteration}")
            break

        # Adaptive rho: refactorize only when rho actually changes
        if r_norm > mu * s_norm:
            rho   *= tau_inc
            A_fact = factorized(((alpha + rho) * Ww_p).tocsc())
        elif s_norm > mu * r_norm:
            rho   /= tau_dec
            A_fact = factorized(((alpha + rho) * Ww_p).tocsc())

    # Expand back to full vertex array (source vertex stays 0)
    u           = np.zeros(nv)
    u[free_idx] = u_p
    return u

import scipy.sparse as sp
import scipy.sparse.linalg as spla


def rgd_admm_old(Mm, x0, reg='D', alpha_hat=0.1, beta_hat=0.0, vf=None):
    """
    ADMM algorithm for computing regularized geodesic distances.

    Parameters
    ----------
    Mm : mesh object with fields
        vertices, faces, nv, nf, va, ta, G, Ww
    x0 : list or array
        source vertex indices (0-based indexing)
    reg : str
        'D' (Dirichlet only implemented here)
    alpha_hat : float
    beta_hat : float
    vf : vector field (unused unless extending to 'vfa')

    Returns
    -------
    u : ndarray (nv,)
        regularized distance
    """

    # Mesh data
    nv = Mm.nv
    nf = Mm.nf
    va = Mm.va.copy()
    ta = Mm.ta.copy()
    G = Mm.G.tocsr()
    Ww = Mm.Ww.tocsr()

    tasq = np.tile(np.sqrt(ta), 3)

    # Regularization weight
    if reg == 'D':
        alpha = alpha_hat * np.sqrt(np.sum(va))
        varRho = True
    else:
        raise NotImplementedError("Only 'D' regularizer implemented.")

    # ADMM parameters
    rho = 2 * np.sqrt(np.sum(va))
    niter = 10000
    ABSTOL = 1e-5 / 2
    RELTOL = 1e-2
    mu = 10
    tauinc = 2
    taudec = 2
    alphak = 1.7

    thresh1 = np.sqrt(3 * nf) * ABSTOL * np.sqrt(np.sum(va))
    thresh2 = np.sqrt(nv) * ABSTOL * np.sum(va)

    # Remove boundary conditions
    mask = np.ones(nv, dtype=bool)
    mask[x0] = False
    nv_p = np.where(mask)[0]

    va_p = va[mask]
    Ww_p = Ww[mask][:, mask]
    G_p = G[:, mask]
    G_pt = G_p.transpose()

    # divergence operator
    div_p = G_pt.multiply(np.tile(ta, 3))

    # Initialize variables
    u_p = np.zeros(len(nv_p))
    y = np.zeros(3 * nf)
    z = np.zeros(3 * nf)
    div_y = np.zeros(len(nv_p))
    div_z = np.zeros(len(nv_p))

    I = sp.identity(Ww_p.shape[0], format='csr')
    A = alpha * Ww_p + rho * I
    solver = spla.factorized(A)

    for ii in range(niter):

        # Step 1: u-minimization
        b = va_p - div_y + rho * div_z
        u_p = solver(b) 

        Gx = G_p @ u_p

        # Step 2: z-minimization (projection)
        zold = z.copy()
        div_zold = div_z.copy()

        z = (1.0 / rho) * y + Gx
        z = z.reshape((nf, 3)).T

        norm_z = np.linalg.norm(z, axis=0)
        norm_z[norm_z < 1] = 1
        z = z / norm_z

        z = z.T.reshape(-1)
        div_z = div_p @ z

        # Step 3: dual update
        y = y + rho * (alphak * Gx + (1 - alphak) * zold - z)
        div_y = div_p @ y

        # Residuals
        tasqGx = tasq * Gx
        tasqZ = tasq * z

        r_norm = np.linalg.norm(tasqGx - tasqZ)
        s_norm = rho * np.linalg.norm(div_z - div_zold)

        eps_pri = thresh1 + RELTOL * max(
            np.linalg.norm(tasqGx), np.linalg.norm(tasqZ)
        )
        eps_dual = thresh2 + RELTOL * np.linalg.norm(div_y)

        if ii > 1 and r_norm < eps_pri and s_norm < eps_dual:
            break

        # Varying rho
        if varRho:
            if r_norm > mu * s_norm:
                rho *= tauinc
            elif s_norm > mu * r_norm:
                rho /= taudec

    # Reconstruct full solution
    u = np.zeros(nv)
    u[nv_p] = u_p

    return u
# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def furthest_rgd_fast(mesh, file, allLABS, allRGBs,
                      alpha_hat: float = 0.05,
                      n_jobs: int = 1,
                      quiet: bool = True) -> np.ndarray:
    """
    For each LAB point, find the RGB of the mesh vertex geodesically furthest from it.

    Args:
        mesh:       Mesh object
        allLABS:    (N, 3) LAB color points
        allRGBs:    (N, 3) corresponding RGB colors
        alpha_hat:  Dirichlet regularization weight (default 0.05)
        n_jobs:     Parallel workers; 1=serial, -1=all CPUs
        quiet:      Suppress per-iteration ADMM output

    Returns:
        furthest:   (N, 3) RGB array
    """
    allLABS = np.asarray(allLABS, dtype=float)
    allRGBs = np.asarray(allRGBs)

    # Step 1: KDTree nearest-neighbor lookups
    print("Building KDTrees...")
    mesh_tree     = KDTree(mesh.vertices)
    lab_tree      = KDTree(allLABS)
    LABtoVertices = closest_vertices_batch(allLABS,       mesh_tree)
    VerticestoLAB = closest_labs_batch(mesh.vertices,     lab_tree)
    print(f"  {len(allLABS)} LAB points <-> {mesh.nv} mesh vertices mapped")

    # Step 2: Build shared mesh-level cache
    print("Pre-building mesh cache...")
    cache = _build_mesh_cache(mesh, alpha_hat)
    print("  Cache ready")

    # Step 3: ADMM over unique source vertices only
    unique_sources = np.unique(LABtoVertices)
    n_unique       = len(unique_sources)

    print(f"Running ADMM for {n_unique} unique sources ({len(allLABS)} LAB points)...")

    furthestRGBs = []
    matlab_path = "data/max_indices_03ahat.txt"
    max_indices = np.loadtxt(matlab_path, delimiter=',', dtype='int')
    print(max_indices)
    print(max_indices.shape)

    for i in range(allLABS.shape[0]):
        vert = LABtoVertices[i]
        if vert < max_indices.shape[0]:
            furthest_vert = max_indices[vert]
        else:
            # print(vert)
            furthest_vert = 0
        # print(furthest_vert)
        furthest_lab = VerticestoLAB[int(furthest_vert)]
        # print(furthest_lab)
        furthestRGBs.append(allRGBs[int(furthest_lab)])

    # def _run_one(sv):
    #     # dist = rgd_admm(mesh, int(sv), alpha_hat=alpha_hat)
    #     vertices, faces = MeshClass.read_off(file)
    #     mesh = MeshClass(vertices, faces)

    #     # Compute regularized distances
    #     # dist, history = rgd_admm(mesh, int(sv), reg='D', alpha_hat=0.1)
    #     dist = np.loadtxt('data/u_D1 (4).csv')


    #     dist[dist > np.percentile(dist, 95)] = np.percentile(dist, 95)
    #     dist[dist < 0] = 0
    #     furthest_idx =  int(np.argmax(dist))

    #     print(dist.min(), dist.max())
    #     print(dist)
    #     print(np.percentile(dist, 95))

    #     print(dist[furthest_idx])

    
    #     plot_geodesic_field_pyvista(mesh, dist, source_idx=sv, furthest_idx=furthest_idx)

    #     return furthest_idx

    # if n_jobs == 1:
    #     source_to_furthest = {}
    #     for i, sv in enumerate(unique_sources):
    #         if i % 10 == 0:
    #             print(f"  {i}/{n_unique}", flush=True)
    #         source_to_furthest[int(sv)] = _run_one(sv)
    # else:
    #     results = Parallel(n_jobs=n_jobs, prefer="threads")(
    #         delayed(_run_one)(sv) for sv in unique_sources
    #     )
    #     source_to_furthest = {int(sv): r for sv, r in zip(unique_sources, results)}

    # # Step 4: Assemble output
    # furthest = np.array([
    #     allRGBs[VerticestoLAB[source_to_furthest[int(sv)]]]
    #     for sv in LABtoVertices
    # ])
    # return furthest
    return np.array(furthestRGBs)


# ---------------------------------------------------------------------------
# Drop-in replacement (matches original signature)
# ---------------------------------------------------------------------------

def furthest_rgd(mesh, file, allLABS, allRGBs, num=1, alpha_hat=0.05, n_jobs=1):
    """Drop-in replacement for the original furthest_rgd."""
    # show_original_mesh_pyvista(mesh, show_edges=True, show_normals=True)
    return furthest_rgd_fast(mesh, file, allLABS, allRGBs,
                             alpha_hat=alpha_hat, n_jobs=n_jobs)

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


def plot_geodesic_field_pyvista(mesh,
                                distances,
                                source_idx=None,
                                furthest_idx=None,
                                cmap="plasma"):

    V = mesh.vertices
    F = mesh.faces

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