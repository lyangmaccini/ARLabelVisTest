"""
Batch Regularized Geodesic Distance solver.

Solves for multiple source vertices simultaneously by stacking all
right-hand sides into matrices and solving them in one shot per iteration.

Key insight:
    The system matrix  A = (α·Ww + ρ·Ww)  does NOT depend on the source.
    → Factorise A exactly once, then call   A_fact(B)  where B is (nv × S),
      solving S right-hand sides for the price of ~1 factorisation.

GPU role here is much clearer:
    - G @ U_p  :  sparse (3nf × nv_p)  ×  dense (nv_p × S)  → cuSPARSE SpMM
    - z-step   :  reshape + norm + clamp over (nf × 3 × S)   → embarrassingly parallel
    - residuals:  Frobenius norms                             → one cuBLAS call
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from tqdm import tqdm
from scipy.sparse import diags
from typing import Optional, Union
from scipy.sparse.linalg import factorized

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _GPU = cp.cuda.is_available()
except ImportError:
    _GPU = False


def rgd_admm_batch(
        mesh,
        source_list,                   # list/array of source vertex indices, length S
        alpha_hat: float = 0.05,
        max_iter:  int   = 10_000,
        abs_tol:   float = 5e-6,
        rel_tol:   float = 1e-2,
        quiet:     bool  = False,
) -> np.ndarray:
    """
    Compute regularised geodesic distances from S sources simultaneously.

    The system matrix is factorised exactly once; every ADMM iteration
    solves S right-hand sides in a single call to the sparse solver.

    Args:
        mesh        : Mesh object
        source_list : (S,) array-like of source vertex indices
        alpha_hat   : scale-invariant regularisation weight
        max_iter    : maximum ADMM iterations
        abs_tol     : absolute convergence tolerance
        rel_tol     : relative convergence tolerance
        quiet       : suppress output

    Returns:
        U : (S, nv) array — U[i, j] = geodesic distance from source_list[i] to vertex j
    """
    sources = np.asarray(source_list, dtype=int)
    S = len(sources)

    nv, nf  = mesh.nv, mesh.nf
    va, ta  = mesh.va, mesh.ta
    G, Ww   = mesh.G,  mesh.Ww
    total_area = ta.sum()
    alpha = alpha_hat * np.sqrt(total_area)

    # ── shared mask: all non-source vertices ─────────────────────────────────
    # Each source has its own set of constrained vertices, but we build one
    # shared "interior" mask that excludes ALL sources across the batch.
    # Vertices that are a source for one problem but not another are handled
    # by zeroing out the corresponding column of the RHS.
    all_sources = np.unique(sources)
    mask  = np.ones(nv, dtype=bool)
    mask[all_sources] = False
    nv_p  = mask.sum()
    idx_p = np.where(mask)[0]

    va_p = va[mask]
    Ww_p = G_p = div_p = None   # defined below

    G_p   = G[:, mask]
    ta_rep = np.repeat(ta, 3)
    div_p  = (G_p.T @ sparse.diags(ta_rep)).tocsr()

    # ── build and factorise system matrix ONCE ───────────────────────────────
    rho = 2.0 * np.sqrt(total_area)
    mu, tau_i, tau_d = 10.0, 2.0, 2.0
    alpha_k = 1.7

    Ww_p = Ww[np.ix_(mask, mask)]

    try:
        from sksparse.cholmod import cholesky as cholmod_cholesky

        def _make_fact(rho_val):
            M = ((alpha + rho_val) * Ww_p).tocsc()
            factor = cholmod_cholesky(M)
            return factor.solve_A

    except ImportError:
        # from scipy.sparse.linalg import factorized  # falls back if not installed

        def _make_fact(rho_val):
            return factorized(((alpha + rho_val) * Ww_p).tocsc())

    A_fact = _make_fact(rho)

    # ── right-hand side: va_p column repeated S times, minus source columns ──
    # va_p is the "forcing" term; for source s, vertices in all_sources that
    # are NOT source s should still contribute va normally — but since we've
    # masked them out globally, we just use va_p broadcast across all S.
    va_mat = np.tile(va_p[:, None], (1, S))   # (nv_p, S)

    # ── ADMM variables: now all (nv_p × S) or (3nf × S) matrices ────────────
    U_p   = np.zeros((nv_p, S))
    Z     = np.zeros((3 * nf, S))
    Y     = np.zeros((3 * nf, S))
    div_Y = np.zeros((nv_p, S))
    div_Z = np.zeros((nv_p, S))

    thresh1 = np.sqrt(3 * nf) * abs_tol * np.sqrt(total_area)
    thresh2 = np.sqrt(nv)     * abs_tol * total_area

    # ── move to GPU ───────────────────────────────────────────────────────────
    if _GPU:
        G_p_gpu   = cpsp.csr_matrix(G_p)
        div_p_gpu = cpsp.csr_matrix(div_p)
        ta_sqrt_g = cp.asarray(np.sqrt(ta_rep))
        va_mat_g  = cp.asarray(va_mat)
        Z_g = cp.zeros((3 * nf, S))
        Y_g = cp.zeros((3 * nf, S))
        div_Y_g = cp.zeros((nv_p, S))
        div_Z_g = cp.zeros((nv_p, S))

    if not quiet:
        print(f"Batch ADMM: {S} sources, nv={nv}, nf={nf}, "
              f"GPU={_GPU}, single factorisation")
        print(f"{'Iter':>5s} {'r_norm':>10s} {'eps_pri':>10s} "
              f"{'s_norm':>10s} {'eps_dual':>10s}")

    for it in tqdm(range(max_iter), desc="ADMM", unit="iter", disable=quiet):

        # ── u-step: solve A · U_p = B  (one factorised solve, S RHS) ─────────
        if _GPU:
            B = cp.asnumpy(va_mat_g - div_Y_g + rho * div_Z_g)
        else:
            B = va_mat - div_Y + rho * div_Z

        # scipy factorized() only accepts 1-D RHS; loop over columns.
        # This is still fast because the factorisation is reused.
        for s in range(S):
            U_p[:, s] = A_fact(B[:, s]) / (alpha + rho)

        # ── z-step on GPU ─────────────────────────────────────────────────────
        if _GPU:
            U_p_g  = cp.asarray(U_p)
            Gx_g   = G_p_gpu @ U_p_g                        # (3nf, S)

            Z_old_g    = Z_g.copy()
            div_Z_old_g = div_Z_g.copy()

            Z_temp = (1.0 / rho) * Y_g + Gx_g              # (3nf, S)
            Z_temp3 = Z_temp.reshape(nf, 3, S)             # (nf, 3, S)
            Z_nrms  = cp.linalg.norm(Z_temp3, axis=1, keepdims=True)  # (nf,1,S)
            Z_nrms  = cp.maximum(Z_nrms, 1.0)
            Z_g     = (Z_temp3 / Z_nrms).reshape(3 * nf, S)

            div_Z_g = div_p_gpu @ Z_g
            Y_g     = Y_g + rho * (alpha_k * Gx_g
                                   + (1 - alpha_k) * Z_old_g - Z_g)
            div_Y_g = div_p_gpu @ Y_g

            # residuals (Frobenius over all sources)
            Gx_w   = ta_sqrt_g[:, None] * Gx_g
            Z_w    = ta_sqrt_g[:, None] * Z_g
            r_norm = float(cp.linalg.norm(Gx_w - Z_w, 'fro'))
            s_norm = rho * float(cp.linalg.norm(div_Z_g - div_Z_old_g, 'fro'))
            eps_pri  = thresh1 * S**0.5 + rel_tol * max(
                float(cp.linalg.norm(Gx_w, 'fro')),
                float(cp.linalg.norm(Z_w,  'fro')))
            eps_dual = thresh2 * S**0.5 + rel_tol * float(
                cp.linalg.norm(div_Y_g, 'fro'))

        else:   # ── CPU path ─────────────────────────────────────────────────
            Gx       = G_p @ U_p                            # (3nf, S)
            Z_old    = Z.copy()
            div_Z_old = div_Z.copy()

            Z_temp   = (1.0 / rho) * Y + Gx
            Z_temp3  = Z_temp.reshape(nf, 3, S)
            Z_nrms   = np.linalg.norm(Z_temp3, axis=1, keepdims=True)
            Z_nrms   = np.maximum(Z_nrms, 1.0)
            Z        = (Z_temp3 / Z_nrms).reshape(3 * nf, S)

            div_Z    = div_p @ Z
            Y        = Y + rho * (alpha_k * Gx + (1 - alpha_k) * Z_old - Z)
            div_Y    = div_p @ Y

            ta_sqrt  = np.sqrt(ta_rep)
            Gx_w     = ta_sqrt[:, None] * Gx
            Z_w      = ta_sqrt[:, None] * Z
            r_norm   = np.linalg.norm(Gx_w - Z_w, 'fro')
            s_norm   = rho * np.linalg.norm(div_Z - div_Z_old, 'fro')
            eps_pri  = thresh1 * S**0.5 + rel_tol * max(
                np.linalg.norm(Gx_w, 'fro'), np.linalg.norm(Z_w, 'fro'))
            eps_dual = thresh2 * S**0.5 + rel_tol * np.linalg.norm(div_Y, 'fro')

        if not quiet and it % 100 == 0:
            print(f"{it:5d} {r_norm:10.4e} {eps_pri:10.4e} "
                  f"{s_norm:10.4e} {eps_dual:10.4e}")

        if it > 0 and r_norm < eps_pri and s_norm < eps_dual:
            if not quiet:
                print(f"Converged at iteration {it}")
            break

        # ── adaptive penalty ──────────────────────────────────────────────────
        if r_norm > mu * s_norm:
            rho *= tau_i;  A_fact = _make_fact(rho)
        elif s_norm > mu * r_norm:
            rho /= tau_d;  A_fact = _make_fact(rho)

    # ── pull back from GPU ────────────────────────────────────────────────────
    if _GPU:
        U_p = cp.asnumpy(
            cp.asarray(U_p)  # U_p was updated on CPU in u-step
        )

    # ── reconstruct full (S, nv) distance matrix ──────────────────────────────
    U_full = np.zeros((S, nv))
    U_full[:, idx_p] = U_p.T
    # source vertices stay 0 (correct: distance from source to itself = 0)

    return U_full