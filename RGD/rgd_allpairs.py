"""
All-Pairs Regularized Geodesic Distances using ADMM

Computes the full distance matrix U where U[i,j] is the regularized geodesic
distance from vertex i to vertex j.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve, factorized
from typing import Tuple, Optional


def rgd_allpairs(mesh,
                 alpha_hat: float = 0.03,
                 max_iter: int = 20000,
                 abs_tol: float = 1e-6,
                 rel_tol: float = 2e-4,
                 quiet: bool = False,
                 return_history: bool = False):
    """
    Compute all-pairs regularized geodesic distances.
    
    Uses Dirichlet regularization only (most efficient for all-pairs).
    
    Args:
        mesh: Mesh object
        alpha_hat: Scale-invariant regularization weight
        max_iter: Maximum ADMM iterations
        abs_tol: Absolute convergence tolerance
        rel_tol: Relative convergence tolerance
        quiet: If True, suppress iteration output
        return_history: If True, return convergence history
        
    Returns:
        U: All-pairs distance matrix (nv x nv), U[i,j] = distance from i to j
        history: (optional) Convergence history dictionary
    """
    # Extract mesh data
    nv = mesh.nv
    nf = mesh.nf
    va = mesh.va
    ta = mesh.ta
    G = mesh.G
    Ww = mesh.Ww
    
    total_area = ta.sum()
    alpha = alpha_hat * np.sqrt(total_area)
    
    # Build sparse diagonal matrices
    va_mat = diags(va)
    va_inv = 1.0 / va
    va_sqrt = np.sqrt(va)
    ta_repeated = np.repeat(ta, 3)
    ta_sqrt = np.sqrt(ta_repeated)
    ta_mat = diags(ta_repeated)
    
    # Divergence operator
    div = G.T @ ta_mat
    
    # ADMM parameters
    rho1 = 2.0 * np.sqrt(total_area)  # For gradient constraints
    rho2 = 10.0 / np.sqrt(total_area)  # For consensus constraints
    mu = 10.0
    tau_inc = 2.0
    tau_dec = 2.0
    alpha_k = 1.7  # Over-relaxation
    
    var_rho = True  # Use varying penalty parameters
    
    # Convergence thresholds
    thresh1 = np.sqrt(3 * nf) * abs_tol * total_area
    thresh2 = np.sqrt(nv) * abs_tol * (total_area ** 2)
    thresh3 = np.sqrt(nv) * abs_tol * (total_area ** 1.5)
    thresh4 = np.sqrt(nv) * abs_tol * total_area
    
    # Initialize variables
    X = np.zeros((nv, nv))  # Distances (column gradients)
    R = np.zeros((nv, nv))  # Distances (row gradients)
    U = np.zeros((nv, nv))  # Consensus distance matrix
    Z = np.zeros((3 * nf, nv))  # Auxiliary for G·X
    Q = np.zeros((3 * nf, nv))  # Auxiliary for G·R
    Y = np.zeros((3 * nf, nv))  # Dual for Z
    S = np.zeros((3 * nf, nv))  # Dual for Q
    H = np.zeros((nv, nv))  # Dual for X-U consensus
    K = np.zeros((nv, nv))  # Dual for R-U^T consensus
    
    div_Z = np.zeros((nv, nv))
    div_Q = np.zeros((nv, nv))
    div_Y = np.zeros((nv, nv))
    div_S = np.zeros((nv, nv))
    
    # History
    history = {
        'r_norm': [], 's_norm': [], 'eps_pri': [], 'eps_dual': [],
        'r_norm2': [], 's_norm2': [], 'eps_pri2': [], 'eps_dual2': [],
        'r_xr1': [], 'r_xr2': [], 's_xr': [],
        'eps_pri_xr1': [], 'eps_pri_xr2': [], 'eps_dual_xr': []
    }
    
    if not quiet:
        print(f"{'Iter':>5s} | {'r1':>8s} {'e1':>8s} | {'s1':>8s} {'d1':>8s} | "
              f"{'r2':>8s} {'e2':>8s} | {'s2':>8s} {'d2':>8s} | "
              f"{'xr1':>8s} {'xr2':>8s} | {'sxr':>8s}")
    
    # Pre-factorize
    rho_changed = True
    if not var_rho:
        A_matrix = (alpha + rho1) * Ww + rho2 * va_mat
        A_fact = factorized(A_matrix.tocsc())
    
    # ADMM iterations
    for iteration in range(max_iter):
        # Step 1: X and R minimization
        # Build right-hand sides
        va_mat_full = va[:, np.newaxis] * np.ones((nv, nv))
        bx = (0.5 * va_mat_full * va_inv - div_Y + rho1 * div_Z 
              - va[:, np.newaxis] * H + rho2 * va[:, np.newaxis] * U)
        br = (0.5 * va_mat_full.T * va_inv - div_S + rho1 * div_Q 
              - va[:, np.newaxis] * K + rho2 * va[:, np.newaxis] * U.T)
        
        if var_rho and rho_changed:
            A_matrix = (alpha + rho1) * Ww + rho2 * va_mat
            A_fact = factorized(A_matrix.tocsc())
            rho_changed = False
        
        # Solve for each column
        X = np.column_stack([A_fact(bx[:, i]) for i in range(nv)])
        R = np.column_stack([A_fact(br[:, i]) for i in range(nv)])
        
        Gx = G @ X
        Gr = G @ R
        
        # Step 2: Z, Q, U minimization
        Z_old = Z.copy()
        div_Z_old = div_Z.copy()
        
        Z = (1.0 / rho1) * Y + Gx
        Z = Z.reshape(nf, 3, nv)
        Z_norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z_norms = np.maximum(Z_norms, 1.0)
        Z = Z / Z_norms
        Z = Z.reshape(3 * nf, nv)
        
        div_Z = div @ Z
        
        Q_old = Q.copy()
        div_Q_old = div_Q.copy()
        
        Q = (1.0 / rho1) * S + Gr
        Q = Q.reshape(nf, 3, nv)
        Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
        Q_norms = np.maximum(Q_norms, 1.0)
        Q = Q / Q_norms
        Q = Q.reshape(3 * nf, nv)
        
        div_Q = div @ Q
        
        U_old = U.copy()
        U1 = 0.5 * ((1.0 / rho2) * (H + K.T) + X + R.T)
        
        # Enforce symmetry and non-negativity
        U = U1 - np.diag(np.diag(U1))  # Zero diagonal
        U = np.maximum(U, 0)  # Non-negative
        
        # Step 3: Dual updates (with over-relaxation)
        Y = Y + rho1 * (alpha_k * Gx + (1 - alpha_k) * Z_old - Z)
        S = S + rho1 * (alpha_k * Gr + (1 - alpha_k) * Q_old - Q)
        H = H + rho2 * (alpha_k * X + (1 - alpha_k) * U_old - U)
        K = K + rho2 * (alpha_k * R + (1 - alpha_k) * U_old.T - U.T)
        
        div_Y = div @ Y
        div_S = div @ S
        
        # Compute residuals (weighted by areas)
        Gx_w = ta_sqrt[:, np.newaxis] * Gx * va_sqrt
        Z_w = ta_sqrt[:, np.newaxis] * Z * va_sqrt
        r_norm = np.linalg.norm(Gx_w - Z_w, 'fro')
        eps_pri = thresh1 + rel_tol * max(np.linalg.norm(Gx_w, 'fro'), 
                                          np.linalg.norm(Z_w, 'fro'))
        s_norm = rho1 * np.linalg.norm((div_Z - div_Z_old) * va, 'fro')
        eps_dual = thresh2 + rel_tol * np.linalg.norm(div_Y * va, 'fro')
        
        Gr_w = ta_sqrt[:, np.newaxis] * Gr * va_sqrt
        Q_w = ta_sqrt[:, np.newaxis] * Q * va_sqrt
        r_norm2 = np.linalg.norm(Gr_w - Q_w, 'fro')
        eps_pri2 = thresh1 + rel_tol * max(np.linalg.norm(Gr_w, 'fro'), 
                                           np.linalg.norm(Q_w, 'fro'))
        s_norm2 = rho1 * np.linalg.norm((div_Q - div_Q_old) * va, 'fro')
        eps_dual2 = thresh2 + rel_tol * np.linalg.norm(div_S * va, 'fro')
        
        # Consensus residuals
        r_xr1 = np.linalg.norm(va_sqrt[:, np.newaxis] * (X - U) * va_sqrt, 'fro')
        r_xr2 = np.linalg.norm(va_sqrt[:, np.newaxis] * (R - U.T) * va_sqrt, 'fro')
        eps_pri_xr1 = thresh3 + rel_tol * min(
            np.linalg.norm(va_sqrt[:, np.newaxis] * X * va_sqrt, 'fro'),
            np.linalg.norm(va_sqrt[:, np.newaxis] * U * va_sqrt, 'fro'))
        eps_pri_xr2 = thresh3 + rel_tol * min(
            np.linalg.norm(va_sqrt[:, np.newaxis] * R * va_sqrt, 'fro'),
            np.linalg.norm(va_sqrt[:, np.newaxis] * U.T * va_sqrt, 'fro'))
        
        s_xr = np.sqrt(2) * rho2 * np.linalg.norm(
            va_sqrt[:, np.newaxis] * (U - U_old) * va_sqrt, 'fro')
        eps_dual_xr = thresh4 + rel_tol * 0.5 * (
            np.linalg.norm(va_sqrt[:, np.newaxis] * H * va_sqrt, 'fro') +
            np.linalg.norm(va_sqrt[:, np.newaxis] * K * va_sqrt, 'fro'))
        
        # Store history
        history['r_norm'].append(r_norm)
        history['s_norm'].append(s_norm)
        history['eps_pri'].append(eps_pri)
        history['eps_dual'].append(eps_dual)
        history['r_norm2'].append(r_norm2)
        history['s_norm2'].append(s_norm2)
        history['eps_pri2'].append(eps_pri2)
        history['eps_dual2'].append(eps_dual2)
        history['r_xr1'].append(r_xr1)
        history['r_xr2'].append(r_xr2)
        history['s_xr'].append(s_xr)
        history['eps_pri_xr1'].append(eps_pri_xr1)
        history['eps_pri_xr2'].append(eps_pri_xr2)
        history['eps_dual_xr'].append(eps_dual_xr)
        
        # Print progress
        if not quiet and iteration % 10 == 0:
            print(f"{iteration:5d} | {r_norm:8.2e} {eps_pri:8.2e} | "
                  f"{s_norm:8.2e} {eps_dual:8.2e} | "
                  f"{r_norm2:8.2e} {eps_pri2:8.2e} | "
                  f"{s_norm2:8.2e} {eps_dual2:8.2e} | "
                  f"{r_xr1:8.2e} {r_xr2:8.2e} | {s_xr:8.2e}")
        
        # Check convergence
        if (r_norm < eps_pri and s_norm < eps_dual and
            r_norm2 < eps_pri2 and s_norm2 < eps_dual2 and
            r_xr1 < eps_pri_xr1 and r_xr2 < eps_pri_xr2 and
            s_xr < eps_dual_xr):
            if not quiet:
                print(f"Converged at iteration {iteration}")
            break
        
        # Adaptive penalty parameters
        if var_rho:
            r1_ratio = r_norm / eps_pri
            s1_ratio = s_norm / eps_dual
            r2_ratio = r_norm2 / eps_pri2
            s2_ratio = s_norm2 / eps_dual2
            xr1_ratio = r_xr1 / eps_pri_xr1
            xr2_ratio = r_xr2 / eps_pri_xr2
            sxr_ratio = s_xr / eps_dual_xr
            
            if r1_ratio > mu * s1_ratio and r2_ratio > mu * s2_ratio:
                rho1 = tau_inc * rho1
                rho_changed = True
            elif s1_ratio > mu * r1_ratio and s2_ratio > mu * r2_ratio:
                rho1 = rho1 / tau_dec
                rho_changed = True
            
            if xr1_ratio > mu * sxr_ratio and xr2_ratio > mu * sxr_ratio:
                rho2 = tau_inc * rho2
                rho_changed = True
            elif sxr_ratio > mu * xr1_ratio and sxr_ratio > mu * xr2_ratio:
                rho2 = rho2 / tau_dec
                rho_changed = True
    
    if return_history:
        return U, history
    else:
        return U
