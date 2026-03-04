"""
ADMM algorithm for computing regularized geodesic distances
Python implementation of the MATLAB rdg_ADMM function
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
import warnings


def rgd_admm(mesh, x0, reg='D', alpha_hat=0.1, beta_hat=0.0, vf=None, 
             max_iter=10000, quiet=True, abstol=1e-5/2, reltol=1e-2):
    """
    Compute regularized geodesic distances using ADMM.
    
    Args:
        mesh: MeshClass object
        x0: Source vertex indices (0-indexed array or single index)
        reg: Regularizer type - 'D' (Dirichlet), 'H' (Hessian), or 'vfa' (vector field alignment)
        alpha_hat: Scale-invariant regularizer weight (default 0.1)
        beta_hat: Vector field alignment weight (default 0, only for 'vfa')
        vf: Vector field for alignment (nf x 3 array, only for 'vfa')
        max_iter: Maximum number of ADMM iterations (default 10000)
        quiet: If True, suppress iteration output (default True)
        abstol: Absolute tolerance for convergence (default 1e-5/2)
        reltol: Relative tolerance for convergence (default 1e-2)
    
    Returns:
        u: Regularized distance function on vertices
        history: Dictionary with convergence history
    """
    
    # Convert x0 to array if single index
    if np.isscalar(x0):
        x0 = np.array([x0])
    else:
        x0 = np.array(x0, dtype=np.int32)
    
    # Get mesh data
    vertices = mesh.vertices
    faces = mesh.faces
    nv = mesh.nv
    nf = mesh.nf
    va = mesh.va
    ta = mesh.ta
    G = mesh.G
    Ww = mesh.Ww
    
    tasq = np.sqrt(ta).reshape(-1, 1)  # For weighting
    tasq = np.tile(tasq, (3, 1)).flatten()
    
    # Set parameters according to regularizer
    total_area = np.sum(va)
    
    if reg == 'D':
        alpha = alpha_hat * np.sqrt(total_area)
        var_rho = True  # Use varying penalty parameter
        Ww_s = None
        
    elif reg == 'H':
        alpha = alpha_hat * np.sqrt(total_area**3)
        # Note: Hessian computation would require curved_hessian function
        # For now, we'll use Dirichlet as approximation
        warnings.warn("Hessian regularizer not fully implemented, using Dirichlet approximation")
        Ww_s = Ww
        var_rho = False
        
    elif reg == 'vfa':
        alpha = alpha_hat * np.sqrt(total_area)
        beta = beta_hat * np.sqrt(total_area)
        
        if vf is None:
            raise ValueError("Vector field vf is required for 'vfa' regularizer")
        
        vf = np.array(vf)
        if vf.shape != (nf, 3):
            raise ValueError(f"Vector field must be shape (nf, 3), got {vf.shape}")
        
        if np.max(mesh.normv(vf)) < 1e-10:
            raise ValueError("Vector field for alignment is empty")
        
        # Build vector field alignment matrix
        Vmat_blocks = []
        for i in range(3):
            row_blocks = []
            for j in range(3):
                diag_vals = vf[:, i] * vf[:, j]
                row_blocks.append(sparse.diags(diag_vals, 0))
            Vmat_blocks.append(row_blocks)
        
        # Stack blocks to form 3nf x 3nf matrix
        Vmat = sparse.bmat(Vmat_blocks, format='csr')
        
        # Construct Ww_s with vector field alignment
        ta_diag = sparse.diags(np.tile(ta, 3), 0)
        I_3nf = sparse.identity(3*nf)
        
        Ww_s = G.T @ ta_diag @ (I_3nf + beta * Vmat) @ G
        var_rho = False
        
    else:
        raise ValueError(f"Unrecognized regularizer: {reg}")
    
    # ADMM parameters
    rho = 2 * np.sqrt(total_area)
    mu = 10.0
    tau_inc = 2.0
    tau_dec = 2.0
    alpha_k = 1.7  # Over-relaxation
    
    if reg == 'H':
        abstol = abstol / 20
        reltol = reltol / 20
    
    thresh1 = np.sqrt(3 * nf) * abstol * np.sqrt(total_area)
    thresh2 = np.sqrt(nv) * abstol * total_area
    
    # Initialize variables
    nv_p = nv - len(x0)  # Number of free vertices
    u_p = np.zeros(nv_p)
    y = np.zeros(3 * nf)
    z = np.zeros(3 * nf)
    div_y = np.zeros(nv_p)
    div_z = np.zeros(nv_p)
    
    # History
    history = {
        'r_norm': np.zeros(max_iter),
        's_norm': np.zeros(max_iter),
        'eps_pri': np.zeros(max_iter),
        'eps_dual': np.zeros(max_iter),
        'rho': np.zeros(max_iter)
    }
    
    # Eliminate boundary conditions (x0 vertices)
    nv_p_idx = np.ones(nv, dtype=bool)
    nv_p_idx[x0] = False
    nv_p_list = np.where(nv_p_idx)[0]
    
    va_p = va[nv_p_idx]
    Ww_p = Ww[nv_p_idx, :][:, nv_p_idx]
    G_p = G[:, nv_p_idx]
    G_pt = G_p.T
    
    # Divergence operator: div = G^T .* (ta repeated 3 times)
    # In MATLAB: div_p = G_pt.*repmat(ta,3,1)';
    # This multiplies each row of G_pt by the corresponding ta value
    ta_rep_col = np.repeat(ta, 3).reshape(-1, 1)  # Make it a column for broadcasting
    div_p = G_pt.multiply(ta_rep_col.T)  # Multiply rows by ta values
    
    if Ww_s is not None:
        Ww_s_p = Ww_s[nv_p_idx, :][:, nv_p_idx]
    
    if not quiet:
        print(f"{'Iter':>4} {'r_norm':>12} {'eps_pri':>12} {'s_norm':>12} {'eps_dual':>12}")
    
    # Pre-factorization for linear solve
    if reg == 'D':
        # For Dirichlet: (alpha + rho) * Ww_p
        factor_matrix = (alpha + rho) * Ww_p
        solve = factorized(factor_matrix.tocsc())
        scale_factor = 1.0 / (alpha + rho)
    else:  # 'H' or 'vfa'
        if not var_rho:
            # For fixed rho: alpha * Ww_s_p + rho * Ww_p
            factor_matrix = alpha * Ww_s_p + rho * Ww_p
            solve = factorized(factor_matrix.tocsc())
            scale_factor = 1.0
    
    # Initialize boundary condition contributions
    # The va_p term enforces that distances increase from sources
    # It comes from the data fidelity term in the optimization
    
    # ADMM iterations
    converged = False
    for ii in range(max_iter):
        # Step 1: u-minimization
        # The right-hand side includes:
        # - va_p: data term (enforces distance growth)
        # - div_y: dual variable contribution
        # - rho * div_z: ADMM penalty term
        b = va_p - div_y + rho * div_z
        
        if reg == 'D':
            u_p = solve(b) * scale_factor
        else:  # 'H' or 'vfa'
            if not var_rho:
                u_p = solve(b)
            else:
                # Re-solve with updated rho (slower)
                factor_matrix = alpha * Ww_s_p + rho * Ww_p
                u_p = sparse.linalg.spsolve(factor_matrix, b)
        
        Gx = G_p @ u_p
        
        # Step 2: z-minimization (projection onto unit ball)
        zold = z.copy()
        div_zold = div_z.copy()
        
        z = (1.0 / rho) * y + Gx
        z_reshaped = z.reshape(3, nf).T  # nf x 3
        
        norm_z = np.linalg.norm(z_reshaped, axis=1)
        norm_z[norm_z < 1.0] = 1.0
        
        z_reshaped = z_reshaped / norm_z[:, np.newaxis]
        z = z_reshaped.T.flatten()
        
        div_z = div_p @ z
        
        # Step 3: dual variable update
        y = y + rho * (alpha_k * Gx + (1 - alpha_k) * zold - z)
        div_y = div_p @ y
        
        # Compute residuals
        tasq_Gx = tasq * Gx
        tasq_z = tasq * z
        
        history['r_norm'][ii] = np.linalg.norm(tasq_Gx - tasq_z)
        history['s_norm'][ii] = rho * np.linalg.norm(div_z - div_zold)
        history['eps_pri'][ii] = thresh1 + reltol * max(
            np.linalg.norm(tasq_Gx), np.linalg.norm(tasq_z)
        )
        history['eps_dual'][ii] = thresh2 + reltol * np.linalg.norm(div_y)
        history['rho'][ii] = rho
        
        if not quiet:
            print(f"{ii+1:4d} {history['r_norm'][ii]:12.4e} "
                  f"{history['eps_pri'][ii]:12.4e} {history['s_norm'][ii]:12.4e} "
                  f"{history['eps_dual'][ii]:12.4e}")
        
        # Check convergence
        if ii > 0 and (history['r_norm'][ii] < history['eps_pri'][ii] and 
                       history['s_norm'][ii] < history['eps_dual'][ii]):
            converged = True
            if not quiet:
                print(f"Converged at iteration {ii+1}")
            break
        
        # Update penalty parameter
        if var_rho:
            if history['r_norm'][ii] > mu * history['s_norm'][ii]:
                rho = tau_inc * rho
                if reg == 'D':
                    factor_matrix = (alpha + rho) * Ww_p
                    solve = factorized(factor_matrix.tocsc())
                    scale_factor = 1.0 / (alpha + rho)
            elif history['s_norm'][ii] > mu * history['r_norm'][ii]:
                rho = rho / tau_dec
                if reg == 'D':
                    factor_matrix = (alpha + rho) * Ww_p
                    solve = factorized(factor_matrix.tocsc())
                    scale_factor = 1.0 / (alpha + rho)
    
    if not converged and not quiet:
        print(f"Warning: Did not converge in {max_iter} iterations")
    
    # Reconstruct full solution
    u = np.zeros(nv)
    u[nv_p_idx] = u_p
    
    # Trim history to actual iterations
    actual_iters = ii + 1
    for key in history:
        history[key] = history[key][:actual_iters]
    
    return u, history
