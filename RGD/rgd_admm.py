"""
Regularized Geodesic Distance Computation using ADMM

This module implements the ADMM algorithm for computing regularized geodesic
distances on triangle meshes.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve, factorized
from typing import Optional, Union, Tuple
import warnings


def rgd_admm(mesh,
             source_indices: Union[int, np.ndarray],
             reg: str = 'D',
             alpha_hat: float = 0.1,
             beta_hat: float = 0.0,
             vector_field: Optional[np.ndarray] = None,
             max_iter: int = 10000,
             abs_tol: float = 1e-5 / 2,
             rel_tol: float = 1e-2,
             quiet: bool = False,
             return_history: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Compute regularized geodesic distance using ADMM.
    
    Solves the optimization problem:
        maximize ∫ u dA - α·R(u)
        subject to ||∇u|| ≤ 1, u(x₀) = 0
    
    where R(u) is a regularization term.
    
    Args:
        mesh: Mesh object containing geometry and operators
        source_indices: Source vertex index or array of indices where u = 0
        reg: Regularization type
            'D' - Dirichlet energy (default): R(u) = ∫||∇u||² dA
            'H' - Hessian energy: R(u) = ∫||Hess(u)||² dA
            'vfa' - Vector field alignment: R(u) = ∫||∇u||² + β·<∇u,v>² dA
        alpha_hat: Scale-invariant regularization weight
            Actual α = alpha_hat * sqrt(total_area)
        beta_hat: Vector field alignment weight (for reg='vfa')
            Actual β = beta_hat * sqrt(total_area)
        vector_field: Target vector field for alignment (nf x 3), required for reg='vfa'
        max_iter: Maximum ADMM iterations
        abs_tol: Absolute convergence tolerance
        rel_tol: Relative convergence tolerance
        quiet: If True, suppress iteration output
        return_history: If True, return convergence history
        
    Returns:
        u: Distance function on all vertices (nv,)
        history: (optional) Dictionary with convergence history
    """
    # Convert source to array
    if np.isscalar(source_indices):
        source_indices = np.array([source_indices], dtype=int)
    else:
        source_indices = np.asarray(source_indices, dtype=int)
    
    # Extract mesh data
    nv = mesh.nv
    nf = mesh.nf
    va = mesh.va
    ta = mesh.ta
    G = mesh.G
    Ww = mesh.Ww
    
    total_area = ta.sum()
    
    # Set regularization parameters
    alpha = alpha_hat * np.sqrt(total_area)
    var_rho = True  # Use varying penalty parameter
    
    if reg == 'D':
        # Dirichlet regularization
        Ww_reg = Ww
        var_rho = True
        
    elif reg == 'H':
        # Hessian regularization - requires external curved_hessian
        try:
            # This would require the curved_hessian library
            # For now, raise an error with instructions
            raise NotImplementedError(
                "Hessian regularization requires the curved_hessian library. "
                "Please install from: "
                "https://github.com/odedstein/ASmoothnessEnergyWithoutBoundaryDistortionForCurvedSurfaces"
            )
        except ImportError:
            raise ImportError("curved_hessian library not found")
        
        alpha = alpha_hat * np.sqrt(total_area**3)
        var_rho = False
        abs_tol = abs_tol / 20
        rel_tol = rel_tol / 20
        
    elif reg == 'vfa':
        # Vector field alignment
        if vector_field is None:
            raise ValueError("vector_field required for reg='vfa'")
        
        beta = beta_hat * np.sqrt(total_area)
        
        vf_norm = np.linalg.norm(vector_field, axis=1).max()
        if vf_norm < 1e-10:
            raise ValueError("Vector field is empty or too small")
        
        # Build alignment matrix V: encourages ∇u to align with vector_field
        # V projects gradients onto vector field directions
        vf = vector_field.reshape(nf, 3)
        
        # Create block matrix for v⊗v (outer product)
        I_blocks = []
        J_blocks = []
        V_blocks = []
        
        for i in range(3):
            for j in range(3):
                I_blocks.extend(np.arange(nf) + i * nf)
                J_blocks.extend(np.arange(nf) + j * nf)
                V_blocks.extend(vf[:, i] * vf[:, j])
        
        V_mat = sparse.csr_matrix((V_blocks, (I_blocks, J_blocks)), 
                                  shape=(3*nf, 3*nf))
        
        # Weighted regularization: W_reg = G^T·diag(ta)·(I + β·V)·G
        ta_diag = diags(np.repeat(ta, 3))
        Ww_reg = G.T @ ta_diag @ (eye(3*nf) + beta * V_mat) @ G
        Ww_reg = Ww_reg.tocsr()
        
        var_rho = False
        
    else:
        raise ValueError(f"Unknown regularization: {reg}")
    
    # ADMM parameters
    rho = 2.0 * np.sqrt(total_area)
    mu = 10.0      # Penalty update factor
    tau_inc = 2.0  # Penalty increase factor
    tau_dec = 2.0  # Penalty decrease factor
    alpha_k = 1.7  # Over-relaxation parameter
    
    # Convergence thresholds
    thresh1 = np.sqrt(3 * nf) * abs_tol * np.sqrt(total_area)
    thresh2 = np.sqrt(nv) * abs_tol * total_area
    
    # Eliminate source vertices (boundary condition u = 0 at sources)
    mask = np.ones(nv, dtype=bool)
    mask[source_indices] = False
    nv_p = mask.sum()
    
    indices_p = np.where(mask)[0]
    
    # Reduced operators (without source vertices)
    va_p = va[mask]
    Ww_p = Ww[np.ix_(mask, mask)]
    G_p = G[:, mask]
    G_pt = G_p.T
    
    # Divergence operator: div = G^T·diag(ta)
    ta_repeated = np.repeat(ta, 3)
    div_p = (G_pt @ sparse.diags(ta_repeated)).tocsr()

    
    if reg in ['vfa', 'H']:
        Ww_reg_p = Ww_reg[np.ix_(mask, mask)]
    
    # Initialize variables
    u_p = np.zeros(nv_p)
    y = np.zeros(3 * nf)
    z = np.zeros(3 * nf)
    div_y = np.zeros(nv_p)
    div_z = np.zeros(nv_p)
    
    # History tracking
    history = {
        'r_norm': [],
        's_norm': [],
        'eps_pri': [],
        'eps_dual': [],
        'rho': []
    }
    
    if not quiet:
        print(f"{'Iter':>5s} {'r_norm':>10s} {'eps_pri':>10s} {'s_norm':>10s} {'eps_dual':>10s}")
    
    # Pre-factorize for efficiency (if rho is fixed)
    if reg == 'D':
        A_fact = factorized((alpha + rho) * Ww_p)
    elif not var_rho:
        A_matrix = alpha * Ww_reg_p + rho * Ww_p
        A_fact = factorized(A_matrix.tocsc())
    
    # ADMM iterations
    for iteration in range(max_iter):
        # Step 1: u-minimization
        # Solve (α·W_reg + ρ·W)·u = va - div(y) + ρ·div(z)
        b = va_p - div_y + rho * div_z
        
        if reg == 'D':
            u_p = A_fact(b) / (alpha + rho)
        elif not var_rho:
            u_p = A_fact(b)
        else:  # Variable rho
            A_matrix = alpha * Ww_reg_p + rho * Ww_p
            u_p = spsolve(A_matrix, b)
        
        Gx = G_p @ u_p
        
        # Step 2: z-minimization (project to unit norm constraint)
        z_old = z.copy()
        div_z_old = div_z.copy()
        
        z_temp = (1.0 / rho) * y + Gx
        z_temp = z_temp.reshape(nf, 3)
        
        # Project: z = z_temp / max(1, ||z_temp||)
        z_norms = np.linalg.norm(z_temp, axis=1)
        z_norms[z_norms < 1.0] = 1.0
        z = (z_temp / z_norms[:, np.newaxis]).flatten()
        
        div_z = div_p @ z
        
        # Step 3: dual variable update (with over-relaxation)
        y = y + rho * (alpha_k * Gx + (1 - alpha_k) * z_old - z)
        div_y = div_p @ y
        
        # Compute residuals
        ta_sqrt = np.sqrt(np.repeat(ta, 3))
        Gx_weighted = ta_sqrt * Gx
        z_weighted = ta_sqrt * z
        
        r_norm = np.linalg.norm(Gx_weighted - z_weighted)
        s_norm = rho * np.linalg.norm(div_z - div_z_old)
        
        eps_pri = thresh1 + rel_tol * max(np.linalg.norm(Gx_weighted), 
                                          np.linalg.norm(z_weighted))
        eps_dual = thresh2 + rel_tol * np.linalg.norm(div_y)
        
        # Store history
        history['r_norm'].append(r_norm)
        history['s_norm'].append(s_norm)
        history['eps_pri'].append(eps_pri)
        history['eps_dual'].append(eps_dual)
        history['rho'].append(rho)
        
        if not quiet and iteration % 100 == 0:
            print(f"{iteration:5d} {r_norm:10.4e} {eps_pri:10.4e} {s_norm:10.4e} {eps_dual:10.4e}")
        
        # Check convergence
        if iteration > 0 and r_norm < eps_pri and s_norm < eps_dual:
            if not quiet:
                print(f"Converged at iteration {iteration}")
            break
        
        # Adaptive penalty parameter
        if var_rho:
            if r_norm > mu * s_norm:
                rho = tau_inc * rho
                if reg == 'D':
                    A_fact = factorized((alpha + rho) * Ww_p)
            elif s_norm > mu * r_norm:
                rho = rho / tau_dec
                if reg == 'D':
                    A_fact = factorized((alpha + rho) * Ww_p)
    
    # Reconstruct full solution
    u = np.zeros(nv)
    u[indices_p] = u_p
    # Sources remain at 0 (already initialized)
    
    if return_history:
        return u, history
    else:
        return u


def compute_gradient_norm(mesh, u: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude on each face.
    
    Args:
        mesh: Mesh object
        u: Scalar function on vertices (nv,)
        
    Returns:
        grad_norms: Gradient magnitudes on faces (nf,)
    """
    grad = mesh.G @ u
    grad = grad.reshape(mesh.nf, 3)
    grad_norms = np.linalg.norm(grad, axis=1)
    return grad_norms


def compute_energy(mesh, 
                  u: np.ndarray,
                  reg: str = 'D',
                  alpha: float = 0.0) -> dict:
    """
    Compute energy components for a distance function.
    
    Args:
        mesh: Mesh object
        u: Distance function (nv,)
        reg: Regularization type
        alpha: Regularization weight
        
    Returns:
        energies: Dictionary with energy components
    """
    # Data term: ∫ u dA
    data_term = np.dot(mesh.va, u)
    
    # Regularization term
    if reg == 'D':
        # Dirichlet: ∫||∇u||² dA = u^T·L·u
        reg_term = u @ mesh.Ww @ u
    else:
        reg_term = 0.0
    
    # Gradient constraint violation
    grad_norms = compute_gradient_norm(mesh, u)
    max_grad = grad_norms.max()
    mean_grad = (grad_norms * mesh.ta).sum() / mesh.ta.sum()
    
    energies = {
        'data': data_term,
        'regularization': reg_term,
        'total': data_term - alpha * reg_term,
        'max_gradient': max_grad,
        'mean_gradient': mean_grad
    }
    
    return energies


def plot_convergence(history: dict, ax=None):
    """
    Plot ADMM convergence history.
    
    Args:
        history: Dictionary from rgd_admm with return_history=True
        ax: Matplotlib axis (optional)
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    iters = np.arange(len(history['r_norm']))
    
    ax.semilogy(iters, history['r_norm'], label='Primal residual', linewidth=2)
    ax.semilogy(iters, history['eps_pri'], 'k--', label='Primal tolerance', linewidth=1)
    ax.semilogy(iters, history['s_norm'], label='Dual residual', linewidth=2)
    ax.semilogy(iters, history['eps_dual'], 'k:', label='Dual tolerance', linewidth=1)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('ADMM Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return ax
