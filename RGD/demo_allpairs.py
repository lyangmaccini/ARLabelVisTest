"""
All-Pairs Distance Demo

Demonstrates computing the full distance matrix between all vertex pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
from mesh import Mesh
from rgd_allpairs import rgd_allpairs
from visualization import plot_distance_field


def main():
    print("=" * 70)
    print("All-Pairs Regularized Geodesic Distances Demo")
    print("=" * 70)
    
    # Create or load a small mesh (all-pairs is expensive!)
    mesh = create_small_mesh()
    # Or load: mesh = Mesh.from_file('cat_rr.off')
    
    print(f"\nMesh: {mesh}")
    print(f"Note: All-pairs computation is O(N²), so keep mesh small (<5K vertices)")
    
    if mesh.nv > 5000:
        print(f"\nWarning: Mesh has {mesh.nv} vertices, this may take a while!")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Compute all-pairs distances
    print("\n" + "-" * 70)
    print("Computing all-pairs distance matrix...")
    print("This may take several minutes for large meshes...")
    print("-" * 70)
    
    U, history = rgd_allpairs(mesh, 
                             alpha_hat=0.03,
                             quiet=False,
                             return_history=True)
    
    print("\nDone!")
    print(f"Distance matrix shape: {U.shape}")
    print(f"Matrix is symmetric: {np.allclose(U, U.T)}")
    print(f"Diagonal is zero: {np.allclose(np.diag(U), 0)}")
    
    # Visualize some distance fields
    print("\n" + "-" * 70)
    print("Visualizing sample distance fields...")
    print("-" * 70)
    
    # Select a few source vertices
    n_samples = 4
    sample_indices = np.linspace(0, mesh.nv - 1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), 
                            subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for i, source_idx in enumerate(sample_indices):
        distances = U[source_idx, :]
        
        plot_distance_field(mesh, distances, 
                          source_vertex=source_idx,
                          ax=axes[i],
                          title=f'From Vertex {source_idx}')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/allpairs_samples.png', 
               dpi=150, bbox_inches='tight')
    print("\nSaved sample plots to: /mnt/user-data/outputs/allpairs_samples.png")
    
    # Analyze distance matrix
    print("\n" + "-" * 70)
    print("Distance Matrix Statistics:")
    print("-" * 70)
    
    # Remove diagonal for statistics
    U_off_diag = U[~np.eye(mesh.nv, dtype=bool)]
    
    print(f"\nMin distance (non-zero): {U_off_diag[U_off_diag > 0].min():.4f}")
    print(f"Max distance: {U_off_diag.max():.4f}")
    print(f"Mean distance: {U_off_diag.mean():.4f}")
    print(f"Median distance: {np.median(U_off_diag):.4f}")
    
    # Find diameter (max distance between any two vertices)
    i_max, j_max = np.unravel_index(U.argmax(), U.shape)
    diameter = U[i_max, j_max]
    print(f"\nDiameter (max pairwise distance): {diameter:.4f}")
    print(f"  Between vertices {i_max} and {j_max}")
    
    # Plot distance distribution
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(U_off_diag, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Pairwise Distances')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/allpairs_histogram.png', 
               dpi=150, bbox_inches='tight')
    print("\nSaved histogram to: /mnt/user-data/outputs/allpairs_histogram.png")
    
    # Plot convergence
    fig3, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    iters = np.arange(len(history['r_norm']))
    ax.semilogy(iters, history['r_norm'], label='Primal residual 1', linewidth=2)
    ax.semilogy(iters, history['r_norm2'], label='Primal residual 2', linewidth=2)
    ax.semilogy(iters, history['r_xr1'], label='Consensus residual', linewidth=2)
    ax.semilogy(iters, history['eps_pri'], 'k--', label='Tolerance', linewidth=1)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('ADMM Convergence (All-Pairs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/allpairs_convergence.png', 
               dpi=150, bbox_inches='tight')
    print("Saved convergence plot to: /mnt/user-data/outputs/allpairs_convergence.png")
    
    # Save distance matrix
    np.save('/mnt/user-data/outputs/distance_matrix.npy', U)
    print("\nSaved distance matrix to: /mnt/user-data/outputs/distance_matrix.npy")
    
    plt.show()


def create_small_mesh(n=10):
    """Create a small test mesh (sphere)."""
    from scipy.spatial import Delaunay
    
    # Fibonacci sphere
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    vertices = np.column_stack([x, y, z])
    
    # Triangulate using convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    faces = hull.simplices
    
    mesh = Mesh(vertices, faces, "small_sphere")
    
    return mesh


if __name__ == '__main__':
    main()
