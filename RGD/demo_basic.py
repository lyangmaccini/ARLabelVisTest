"""
Basic Demo: Regularized Geodesic Distances with Dirichlet Regularization

This demo shows how to compute standard and regularized geodesic distances
and compare them visually.
"""

import numpy as np
import matplotlib.pyplot as plt
from mesh import Mesh
from rgd_admm import rgd_admm
from visualization import plot_distance_field, plot_comparison


def main():
    print("=" * 70)
    print("Regularized Geodesic Distances - Basic Demo")
    print("=" * 70)
    
    # Load mesh (you'll need to provide a .OFF file)
    # For this demo, we'll create a simple grid mesh
    mesh = create_simple_mesh()
    # Alternatively, load from file:
    # mesh = Mesh.from_file('spot_rr.off')
    
    print(f"\nMesh loaded: {mesh}")
    
    # Select source vertex (center of mesh)
    source_vertex = mesh.nv // 2
    print(f"Source vertex: {source_vertex}")
    
    # Compute distances with different regularization strengths
    print("\n" + "-" * 70)
    print("Computing geodesic distances...")
    print("-" * 70)
    
    # No regularization (standard geodesic)
    print("\n1. Standard geodesic (no regularization)...")
    u0 = rgd_admm(mesh, source_vertex, alpha_hat=0.0, quiet=True)
    
    # Light regularization
    print("2. Light regularization (alpha_hat = 0.05)...")
    u1 = rgd_admm(mesh, source_vertex, alpha_hat=0.05, quiet=True)
    
    # Medium regularization
    print("3. Medium regularization (alpha_hat = 0.10)...")
    u2 = rgd_admm(mesh, source_vertex, alpha_hat=0.10, quiet=True)
    
    # Heavy regularization
    print("4. Heavy regularization (alpha_hat = 0.15)...")
    u3 = rgd_admm(mesh, source_vertex, alpha_hat=0.15, quiet=True)
    
    print("\nDone!")
    
    # Visualize results
    print("\n" + "-" * 70)
    print("Visualizing results...")
    print("-" * 70)
    
    distance_fields = [u0, u1, u2, u3]
    labels = [
        'Standard Geodesic\n(α=0)',
        'Light Regularization\n(α=0.05)',
        'Medium Regularization\n(α=0.10)',
        'Heavy Regularization\n(α=0.15)'
    ]
    
    fig = plot_comparison(mesh, distance_fields, labels, 
                         source_vertex=source_vertex,
                         n_cols=2, figsize=(12, 10))
    
    plt.savefig('/mnt/user-data/outputs/rgd_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to: /mnt/user-data/outputs/rgd_comparison.png")
    
    # Show statistics
    print("\n" + "-" * 70)
    print("Statistics:")
    print("-" * 70)
    
    for i, (u, label) in enumerate(zip(distance_fields, labels)):
        max_dist = u.max()
        mean_dist = (u * mesh.va).sum() / mesh.va.sum()
        print(f"\n{label.split(chr(10))[0]}:")
        print(f"  Max distance: {max_dist:.4f}")
        print(f"  Mean distance: {mean_dist:.4f}")
    
    plt.show()


def create_simple_mesh():
    """Create a simple test mesh (grid on a sphere)."""
    from scipy.spatial import Delaunay
    
    # Create points on a sphere
    n_theta = 20
    n_phi = 20
    
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)
    
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # Create 2D parameterization for triangulation
    points_2d = np.column_stack([theta_grid.flatten(), phi_grid.flatten()])
    
    # Triangulate
    tri = Delaunay(points_2d)
    faces = tri.simplices
    
    # Remove degenerate triangles
    valid = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
        if area > 1e-6:
            valid.append(face)
    
    faces = np.array(valid)
    
    mesh = Mesh(vertices, faces, "sphere_grid")
    mesh.normalize_mesh()
    
    return mesh


if __name__ == '__main__':
    main()
