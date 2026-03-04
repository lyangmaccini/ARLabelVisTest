"""
Demo script for regularized geodesic distances
Python implementation of the MATLAB demo.m
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mesh_class import MeshClass
from rgd_admm import rgd_admm
from utils import (smooth_vf, interpolate_faces_to_vertices, 
                   interpolate_vertices_to_faces, compute_geodesic_gaussian)


def load_mesh_from_off(filename):
    """Load mesh from OFF file."""
    vertices, faces = MeshClass.read_off(filename)
    return MeshClass(vertices, faces)


def visualize_distances(mesh, u, x0, title='Distance Function', show_isolines=False):
    """
    Visualize distance function on mesh.
    
    Args:
        mesh: MeshClass object
        u: Distance values at vertices
        x0: Source vertex indices
        title: Plot title
        show_isolines: Whether to show isolines
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh with distance colors
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Create triangulation
    from matplotlib.tri import Triangulation
    triang = Triangulation(vertices[:, 0], vertices[:, 1], faces)
    
    # Plot surface
    surf = ax.plot_trisurf(triang, vertices[:, 2], cmap='jet', 
                          antialiased=True, linewidth=0.0, alpha=0.9)
    surf.set_array(u)
    
    # Mark source points
    source_points = vertices[x0]
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2],
              c='red', s=100, marker='o', edgecolors='black', linewidths=2)
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    return fig


def demo_dirichlet_regularization(mesh, x0):
    """
    Demo of Dirichlet energy regularization.
    
    Args:
        mesh: MeshClass object
        x0: Source vertex index or indices
    """
    print("=" * 60)
    print("Demo: Dirichlet Energy Regularization")
    print("=" * 60)
    
    # Different regularization weights
    alpha_hat_values = [0.0, 0.05, 0.15]
    titles = ['No Regularization', 
              'Dirichlet Regularization (α=0.05)',
              'Dirichlet Regularization (α=0.15)']
    
    results = []
    for alpha_hat, title in zip(alpha_hat_values, titles):
        print(f"\nComputing {title}...")
        u, history = rgd_admm(mesh, x0, reg='D', alpha_hat=alpha_hat, quiet=False)
        results.append((u, title))
        print(f"  Final r_norm: {history['r_norm'][-1]:.4e}")
        print(f"  Final s_norm: {history['s_norm'][-1]:.4e}")
        print(f"  Iterations: {len(history['r_norm'])}")
    
    # Visualize all results
    for u, title in results:
        fig = visualize_distances(mesh, u, x0, title=title)
        plt.show()
    
    return results


def demo_vector_field_alignment(mesh, x0):
    """
    Demo of vector field alignment regularization.
    
    Args:
        mesh: MeshClass object
        x0: Source vertex index or indices
    """
    print("\n" + "=" * 60)
    print("Demo: Vector Field Alignment")
    print("=" * 60)
    
    # Create a simple vector field
    # For demo, create a radial field from mesh center
    print("\nCreating vector field...")
    barycenters = mesh.barycenter()
    center = np.mean(barycenters, axis=0)
    
    # Radial vectors
    vf = barycenters - center
    vf = mesh.normalize_vf(vf)
    
    # Smooth the vector field
    print("Smoothing vector field...")
    vf_smooth = smooth_vf(mesh, vf, num_iterations=5)
    
    # Optionally localize with geodesic Gaussian
    # (simplified version - just use uniform weights for demo)
    
    # Compute regularized distances
    alpha_hat = 0.05
    beta_hat = 50.0
    
    print(f"\nComputing with vector field alignment...")
    print(f"  alpha_hat = {alpha_hat}")
    print(f"  beta_hat = {beta_hat}")
    
    u_vfa, history = rgd_admm(mesh, x0, reg='vfa', 
                              alpha_hat=alpha_hat, 
                              beta_hat=beta_hat, 
                              vf=vf_smooth, 
                              quiet=False)
    
    print(f"  Final r_norm: {history['r_norm'][-1]:.4e}")
    print(f"  Final s_norm: {history['s_norm'][-1]:.4e}")
    print(f"  Iterations: {len(history['r_norm'])}")
    
    # Visualize
    fig = visualize_distances(mesh, u_vfa, x0, 
                             title='Vector Field Alignment Regularization')
    
    # Add vector field visualization
    ax = fig.axes[0]
    barycenters_subsample = barycenters[::max(1, mesh.nf // 100)]
    vf_subsample = vf_smooth[::max(1, mesh.nf // 100)]
    
    scale = np.mean(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0))) * 0.05
    ax.quiver(barycenters_subsample[:, 0], 
             barycenters_subsample[:, 1],
             barycenters_subsample[:, 2],
             vf_subsample[:, 0] * scale,
             vf_subsample[:, 1] * scale,
             vf_subsample[:, 2] * scale,
             color='black', linewidth=1.5, arrow_length_ratio=0.3)
    
    plt.show()
    
    return u_vfa


def create_simple_mesh():
    """
    Create a simple mesh for testing (sphere-like).
    """
    # Create icosphere
    print("Creating test mesh (icosphere)...")
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    vertices = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=np.float64)
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])
    
    # Icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    
    return MeshClass(vertices, faces)


def main():
    """Main demo function."""
    print("Regularized Geodesic Distances - Python Implementation")
    print("=" * 60)
    
    # Option 1: Load mesh from OFF file (if available)
    # Uncomment and modify path as needed:
    # mesh = load_mesh_from_off('path/to/mesh.off')
    
    # Option 2: Use simple test mesh
    mesh = create_simple_mesh()
    
    print(f"Mesh loaded:")
    print(f"  Vertices: {mesh.nv}")
    print(f"  Faces: {mesh.nf}")
    print(f"  Total area: {np.sum(mesh.va):.4f}")
    
    # Choose source vertex (top of sphere)
    x0 = np.argmax(mesh.vertices[:, 1])
    print(f"\nSource vertex: {x0}")
    print(f"Source position: {mesh.vertices[x0]}")
    
    # Demo 1: Dirichlet regularization
    results_dirichlet = demo_dirichlet_regularization(mesh, x0)
    
    # Demo 2: Vector field alignment (optional)
    # Uncomment to run:
    # result_vfa = demo_vector_field_alignment(mesh, x0)
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
