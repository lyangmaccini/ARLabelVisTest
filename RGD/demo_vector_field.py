"""
Vector Field Alignment Demo

Shows how to use vector field alignment regularization to guide geodesic
distances along preferred directions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mesh import Mesh
from rgd_admm import rgd_admm
from utils import smooth_vector_field, geodesic_gaussian
from visualization import plot_distance_field, plot_vector_field, plot_comparison


def main():
    print("=" * 70)
    print("Vector Field Alignment Demo")
    print("=" * 70)
    
    # Load or create mesh
    mesh = create_cylinder_mesh()
    # Or load from file:
    # mesh = Mesh.from_file('spot_rr.off')
    
    print(f"\nMesh: {mesh}")
    
    # Define source vertex
    source_vertex = mesh.nv // 4  # Near one end
    print(f"Source vertex: {source_vertex}")
    
    # Create a vector field with some constraints
    print("\n" + "-" * 70)
    print("Creating vector field...")
    print("-" * 70)
    
    # Choose some faces and directions
    nf = mesh.nf
    constraint_faces = [nf // 4, nf // 2]
    constraint_directions = [
        [1.0, 0.0, 0.5],   # Direction 1
        [0.0, 1.0, 0.2]    # Direction 2
    ]
    
    print(f"Constraint faces: {constraint_faces}")
    print(f"Directions: {constraint_directions}")
    
    # Smooth the vector field across the mesh
    vf_smooth = smooth_vector_field(mesh, 
                                   sparse_vf=np.zeros((nf, 3)),
                                   power=2,
                                   constraint_faces=np.array(constraint_faces),
                                   constraint_values=np.array(constraint_directions))
    
    # Optionally localize the vector field using a geodesic Gaussian
    localize = True
    if localize:
        print("\nLocalizing vector field with geodesic Gaussian...")
        
        # Get vertices of constraint faces
        vf_vertices = mesh.faces[constraint_faces].flatten()
        
        # Compute geodesic Gaussian
        gaussian = geodesic_gaussian(mesh, vf_vertices, 
                                    sigma_squared=mesh.ta.sum() / 100)
        
        # Apply to vector field (interpolate to faces)
        gaussian_faces = mesh.interpolate_vertex_to_face(gaussian)
        vf_smooth = vf_smooth * gaussian_faces[:, np.newaxis]
    
    # Compute distances
    print("\n" + "-" * 70)
    print("Computing distances...")
    print("-" * 70)
    
    # Standard geodesic
    print("\n1. Standard geodesic...")
    u_standard = rgd_admm(mesh, source_vertex, alpha_hat=0.0, quiet=True)
    
    # Dirichlet regularization
    print("2. Dirichlet regularization...")
    u_dirichlet = rgd_admm(mesh, source_vertex, 
                          reg='D', alpha_hat=0.05, quiet=True)
    
    # Vector field alignment
    print("3. Vector field alignment...")
    u_vfa = rgd_admm(mesh, source_vertex,
                    reg='vfa',
                    alpha_hat=0.05,
                    beta_hat=100.0,
                    vector_field=vf_smooth,
                    quiet=True)
    
    print("\nDone!")
    
    # Visualize
    print("\n" + "-" * 70)
    print("Visualizing results...")
    print("-" * 70)
    
    # Plot distance fields
    distance_fields = [u_standard, u_dirichlet, u_vfa]
    labels = [
        'Standard Geodesic',
        'Dirichlet Regularization',
        'Vector Field Alignment'
    ]
    
    fig1 = plot_comparison(mesh, distance_fields, labels,
                          source_vertex=source_vertex,
                          n_cols=3, figsize=(15, 5))
    
    plt.savefig('/mnt/user-data/outputs/vfa_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved distance comparison to: /mnt/user-data/outputs/vfa_comparison.png")
    
    # Plot vector field
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')
    plot_vector_field(mesh, vf_smooth, vertex_colors=u_vfa, 
                     scale=0.1, density=0.2, ax=ax,
                     title='Vector Field Alignment Result')
    
    plt.savefig('/mnt/user-data/outputs/vfa_vector_field.png', dpi=150, bbox_inches='tight')
    print("Saved vector field plot to: /mnt/user-data/outputs/vfa_vector_field.png")
    
    # Statistics
    print("\n" + "-" * 70)
    print("Statistics:")
    print("-" * 70)
    
    for u, label in zip(distance_fields, labels):
        max_dist = u.max()
        mean_dist = (u * mesh.va).sum() / mesh.va.sum()
        print(f"\n{label}:")
        print(f"  Max distance: {max_dist:.4f}")
        print(f"  Mean distance: {mean_dist:.4f}")
    
    plt.show()


def create_cylinder_mesh(radius=1.0, height=2.0, n_circ=20, n_height=20):
    """Create a cylinder mesh for testing."""
    # Create vertices
    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    z = np.linspace(0, height, n_height)
    
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    z = z_grid
    
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # Create faces
    faces = []
    for i in range(n_height - 1):
        for j in range(n_circ):
            j_next = (j + 1) % n_circ
            
            v0 = i * n_circ + j
            v1 = i * n_circ + j_next
            v2 = (i + 1) * n_circ + j
            v3 = (i + 1) * n_circ + j_next
            
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    faces = np.array(faces)
    
    mesh = Mesh(vertices, faces, "cylinder")
    mesh.center_mesh()
    
    return mesh


if __name__ == '__main__':
    main()
