"""
Simple example of using the RGD Python implementation
"""

import numpy as np
from mesh_class import MeshClass
from rgd_admm import rgd_admm


def main():
    """Simple usage example."""
    print("RGD Python Implementation - Simple Example")
    print("=" * 50)
    
    # Create a simple tetrahedral mesh
    print("\n1. Creating mesh...")
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    
    mesh = MeshClass(vertices, faces)
    print(f"   Mesh has {mesh.nv} vertices and {mesh.nf} faces")
    
    # Compute geodesic distance from vertex 0
    print("\n2. Computing geodesic distances...")
    source = 0
    
    # No regularization
    print("   a) No regularization (alpha_hat=0)")
    u_geo, _ = rgd_admm(mesh, source, reg='D', alpha_hat=0.0, quiet=True)
    print(f"      Distances: {u_geo}")
    
    # With regularization
    print("   b) With Dirichlet regularization (alpha_hat=0.1)")
    u_smooth, _ = rgd_admm(mesh, source, reg='D', alpha_hat=0.1, quiet=True)
    print(f"      Distances: {u_smooth}")
    
    print("\n3. Comparison:")
    print(f"   Vertex | Geodesic | Regularized | Difference")
    print(f"   " + "-" * 50)
    for i in range(mesh.nv):
        print(f"   {i:6d} | {u_geo[i]:8.4f} | {u_smooth[i]:11.4f} | {abs(u_geo[i]-u_smooth[i]):10.4f}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nNext steps:")
    print("- Try different regularization weights (alpha_hat)")
    print("- Load your own mesh from OFF file")
    print("- Experiment with vector field alignment")
    print("- See demo.py for more examples")


if __name__ == '__main__':
    main()
