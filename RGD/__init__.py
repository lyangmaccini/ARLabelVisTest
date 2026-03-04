"""
Regularized Geodesic Distances (Python)

A Python implementation of "A Convex Optimization Framework for Regularized 
Geodesic Distances" by Edelstein et al., SIGGRAPH 2023.

Main components:
- Mesh: Triangle mesh data structure with differential operators
- rgd_admm: Compute single-source regularized geodesic distances
- rgd_allpairs: Compute all-pairs distance matrix
- visualization: Plotting utilities
- utils: Helper functions
"""

from .mesh import Mesh
from .rgd_admm import rgd_admm, compute_gradient_norm, compute_energy, plot_convergence
from .rgd_allpairs import rgd_allpairs
from .utils import (read_off, write_off, compute_cotangent_laplacian,
                   smooth_vector_field, normalize_vectors, geodesic_gaussian)
from .visualization import (plot_mesh, plot_distance_field, plot_vector_field,
                            plot_comparison, plot_gradient_magnitude)

__version__ = '1.0.0'
__author__ = 'Adapted from MATLAB code by Michal Edelstein et al.'

__all__ = [
    # Core classes and functions
    'Mesh',
    'rgd_admm',
    'rgd_allpairs',
    
    # Utilities
    'read_off',
    'write_off',
    'compute_cotangent_laplacian',
    'smooth_vector_field',
    'normalize_vectors',
    'geodesic_gaussian',
    'compute_gradient_norm',
    'compute_energy',
    
    # Visualization
    'plot_mesh',
    'plot_distance_field',
    'plot_vector_field',
    'plot_comparison',
    'plot_gradient_magnitude',
    'plot_convergence',
]
