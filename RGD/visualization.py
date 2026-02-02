"""
Visualization utilities for meshes and geodesic distance fields
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple


def plot_mesh(mesh,
              vertex_colors: Optional[np.ndarray] = None,
              face_colors: Optional[np.ndarray] = None,
              cmap: str = 'viridis',
              show_edges: bool = False,
              ax: Optional[plt.Axes] = None,
              figsize: Tuple[int, int] = (10, 8),
              title: Optional[str] = None,
              colorbar: bool = True,
              vmin: Optional[float] = None,
              vmax: Optional[float] = None) -> plt.Axes:
    """
    Plot a triangle mesh with optional colors.
    
    Args:
        mesh: Mesh object
        vertex_colors: Colors per vertex (nv,) or (nv, 3)
        face_colors: Colors per face (nf,) or (nf, 3)
        cmap: Colormap name
        show_edges: Whether to show mesh edges
        ax: Matplotlib 3D axis (created if None)
        figsize: Figure size if creating new figure
        title: Plot title
        colorbar: Whether to show colorbar
        vmin, vmax: Color scale limits
        
    Returns:
        ax: Matplotlib 3D axis
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    
    # Create triangle collection
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Build face vertices
    verts = vertices[faces]
    
    # Determine colors
    if vertex_colors is not None:
        if vertex_colors.ndim == 1:
            # Scalar per vertex - interpolate to faces
            colors_scalar = (vertex_colors[faces[:, 0]] + 
                           vertex_colors[faces[:, 1]] + 
                           vertex_colors[faces[:, 2]]) / 3
        else:
            # RGB per vertex - average to faces
            colors_scalar = (vertex_colors[faces[:, 0]] + 
                           vertex_colors[faces[:, 1]] + 
                           vertex_colors[faces[:, 2]]) / 3
    elif face_colors is not None:
        colors_scalar = face_colors
    else:
        colors_scalar = None
    
    # Normalize colors
    if colors_scalar is not None and colors_scalar.ndim == 1:
        if vmin is None:
            vmin = colors_scalar.min()
        if vmax is None:
            vmax = colors_scalar.max()
        
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap(cmap)
        face_colors_rgba = colormap(norm(colors_scalar))
    elif colors_scalar is not None:
        # Already RGB/RGBA
        face_colors_rgba = colors_scalar
    else:
        face_colors_rgba = 'lightgray'
    
    # Create collection
    collection = Poly3DCollection(verts, 
                                 facecolors=face_colors_rgba,
                                 edgecolors='k' if show_edges else 'none',
                                 linewidths=0.1 if show_edges else 0,
                                 alpha=1.0)
    
    ax.add_collection3d(collection)
    
    # Set limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    
    # Colorbar
    if colorbar and colors_scalar is not None and colors_scalar.ndim == 1:
        mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array(colors_scalar)
        plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    
    return ax


def plot_distance_field(mesh,
                        distances: np.ndarray,
                        source_vertex: Optional[int] = None,
                        n_isolines: int = 15,
                        cmap: str = 'jet',
                        ax: Optional[plt.Axes] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        title: Optional[str] = None,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None) -> plt.Axes:
    """
    Visualize a distance field on a mesh with isolines.
    
    Args:
        mesh: Mesh object
        distances: Distance values on vertices (nv,)
        source_vertex: Source vertex index to mark (optional)
        n_isolines: Number of isolines to display
        cmap: Colormap
        ax: Matplotlib axis (created if None)
        figsize: Figure size
        title: Plot title
        vmin, vmax: Distance range for color mapping
        
    Returns:
        ax: Matplotlib 3D axis
    """
    # Plot mesh with distance colors
    ax = plot_mesh(mesh, vertex_colors=distances, cmap=cmap, 
                  ax=ax, figsize=figsize, title=title,
                  vmin=vmin, vmax=vmax, show_edges=False)
    
    # Mark source if provided
    if source_vertex is not None:
        if np.isscalar(source_vertex):
            source_vertex = [source_vertex]
        
        source_pos = mesh.vertices[source_vertex]
        ax.scatter(source_pos[:, 0], source_pos[:, 1], source_pos[:, 2],
                  c='red', s=100, marker='o', edgecolors='white', linewidths=2,
                  label='Source', zorder=10)
    
    # TODO: Add isolines (requires contour extraction on mesh)
    # This is more complex and would need additional libraries like gptoolbox
    
    return ax


def plot_vector_field(mesh,
                      vector_field: np.ndarray,
                      vertex_colors: Optional[np.ndarray] = None,
                      scale: float = 1.0,
                      density: float = 1.0,
                      ax: Optional[plt.Axes] = None,
                      figsize: Tuple[int, int] = (10, 8),
                      arrow_color: str = 'black',
                      title: Optional[str] = None) -> plt.Axes:
    """
    Visualize a vector field on a mesh.
    
    Args:
        mesh: Mesh object
        vector_field: Vectors on faces (nf x 3)
        vertex_colors: Optional vertex colors for mesh (nv,)
        scale: Arrow length scale
        density: Arrow density (1.0 = all faces, 0.5 = half, etc.)
        ax: Matplotlib axis
        figsize: Figure size
        arrow_color: Color for arrows
        title: Plot title
        
    Returns:
        ax: Matplotlib 3D axis
    """
    # Plot mesh
    ax = plot_mesh(mesh, vertex_colors=vertex_colors, 
                  ax=ax, figsize=figsize, title=title)
    
    # Get face centers
    centers = mesh.barycenters()
    
    # Subsample if density < 1
    if density < 1.0:
        n_show = int(mesh.nf * density)
        indices = np.random.choice(mesh.nf, n_show, replace=False)
    else:
        indices = np.arange(mesh.nf)
    
    # Draw arrows
    vf_norms = np.linalg.norm(vector_field, axis=1, keepdims=True)
    vf_norms[vf_norms < 1e-10] = 1.0
    vf_normalized = vector_field / vf_norms
    
    for idx in indices:
        center = centers[idx]
        direction = vf_normalized[idx] * scale
        
        # Draw arrow in both directions (for line fields)
        ax.quiver(center[0], center[1], center[2],
                 direction[0], direction[1], direction[2],
                 color=arrow_color, arrow_length_ratio=0.3, linewidth=1.5)
        ax.quiver(center[0], center[1], center[2],
                 -direction[0], -direction[1], -direction[2],
                 color=arrow_color, arrow_length_ratio=0.3, linewidth=1.5)
    
    return ax


def plot_comparison(mesh,
                   distance_fields: list,
                   labels: list,
                   source_vertex: Optional[int] = None,
                   n_cols: int = 3,
                   figsize: Tuple[int, int] = (15, 5),
                   cmap: str = 'jet') -> plt.Figure:
    """
    Plot multiple distance fields side by side for comparison.
    
    Args:
        mesh: Mesh object
        distance_fields: List of distance arrays
        labels: List of labels for each field
        source_vertex: Source vertex to mark
        n_cols: Number of columns in subplot grid
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        fig: Matplotlib figure
    """
    n_fields = len(distance_fields)
    n_rows = (n_fields + n_cols - 1) // n_cols
    
    # Get global min/max for consistent coloring
    all_dists = np.concatenate(distance_fields)
    vmin, vmax = all_dists.min(), all_dists.max()
    
    fig = plt.figure(figsize=figsize)
    
    for i, (dist, label) in enumerate(zip(distance_fields, labels)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        plot_distance_field(mesh, dist, source_vertex=source_vertex,
                          ax=ax, title=label, vmin=vmin, vmax=vmax, cmap=cmap)
    
    plt.tight_layout()
    return fig


def plot_gradient_magnitude(mesh,
                           distances: np.ndarray,
                           ax: Optional[plt.Axes] = None,
                           figsize: Tuple[int, int] = (10, 8),
                           title: str = 'Gradient Magnitude') -> plt.Axes:
    """
    Plot gradient magnitude on faces.
    
    Args:
        mesh: Mesh object
        distances: Distance function on vertices
        ax: Matplotlib axis
        figsize: Figure size
        title: Plot title
        
    Returns:
        ax: Matplotlib 3D axis
    """
    from .rgd_admm import compute_gradient_norm
    
    grad_norms = compute_gradient_norm(mesh, distances)
    
    ax = plot_mesh(mesh, face_colors=grad_norms, cmap='plasma',
                  ax=ax, figsize=figsize, title=title)
    
    return ax


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300):
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filename}")


def interactive_view(mesh,
                    vertex_colors: Optional[np.ndarray] = None):
    """
    Create interactive 3D view (requires PyVista).
    
    Args:
        mesh: Mesh object
        vertex_colors: Optional vertex colors
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista")
        return
    
    # Create PyVista mesh
    faces_pv = np.hstack([np.full((mesh.nf, 1), 3), mesh.faces])
    pv_mesh = pv.PolyData(mesh.vertices, faces_pv)
    
    if vertex_colors is not None:
        pv_mesh['colors'] = vertex_colors
    
    # Plot
    plotter = pv.Plotter()
    if vertex_colors is not None:
        plotter.add_mesh(pv_mesh, scalars='colors', cmap='jet', 
                        show_edges=False, smooth_shading=True)
    else:
        plotter.add_mesh(pv_mesh, color='lightgray', 
                        show_edges=True, smooth_shading=True)
    
    plotter.show()
