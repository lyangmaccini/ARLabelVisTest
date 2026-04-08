import trimesh
import numpy as np
from utils.distances import furthest_delta_e76_points

def save_off_file(filename, mesh):
    # if filename.split("/")[0] != "data":
        # filename = "data/" + filename
    off_data = trimesh.exchange.off.export_off(mesh)
    # print("hello")
    with open(filename, "w") as f:
        f.write(off_data)
    print("Saved to " + filename)

def process_colors(rgb_range, allLABs):
    CandidateLABs = []
    CandidateRGBs = []
    
    CandidateRGBs = rgb_range.tolist()
    print('finished converting RGB to a list')
    CandidateLABs = np.apply_along_axis(lambda a: furthest_delta_e76_points(a, allLABs), axis=1, arr=rgb_range)
    CandidateLABs = CandidateLABs.tolist()
    print('finished generating all candidateLABs')
    return CandidateLABs, CandidateRGBs

def read_off(filename):
    """
    Read mesh from OFF file.
    
    Args:
        filename: Path to OFF file
        
    Returns:
        vertices, faces arrays
    """
    if not filename.endswith('.off'):
        filename = filename + '.off'
    with open(filename, 'r') as f:
        # Read header
        line = f.readline().strip()
        if line != 'OFF':
            raise ValueError('Not a valid OFF file')
        
        # Read counts
        line = f.readline().strip()
        while line.startswith('#') or len(line) == 0:
            line = f.readline().strip()
        
        counts = [int(x) for x in line.split()]
        nv, nf = counts[0], counts[1]
        
        # Read vertices
        vertices = []
        for i in range(nv):
            line = f.readline().strip()
            vertices.append([float(x) for x in line.split()])
        
        # Read faces
        faces = []
        for i in range(nf):
            line = f.readline().strip()
            parts = [int(x) for x in line.split()]
            if parts[0] != 3:
                raise ValueError('Only triangle meshes supported')
            faces.append(parts[1:4])
    
    return np.array(vertices), np.array(faces)

def write_point_cloud_ply(points, filename):
    """Write point cloud as PLY file"""
    with open(filename, 'w') as f:
        # Header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        
        # Data
        for point in points:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')