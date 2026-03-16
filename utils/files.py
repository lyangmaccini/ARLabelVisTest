import trimesh

def save_off_file(filename, mesh):
    if filename.split("/")[0] != "data":
        filename = "data/" + filename
    off_data = trimesh.exchange.off.export_off(mesh)
    # print("hello")
    with open(filename, "w") as f:
        f.write(off_data)
    print("Saved to " + filename)

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