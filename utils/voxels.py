import numpy as np 

def convertToVoxels(allLABs, dim):
    # print("expected size: " + str(len(allLABs)))
    x_max, y_max, z_max = np.max(allLABs, axis=0)
    x_min, y_min, z_min = np.min(allLABs, axis=0)

    max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

    voxels = np.zeros((dim, dim, dim), dtype=np.bool8)

    for lab in allLABs:
        x = lab[0] - x_min
        y = lab[1] - y_min
        z = lab[2] - z_min

        x = dim / max_range * x 
        y = dim / max_range * y 
        z = dim / max_range * z 
        
        x = int(x)
        y = int(y)
        z = int(z)

        voxels[x][y][z] = True

    print(np.min(allLABs, axis=0))
    print(np.max(allLABs, axis=0))
    print(voxels.shape)

    print("sum of numpy array: " + str(np.sum(voxels)))
    # print("done converting to voxels")
    return voxels