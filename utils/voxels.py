import numpy as np 
from utils.binvox_rw import write, Voxels
from utils.color_spaces import RGBtoLAB

# def convertToVoxels(allPoints, dim):
#     # print("expected size: " + str(len(allLABs)))
#     x_max, y_max, z_max = np.max(allPoints, axis=0)
#     x_min, y_min, z_min = np.min(allPoints, axis=0)

#     max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

#     voxels = np.zeros((dim, dim, dim), dtype=bool)

#     for lab in allPoints:
#         print("LAB")
#         x = lab[0] - x_min
#         y = lab[1] - y_min
#         z = lab[2] - z_min
#         print(lab)
#         print([x,y,z])
#         # everything over 0,0,0

#         x = dim / max_range * x 
#         y = dim / max_range * y 
#         z = dim / max_range * z 
#         print([x,y,z])
        
#         x = int(x)
#         y = int(y)
#         z = int(z)
#         print([x,y,z])

#         voxels[x][y][z] = True

#     print(np.min(allPoints, axis=0))
#     print(np.max(allPoints, axis=0))
#     print(voxels.shape)

#     print("sum of numpy array: " + str(np.sum(voxels)))
#     # print("done converting to voxels")
#     return voxels

def convertToVoxels(allLABs, dim):
    # print("expected size: " + str(len(allLABs)))
    x_max, y_max, z_max = np.max(allLABs, axis=0)
    x_min, y_min, z_min = np.min(allLABs, axis=0)



    max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

    # print(max_range)

    voxels = np.zeros((dim, dim, dim), dtype=np.bool_)

    for lab in allLABs:
        # print("LAB")
        # print(lab)
        x = lab[0] - x_min
        y = lab[1] - y_min
        z = lab[2] - z_min
        # print("XYZ")
        # print([x,y,z])

        x = dim / max_range * x
        y = dim / max_range * y
        z = dim / max_range * z

        # print("XYZ")
        # print([x,y,z])
       
        x = int(x)
        y = int(y)
        z = int(z)

        # print("INDICES")
        # print([x,y,z])

        voxels[x][y][z] = True

    print(np.min(allLABs, axis=0))
    print(np.max(allLABs, axis=0))
    print(voxels.shape)

    print("sum of numpy array: " + str(np.sum(voxels)))
    # print("done converting to voxels")
    return voxels


def writeVoxels(allPoints, dim, filename):
    # numpy_voxels = convertToVoxels(allPoints, dim)
    # voxels = Voxels(numpy_voxels, [dim, dim, dim], [0.0, 0.0, 0.0], 1.0, 'xyz')
    # with open(filename, "w", encoding="latin-1") as fp:
    #     write(voxels, fp)
    stepSize = 16
    allRGB = np.array([[r-1, g-1, b-1] for r in range(0, 257, stepSize)
                               for g in range(0, 257, stepSize)
                               for b in range(0, 257, stepSize)])
    allRGB = np.where(allRGB < 0, 0, allRGB)
    allRGB = np.where(allRGB > 255, 255, allRGB)
    allLABs = RGBtoLAB(allRGB)

    dim = 32
    voxels = convertToVoxels(allLABs, dim)
    v = Voxels(voxels, [dim, dim, dim], [0.0, 0.0, 0.0], 1.0, 'xyz')
    filepath = "allLABs.binvox"
    with open(filepath, 'w', encoding="latin-1") as fp:
        write(v, fp)
    print("Saved to file " + filepath)
