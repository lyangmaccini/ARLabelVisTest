import numpy as np
import matplotlib.pyplot as plt
import math
import alphashape
import matplotlib.pyplot as plt
import trimesh
from utils.binvox_rw import read_as_3d_array
from skimage import measure
import pyvista as pv
from scipy.spatial import cKDTree

def bindToNeuralBounding(bounded_binvox_filepath, dim, allPoints, allRGBs, visualize=False):
    voxels = np.zeros((dim, dim, dim))
    with open(bounded_binvox_filepath, "rb") as fp:
        voxels = read_as_3d_array(fp).data

    print("meshing")
    verts, faces, _, _ = measure.marching_cubes(voxels, 0.0)
    new_faces = []
    for face in faces:
        arr = np.insert(face, 0, 3)
        new_faces.append(arr)
    faces = np.hstack(new_faces)

    mesh = pv.PolyData(verts, faces)
    print("subdividing")
    mesh.subdivide(1, subfilter='loop', inplace=True)
    verts = mesh.points # in voxel space

    x_max, y_max, z_max = np.max(allPoints, axis=0)
    x_min, y_min, z_min = np.min(allPoints, axis=0)
    max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

    for i in range(len(verts)):
        new_x = max_range * verts[i][0] / dim
        new_y = max_range * verts[i][1] / dim
        new_z = max_range * verts[i][2] / dim

        new_x += x_min
        new_y += y_min
        new_z += z_min

        verts[i] = [new_x, new_y, new_z]

    print("binding")
    boundedLABs = bindToMeshBinding(verts, allPoints, voxels, dim)

    # for visualization (bounded LABs):
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(boundedLABs[:,0], boundedLABs[:,1], boundedLABs[:,2], c = allRGBs/255.0, alpha=0.15)
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])  
        ax.set_zlim([-100, 100])  
        ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.set_xlabel("L")
        ax.set_ylabel("a")
        ax.set_zlabel("b")
        ax.set_title("Bounded LABs Surface Plot")
        plt.show()

    print("number allLAB", len(allPoints))
    print("number boundedLABs", len(boundedLABs))
    return boundedLABs

def bindToMeshBinding(meshVertices, allLABs, voxels, voxelDim):
    x_max, y_max, z_max = np.max(allLABs, axis=0)
    x_min, y_min, z_min = np.min(allLABs, axis=0)
    max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

    tree = cKDTree(meshVertices)

    labs_array = np.array(allLABs)

    voxels_indices = pointsToVoxels(labs_array, voxelDim, x_min, y_min, z_min, max_range)
    inside = voxels[voxels_indices[:,0], voxels_indices[:,1], voxels_indices[:,2]].astype(bool)
    outside_mask = ~inside

    if np.any(outside_mask):
        _, indices = tree.query(labs_array[outside_mask])
        labs_array[outside_mask] = meshVertices[indices]

    return labs_array

def pointsToVoxels(labs, voxelDim, x_min, y_min, z_min, max_range):
    coords = (labs - np.array([x_min, y_min, z_min])) / max_range * voxelDim
    coords = np.clip(coords.astype(int), 0, voxelDim - 1)
    return coords

def vertex_distance(vertex, lab):
    return math.sqrt((vertex[0] - lab[0]) ** 2 + (vertex[1] - lab[1]) ** 2 + (vertex[2] - lab[2]) ** 2)

def bindToOptimizedMeshBinding(mesh: trimesh.Trimesh, allLABs):
    vertices = mesh.vertices
    print(len(allLABs))
    for i, lab in enumerate(allLABs):
        # print(i)
        if i % 1000 == 0:
            print(i)

        bestVertex = vertices[0]
        minDistance = vertex_distance(bestVertex, lab)
        for vertex in vertices:
            distance = vertex_distance(vertex, lab)
            if (distance < minDistance):
                bestVertex = vertex
                minDistance = distance
        allLABs[i] = bestVertex
    return allLABs 

def bindLABtoSphere(allLABPoints, allRGB):

    # Calculate the center of the point cloud
    center_x = np.sum(allLABPoints[:, 0]) / len(allLABPoints)
    center_y = np.sum(allLABPoints[:, 1]) / len(allLABPoints)
    center_z = np.sum(allLABPoints[:, 2]) / len(allLABPoints)
    center = np.array([center_x, center_y, center_z])
    print("center:")
    print(center)

    # Calculate which of the boundary points is closest to the center
    # Its distance becomes the radius of our binding circle
    shape = alphashape.alphashape(allLABPoints, alpha=0.005)
    boundaryLABs = shape.vertices
    bounded_distance = math.inf
    for point in boundaryLABs:
        distance = math.sqrt(pow(point[0] - center[0], 2) + pow(point[1] - center[1], 2) + pow(point[2] - center[2], 2))
        if ((distance < bounded_distance)):
            bounded_distance = distance
    print("bounded distance:")
    print(bounded_distance)

    inc = 0
    for i in range(len(allLABPoints)):
        point = allLABPoints[i]
        distance = math.sqrt(pow(point[0] - center[0], 2) + pow(point[1] - center[1], 2) + pow(point[2] - center[2], 2))
        # If the point is outside of the bounding sphere
        if (distance > bounded_distance):
            # print(point)
            inc = inc + 1
            # Move the point alond the radius by an amount equal to how far it is from the boundary
            move_amount = distance - bounded_distance
            # print(move_amount)
            direction = point - center # direction from center to point
            direction = direction / np.linalg.norm(direction)
            new_x = allLABPoints[i][0] - move_amount * direction[0]
            new_y = allLABPoints[i][1] - move_amount * direction[1]
            new_z = allLABPoints[i][2] - move_amount * direction[2]
            allLABPoints[i] = np.array([new_x, new_y, new_z])
            # print(allLABPoints[i])
    print("number points moved:")
    print(inc)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel("L")
    ax.set_ylabel("a")
    ax.set_zlabel("b")

    #Plot the bounding sphere
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    r = bounded_distance
    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] +r * np.cos(phi)
    ax.plot_surface(x, y, z, color='red', alpha=0.5)

    # Plot the positions of the bounded LAB points in their original RGB color
    X = allLABPoints[:, 0]
    Y = allLABPoints[:, 1]
    Z = allLABPoints[:, 2] 
    ax.scatter(X, Y, Z, c = allRGB/255.0, alpha=0.15)

    #Plot the possible boundary points
    # ax.scatter(boundaryLABs[:,0], boundaryLABs[:,1], boundaryLABs[:,2], c='blue')

    ax.set_xlim([-100, 100])   # Set x-axis limits
    ax.set_ylim([-100, 100])   # Set y-axis limits
    ax.set_zlim([-100, 100])   # Set z-axis limits
    ax.set_box_aspect([1.0, 1.0, 1.0])

    plt.show()

    return allLABPoints
