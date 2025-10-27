import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from skimage.color import deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, lab2rgb 
from tqdm import tqdm
import os
from multiprocessing import Pool
import time
import math
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import alphashape
from scipy.optimize import curve_fit
import cvxpy as cp
import binvox_rw 
from skimage import measure
import matplotlib.pyplot as plt
import pyvista as pv 

def bindToMeshBinding(meshVertices, allLABs, voxels, voxelDim):
    x_max, y_max, z_max = np.max(allLABs, axis=0)
    x_min, y_min, z_min = np.min(allLABs, axis=0)
    max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])
    print(len(allLABs))
    for i, lab in enumerate(allLABs):
        # print(i)
        if i % 1000 == 0:
            print(i)
        voxel = pointToVoxel(lab, voxelDim, x_min, y_min, z_min, max_range)
        if not bool(voxels[voxel[0]][voxel[1]][voxel[2]]):
            # print(i)
        # if not insideVoxels(voxels, lab):
            minDistance = 1000000000
            bestVertex = meshVertices[0]
            for vertex in meshVertices:
                distance = math.sqrt((vertex[0] - lab[0]) ** 2 + (vertex[1] - lab[1]) ** 2 + (vertex[2] - lab[2]) ** 2)
                if (distance < minDistance):
                    bestVertex = vertex
                    minDistance = distance
            allLABs[i] = bestVertex
    return allLABs 

def bindLABtoEllipsoid(allLABPoints, allRGB):

    # Create a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Find which point make up the boundary
    shape = alphashape.alphashape(allLABPoints, alpha=0.05)
    boundaryLABs = shape.vertices

    dim  = boundaryLABs.shape[1]
    hull = ConvexHull(boundaryLABs)
    A = hull.equations[:,0:dim]
    b = -hull.equations[:,dim]
    B = cp.Variable((dim,dim), PSD=True) #Ellipsoid
    d = cp.Variable(dim)                 #Center

    constraints = [cp.norm(B@A[i],1.5)+A[i]@d<=b[i] for i in range(len(A))]
    prob = cp.Problem(cp.Minimize(-cp.log_det(B)), constraints)
    optval = prob.solve()

    shape_matrix = np.linalg.inv(B.value.T @ B.value)
    center = d.value
    eigenvalues, eigenvectors = np.linalg.eig(shape_matrix)
    rotation = eigenvectors
    rotation_inv = np.linalg.inv(rotation)
    axes = 1/ np.sqrt(eigenvalues)

    # To graph the ellipsoid:
    # u = np.linspace(0.0, 2.0 * np.pi, 100)
    # v = np.linspace(0.0, np.pi, 100)
    # x = axes[0] * np.outer(np.cos(u), np.sin(v))
    # y = axes[1] * np.outer(np.sin(u), np.sin(v))
    # z = axes[2] * np.outer(np.ones_like(u), np.cos(v))
    # for i in range(len(x)):
    #     for j in range(len(x)):
    #         [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    # ax.scatter(x, y, z, c = "blue", alpha=0.5)

    # Create a copy of the original Lab Points
    og_allLabPoints = np.copy(allLABPoints)

    for i, point in enumerate(allLABPoints):
        # If the point is outside of the binding ellipsoid, move it onto the ellipsoid using ray intersection
        if not insideRotatedEllipsoid(point[0], point[1], point[2], shape_matrix, center):
            point = point - center
            point = rotation_inv @ point
            direction = -point
            new_point = ellipsoidIntersection(point[0], point[1], point[2], direction[0], direction[1], direction[2], 1, 1, 1)
            new_point = rotation @ new_point
            new_point = new_point  + center
            allLABPoints[i] = np.asarray(new_point)

    # Plot the positions of the bounded LAB points in their original RGB color
    X = allLABPoints[:, 0]
    Y = allLABPoints[:, 1]
    Z = allLABPoints[:, 2]
    # ax.scatter(X, Y, Z, c = allRGB/255.0, alpha=0.5)
    
    # Plot the positions of the original/unchanged LAB points in their original RGB color
    originalX = og_allLabPoints[:, 0]
    originalY = og_allLabPoints[:, 1]
    originalZ = og_allLabPoints[:, 2]
    # ax.scatter(originalX, originalY, originalZ, c = "red", alpha=0.05)

    # Set labels and title
    ax.set_xlim([-100, 100])   # Set x-axis limits
    ax.set_ylim([-100, 100])   # Set y-axis limits
    ax.set_zlim([-100, 100])   # Set z-axis limits
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlabel("L")
    ax.set_ylabel("a")
    ax.set_zlabel("b")
    ax.set_title("Ellipsoid Surface Plot")

    # plt.show()

    return allLABPoints

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
            inc = inc + 1
            # Move the point alond the radius by an amount equal to how far it is from the boundary
            move_amount = distance - bounded_distance
            direction = point - center # direction from center to point
            direction = direction / np.linalg.norm(direction)
            new_x = allLABPoints[i][0] - move_amount * direction[0]
            new_y = allLABPoints[i][1] - move_amount * direction[1]
            new_z = allLABPoints[i][2] - move_amount * direction[2]
            allLABPoints[i] = np.array([new_x, new_y, new_z])
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
    # ax.plot_surface(x, y, z, color='red', alpha=0.5)

    # Plot the positions of the bounded LAB points in their original RGB color
    X = allLABPoints[:, 0]
    Y = allLABPoints[:, 1]
    Z = allLABPoints[:, 2] 
    #ax.scatter(X, Y, Z, c = allRGB/255.0, alpha=0.15)

    #Plot the possible boundary points
    #ax.scatter(boundaryLABs[:,0], boundaryLABs[:,1], boundaryLABs[:,2], c='blue')

    ax.set_xlim([-100, 100])   # Set x-axis limits
    ax.set_ylim([-100, 100])   # Set y-axis limits
    ax.set_zlim([-100, 100])   # Set z-axis limits
    ax.set_box_aspect([1.0, 1.0, 1.0])

    plt.show()

    return allLABPoints


# HELPERS:
def insideRotatedEllipsoid(px, py, pz, shape_matrix, center):
    temp = np.array([px - center[0], py - center[1], pz - center[2]])
    return temp.T @ shape_matrix @ temp - 1 < 0

def ellipsoidIntersection(px, py, pz, dx, dy, dz, a, b, c):
    # a, b, c = ellipsoid radii
    # px, py, pz = original point
    # dx, dy, dz = center - point

    A1 = (b ** 2) * (c ** 2) * (dx ** 2) + (a ** 2) * (c ** 2) * (dy ** 2) + (a ** 2) * (b ** 2) * (dz ** 2)
    B1 = 2 * (b ** 2) * (c ** 2) * px * dx + 2 * (a ** 2) * (c ** 2) * py * dy + 2 * (a ** 2) * (b ** 2) * pz * dz
    C1 = (b ** 2) * (c ** 2) * (px ** 2) + (a ** 2) * (c ** 2) * (py ** 2) + (a ** 2) * (b ** 2) * (pz ** 2) - (a ** 2) * (b ** 2) * (c ** 2)
    D = B1 ** 2 - 4 * A1 * C1
    t = 0
    if D > 0:
        t1 = (-B1 + math.sqrt(D)) / (2 * A1)
        t2 = (-B1 - math.sqrt(D)) / (2 * A1)
        # print("hello")
        # print(t1)
        # print(t2)
        if (t1 < 0):
            if (t2 > 0):
                t = t2
            else:
                print("warning1")
                #if both negative no solution
        if (t2 < 0):
            if (t1 > 0):
                t = t1
            else:
                print("warning2")
        if (t1 > 0 and t2 > 0):
            if (t1 < t2):
                t = t1
            else:
                t = t2
    elif D == 0:
        t = -B1 / (2 * A1)
        #if determinant negative no solution 
    else:
        print("warning3")

    return [px + t * dx, py + t * dy, pz + t * dz]

def pointToVoxel(point, dim, x_min, y_min, z_min, max_range):
        x = point[0] - x_min
        y = point[1] - y_min
        z = point[2] - z_min

        x = dim / max_range * x 
        y = dim / max_range * y 
        z = dim / max_range * z 
        
        x = int(x)
        y = int(y)
        z = int(z)
        return np.array([x, y, z])

def voxelToPoint(allLABs, voxel, dim):
    x_max, y_max, z_max = np.max(allLABs, axis=0)
    x_min, y_min, z_min = np.min(allLABs, axis=0)

    max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

    new_x = max_range * voxel[0] / dim
    new_y = max_range * voxel[1] / dim
    new_z = max_range * voxel[2] / dim

    new_x += x_min
    new_y += y_min
    new_z += z_min

    return np.array([new_x, new_y, new_z])

def insideVoxels(voxels, point):
    voxel = pointToVoxel(point)
    return bool(voxels[voxel])


# UNUSED:
def ellipsoidFunc(data, a, b, c):
    x, y, z = data
    center = np.asarray([63, -61, 6])
    return ((x - center[0])**2/a) + ((y - center[1])**2/b) + ((z - center[2])**2/c) - 1

def ellipsoidFuncPos(center, x, y, a, b, c):
    return c * np.sqrt(1 - (((x - center[0]) / a) ** 2 + ((y - center[1])/b) ** 2)) + center[2]

def ellipsoidFuncNeg(center, x, y, a, b, c):
    return -c * np.sqrt(1 - (((x - center[0]) / a) ** 2 + ((y - center[1])/b) ** 2)) + center[2]


def insideEllipsoid(px, py, pz, a, b, c):
    return (px ** 2) / (a ** 2) + (py ** 2) / (b ** 2) + (pz ** 2) / (c ** 2) - 1 < 0
