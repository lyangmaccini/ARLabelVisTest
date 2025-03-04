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


def RGBToLAB(RGB):
    RGB = np.array(RGB) / 255.0
    mask = RGB > 0.04045

    RGB[mask] = ((RGB[mask] + 0.055) / 1.055) ** 2.4
    RGB[~mask] /= 12.92
    RGB *= 100


    XYZ = np.dot(RGB, np.array([[0.4124, 0.3576, 0.1805],
                                [0.2126, 0.7152, 0.0722],
                                [0.0193, 0.1192, 0.9505]]))

    XYZ /= np.array([95.047, 100.0, 108.883])
    mask = XYZ > 0.008856
    XYZ[mask] = XYZ[mask] ** (1/3)
    XYZ[~mask] = (7.787 * XYZ[~mask]) + (16/116)

    L = (116 * XYZ[:, 1]) - 16
    a = 500 * (XYZ[:, 0] - XYZ[:, 1])
    b = 200 * (XYZ[:, 1] - XYZ[:, 2])

    return np.stack([L, a, b], axis=1)

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
    #ax.plot_surface(x, y, z, color='red', alpha=0.5)

    # Plot the positions of the bounded LAB points in their original RGB color
    X = allLABPoints[:, 0]
    Y = allLABPoints[:, 1]
    Z = allLABPoints[:, 2] 
    ax.scatter(X, Y, Z, c = allRGB/255.0, alpha=0.15)

    #Plot the possible boundary points
    #ax.scatter(boundaryLABs[:,0], boundaryLABs[:,1], boundaryLABs[:,2], c='blue')

    ax.set_xlim([-100, 50])   # Set x-axis limits
    ax.set_ylim([-100, 50])   # Set y-axis limits
    ax.set_zlim([-100, 50])   # Set z-axis limits
    ax.set_box_aspect([1.0, 1.0, 1.0])
    plt.show()

    return allLABPoints


def findPointAtMaxDistance(inputRGB, allLABPoints):
    inputLAB = RGBToLAB([inputRGB])[0]
    allLABPoints = np.array(allLABPoints)

    distances = deltaE_cie76(np.tile(inputLAB, (len(allLABPoints), 1)), allLABPoints)
    max_distance_index = np.argmax(distances)

    return allLABPoints[max_distance_index]

def process_colors(rgb_range, allLABs):
    CandidateLABs = []
    CandidateRGBs = []
    
    CandidateRGBs = rgb_range.tolist()
    print('finished converting RGB to a list')
    CandidateLABs = np.apply_along_axis(lambda a: findPointAtMaxDistance(a, allLABs), axis=1, arr=rgb_range)
    CandidateLABs = CandidateLABs.tolist()
    print('finished generating all candidateLABs')
    return CandidateLABs, CandidateRGBs



def main():
    num_cpus = os.cpu_count() 
    n_processes = num_cpus - 4 # Change this to use more/less CPUs. 
    print("Number of CPUs:", num_cpus, "Number of CPUs we are using:", n_processes)

    stepSize = 16 
    # Sample: 0, 15, 31, 45, ... 255
    allRGB = np.array([[r-1, g-1, b-1] for r in range(0, 257, stepSize)
                               for g in range(0, 257, stepSize)
                               for b in range(0, 257, stepSize)])
    allRGB = np.where(allRGB < 0, 0, allRGB)
    allRGB = np.where(allRGB > 255, 255, allRGB)
    allLABs = RGBToLAB(allRGB)
    boundedLABs = bindLABtoSphere(allLABs, allRGB)

    print("number allLAB", len(allLABs))
    print("number boundedLABs", len(boundedLABs))

    print("finished preparing all LABs")
    rgb_ranges = np.array_split(allRGB, n_processes)

    with Pool(n_processes) as pool:
        results = pool.starmap(process_colors, [(rgb_range, boundedLABs) for rgb_range in rgb_ranges])

    CandidateLABs = []
    CandidateRGBs = []
    for lab, rgb in results:
        CandidateLABs += lab
        CandidateRGBs += rgb

    with open("CandidateLABvals.txt", "w") as f, open("CorrespondingRGBVals.txt", "w") as f2:
        for lab, rgb in zip(CandidateLABs, CandidateRGBs):
            f.write(f"{lab[0]},{lab[1]},{lab[2]}\n")
            f2.write(f"{rgb[0]},{rgb[1]},{rgb[2]}\n")

    print("Number of unique LABs:", len(CandidateLABs))


if __name__ == "__main__":
    main()