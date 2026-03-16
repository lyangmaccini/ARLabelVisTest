import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from skimage.color import deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, lab2rgb 
from tqdm import tqdm
import os
from multiprocessing import Pool
import time
import scipy
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, interpn

class LABData:
    l: float
    a: float
    b: float

def InitializeLookupTexture(a, b, c):
    lst = [[ [LABData for col in range(a)] for col in range(b)] for row in range(c)]
    return lst

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

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    return np.asarray([L, a, b])

def interpolate_files(RGB_filepath, LAB_filepath, interval, new_LAB_filepath):
    RGB_file = open(RGB_filepath, "r")
    LAB_file = open(LAB_filepath, "r")


    rgbVals = RGB_file.readlines()
    labVals = LAB_file.readlines()

    rgbVals = [np.asarray(rgb.strip("\n").split(",")).astype(np.uint8) for rgb in rgbVals]
    labVals = [np.asarray(lab.strip("\n").split(",")).astype('float') for lab in labVals]

    labVals = np.asarray(labVals)
    rgbVals = np.asarray(rgbVals)

    new_LAB_file = open(new_LAB_filepath, "w")

    LookupTexture =  InitializeLookupTexture(256, 256, 256)

    # Write existing rgb and lab values into the lookup texture
    for idx in range(len(rgbVals)):
        rgb = rgbVals[idx]
        lab = labVals[idx]
        labPoint = LABData()
        labPoint.l = lab[0]
        labPoint.a = lab[1]
        labPoint.b = lab[2]

        LookupTexture[rgb[0]][rgb[1]][rgb[2]] = labPoint

    # Use the built-in RegularGridInterpolator
    stepsize = int(256/4)
    if interval == 1:
        x = np.arange(0, 256, interval)
        x[0] = 0
        y = np.arange(0, 256, interval)
        y[0] = 0
        z = np.arange(0, 256, interval)
        z[0] = 0
    else:
        x = np.arange(-1, 256, interval)
        x[0] = 0
        y = np.arange(-1, 256, interval)
        y[0] = 0
        z = np.arange(-1, 256, interval)
        z[0] = 0
    X, Y, Z = np.meshgrid(x, y, z)
    values = np.zeros((stepsize + 1, stepsize + 1, stepsize + 1, 3)) #Step size + 1 number of values in each dimension
    for i in range(stepsize + 1):
        for j in range(stepsize + 1):
            for k in range(stepsize + 1):
                LAB = LookupTexture[x[i]][y[j]][z[k]]
                values[i,j,k, 0] = LAB.l
                values[i,j,k, 1] = LAB.a
                values[i,j,k, 2] = LAB.b
    fn = RegularGridInterpolator((x,y,z), values)

    for r in range(0, 256):
        for g in range(0, 256):
            for b in range(0, 256):
                if r%16 == 15 and g%16 == 15 and b%15 == 16:
                    labVal = LookupTexture[r][g][b]
                    new_LAB_file.write(str(labVal.l) + "," + str(labVal.a) + "," + str(labVal.b) + "\n")
                else:
                    interpolatedLAB = fn(np.asarray([r, g, b]))[0]
                    new_LAB_file.write(str(interpolatedLAB[0]) + "," + str(interpolatedLAB[1]) + "," + str(interpolatedLAB[2]) + "\n")

    new_LAB_file.close()

if __name__ == "__main__":
    RGB_file = open("CorrespondingRGBVals_MATLAB_030_4.txt", "r")
    LAB_file = open("CandidateLABvals_MATLAB_030_4.txt", "r")


    rgbVals = RGB_file.readlines()
    labVals = LAB_file.readlines()

    rgbVals = [np.asarray(rgb.strip("\n").split(",")).astype(np.uint8) for rgb in rgbVals]
    labVals = [np.asarray(lab.strip("\n").split(",")).astype('float') for lab in labVals]

    labVals = np.asarray(labVals)
    rgbVals = np.asarray(rgbVals)
    # uniqueLABVals, uniqueLABValsIndices = np.unique(labVals, axis=0, return_index = True)
    # correspondingRGBVals = np.take(rgbVals, uniqueLABValsIndices, axis=0)
    # correspondingRGBValsToLAB = np.asarray([RGBToLAB(rgb) for rgb in correspondingRGBVals])

    # new_RGB_file = open("AllCorrespondingRGBVals_16.txt", "w")
    new_LAB_file = open("AllCandidateLABvals_MATLAB_030_4.txt", "w")

    LookupTexture =  InitializeLookupTexture(256, 256, 256)

    # Write existing rgb and lab values into the lookup texture
    for idx in range(len(rgbVals)):
        rgb = rgbVals[idx]
        lab = labVals[idx]
        labPoint = LABData()
        labPoint.l = lab[0]
        labPoint.a = lab[1]
        labPoint.b = lab[2]

        LookupTexture[rgb[0]][rgb[1]][rgb[2]] = labPoint

    # Use the built-in RegularGridInterpolator
    # stepsize = 4
    # interval = 64
    stepsize = 64
    interval = 4
    if interval == 1:
        x = np.arange(0, 256, interval)
        x[0] = 0
        y = np.arange(0, 256, interval)
        y[0] = 0
        z = np.arange(0, 256, interval)
        z[0] = 0
    else:
        x = np.arange(-1, 256, interval)
        x[0] = 0
        y = np.arange(-1, 256, interval)
        y[0] = 0
        z = np.arange(-1, 256, interval)
        z[0] = 0
    print(x)
    print(x.shape)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    values = np.zeros((stepsize + 1, stepsize + 1, stepsize + 1, 3)) #Step size + 1 number of values in each dimension
    for i in range(stepsize + 1):
        for j in range(stepsize + 1):
            for k in range(stepsize + 1):
                LAB = LookupTexture[x[i]][y[j]][z[k]]
                values[i,j,k, 0] = LAB.l
                values[i,j,k, 1] = LAB.a
                values[i,j,k, 2] = LAB.b
    fn = RegularGridInterpolator((x,y,z), values)

    print(np.max(values))
    evaluation_values = values / np.max(values)

    g_x = np.gradient(evaluation_values, axis=0)
    g_y = np.gradient(evaluation_values, axis=1)
    g_z = np.gradient(evaluation_values, axis=2)

    g = g_x ** 2 + g_y ** 2 + g_z ** 2

    g_across_LAB = np.sum(g, axis=-1)

    print("percentiel " + str(np.percentile(g_across_LAB, 95)))
    mean_g = np.mean(g_across_LAB)
    max_g = np.max(g_across_LAB)

    print("Mean gradient: " + str(mean_g))
    print("Max gradient: " + str(max_g))

    dx = evaluation_values[1:, :, :, :] - evaluation_values[:-1, :, :, :]
    dy = evaluation_values[:, 1:, :, :] - evaluation_values[:, :-1, :, :]
    dz = evaluation_values[:, :, 1:, :] - evaluation_values[:, :, :-1, :]

    dist_x = np.linalg.norm(dx, axis=-1)
    dist_y = np.linalg.norm(dy, axis=-1)
    dist_z = np.linalg.norm(dz, axis=-1)

    max_dist = max(dist_x.max(), dist_y.max(), dist_z.max())

    mean_dist = (dist_x.mean() + dist_y.mean() + dist_z.mean()) / 3

    print("Mean neighboring distances: " + str(mean_dist))
    print("max neightboring distance: " + str(max_dist))

    all_dists = np.concatenate([dist_x.ravel(), dist_y.ravel(), dist_z.ravel()])

    sharpness = np.percentile(all_dists, 95)
    print("sharpness: " + str(sharpness))

    print("var: " + str(np.var(all_dists)))

    for r in range(0, 256):
        for g in range(0, 256):
            for b in range(0, 256):
                if r%16 == 15 and g%16 == 15 and b%15 == 16:
                    labVal = LookupTexture[r][g][b]
                    # new_RGB_file.write(str(r)+"," + str(g)+"," + str(b)+"\n")
                    new_LAB_file.write(str(labVal.l) + "," + str(labVal.a) + "," + str(labVal.b) + "\n")
                else:
                    interpolatedLAB = fn(np.asarray([r, g, b]))[0]
                    # new_RGB_file.write(str(r)+"," + str(g)+"," + str(b)+"\n")
                    new_LAB_file.write(str(interpolatedLAB[0]) + "," + str(interpolatedLAB[1]) + "," + str(interpolatedLAB[2]) + "\n")

    # new_RGB_file.close()
    new_LAB_file.close()