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

RGB_file = open("CorrespondingRGBVals.txt", "r")
LAB_file = open("CandidateLABvals.txt", "r")

rgbVals = RGB_file.readlines()
labVals = LAB_file.readlines()

rgbVals = [np.asarray(rgb.strip("\n").split(",")).astype(np.uint8) for rgb in rgbVals]
labVals = [np.asarray(lab.strip("\n").split(",")).astype('float') for lab in labVals]

labVals = np.asarray(labVals)
rgbVals = np.asarray(rgbVals)
# uniqueLABVals, uniqueLABValsIndices = np.unique(labVals, axis=0, return_index = True)
# correspondingRGBVals = np.take(rgbVals, uniqueLABValsIndices, axis=0)
# correspondingRGBValsToLAB = np.asarray([RGBToLAB(rgb) for rgb in correspondingRGBVals])

new_RGB_file = open("AllCorrespondingRGBVals.txt", "w")
new_LAB_file = open("AllCandidateLABvals.txt", "w")

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
x = np.arange(-1, 256, 16)
x[0] = 0
y = np.arange(-1, 256, 16)
y[0] = 0
z = np.arange(-1, 256, 16)
z[0] = 0
X, Y, Z = np.meshgrid(x, y, z)
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
values = np.zeros((17, 17, 17, 3)) #Step size + 1 number of values in each dimension
for i in range(17):
    for j in range(17):
        for k in range(17):
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
                new_RGB_file.write(str(r)+"," + str(g)+"," + str(b)+"\n")
                new_LAB_file.write(str(labVal.l) + "," + str(labVal.a) + "," + str(labVal.b) + "\n")
            else:
                interpolatedLAB = fn(np.asarray([r, g, b]))[0]
                new_RGB_file.write(str(r)+"," + str(g)+"," + str(b)+"\n")
                new_LAB_file.write(str(interpolatedLAB[0]) + "," + str(interpolatedLAB[1]) + "," + str(interpolatedLAB[2]) + "\n")

new_RGB_file.close()
new_LAB_file.close()