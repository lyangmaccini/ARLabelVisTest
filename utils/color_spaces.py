import numpy as np
import math
import matplotlib.pyplot as plt

def RGBtoLinear(RGB):
    RGB = np.array(RGB) / 255.0
    mask = RGB > 0.04045

    RGB[mask] = ((RGB[mask] + 0.055) / 1.055) ** 2.4
    RGB[~mask] /= 12.92
    return RGB

def RGBtoXYZ(RGB):
    return np.dot(RGB, np.array([[0.4124, 0.3576, 0.1805],
                                [0.2126, 0.7152, 0.0722],
                                [0.0193, 0.1192, 0.9505]]))

def RGBtoOKLAB(RGB):
    # input: RGB array, (n, 3)
    # sRGB -> linear RGB
    RGB = RGBtoLinear(RGB)

    # linear RGB -> OKLAB
    l = 0.4122214708 * RGB[:,0] + 0.5363325363 * RGB[:,1] + 0.0514459929 * RGB[:,2]
    m = 0.2119034982 * RGB[:,0] + 0.6806995451 * RGB[:,1] + 0.1073969566 * RGB[:,2]
    s = 0.0883024619 * RGB[:,0] + 0.2817188376 * RGB[:,1] + 0.6299787005 * RGB[:,2]

    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    RGB[:,0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    RGB[:,1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    RGB[:,2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return RGB 

def XYZtoLAB(XYZ):
    XYZ /= np.array([95.047, 100.0, 108.883])
    mask = XYZ > 0.008856
    XYZ[mask] = XYZ[mask] ** (1/3)
    XYZ[~mask] = (7.787 * XYZ[~mask]) + (16/116)

    L = (116 * XYZ[:, 1]) - 16
    a = 500 * (XYZ[:, 0] - XYZ[:, 1])
    b = 200 * (XYZ[:, 1] - XYZ[:, 2]) #bLab

    return np.stack([L, a, b], axis=1)

# def RGBtoLAB(RGB):
#     linearRGB = RGBtoLinear(RGB)
#     XYZ = RGBtoXYZ(linearRGB)
#     return XYZtoLAB(XYZ)

def RGBtoLAB(RGB):
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

def RGBtoOKLCH(RGB):
    print(RGB.shape)
    LAB = RGBtoLAB(RGB)
    l = LAB[:,0]
    a = LAB[:,1]
    bLab = LAB[:,2]

    c = np.sqrt(a ** 2 + bLab ** 2)
    h = np.arctan2(bLab, a) * (180 / math.pi)
    print(h.shape)
    h[h < 0] += 360

    return np.stack([l / 100, c / 100, h], axis=1)

def getAllRGBS(stepSize=4):
    allRGBs = np.array([[r-1, g-1, b-1] for r in range(0, 257, stepSize)
                               for g in range(0, 257, stepSize)
                               for b in range(0, 257, stepSize)])
    allRGBs = np.where(allRGBs < 0, 0, allRGBs)
    allRGBs = np.where(allRGBs > 255, 255, allRGBs)
    return allRGBs

def testOKLAB(stepSize=4):
    allRGBs = getAllRGBS(stepSize=stepSize)
    allOKLABs = RGBtoOKLAB(allRGBs)
    plot_lab_points_3d(allOKLABs, RGBs=allRGBs)

def testOKLCH(stepSize=4):
    allRGBs = getAllRGBS(stepSize=stepSize)
    allOKLCHs = RGBtoOKLCH(allRGBs)
    plot_lab_points_3d(allOKLCHs, RGBs=allRGBs)

def plot_lab_points_3d(newPoints, RGBs=None, subsample=1):
    """
    allLABs: (N,3) LAB points
    allRGBs: (N,3) RGB points in [0,255]
    furthestRGBs: (M,3) optional RGB points to highlight
    subsample: plot every k-th point for speed
    """

    newPoints = np.asarray(newPoints)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if RGBs is not None:
        ax.scatter(
            newPoints[::subsample, 0],
            newPoints[::subsample, 1],
            newPoints[::subsample, 2],
            c=RGBs[::subsample]/255.0,
            s=120,
            linewidth=1.5
        )
    else:
        ax.scatter(
            newPoints[::subsample, 0],
            newPoints[::subsample, 1],
            newPoints[::subsample, 2],
            s=120,
            linewidth=1.5
        )

    ax.set_xlabel("L*")
    ax.set_ylabel("a*")
    ax.set_zlabel("b*")

    ax.set_title("New space (colored by original RGB)")
    ax.legend()
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.show()

def main():
    stepSize = 8
    testOKLAB(stepSize=stepSize)
    testOKLCH(stepSize=stepSize)

if __name__ == "__main__":
    main()