import numpy as np
from spectral import *
import matplotlib.pyplot as plt


def smooth(signature, box_pts):
    box = np.ones(box_pts) / box_pts
    signature = np.convolve(signature, box, mode='same')

    return signature


file_path = input("Please enter the file path: ")
x = input("Please enter the x coordinate: ")
y = input("Please enter the y coordinate: ")
smooth_factor = input("Please enter smooth factor: ")

hdr = envi.open(file_path)
pixel = hdr.read_pixel(y, x)
pixel = smooth(pixel, smooth_factor)
plt.plot(hdr.bands.centers, pixel)
plt.show()
