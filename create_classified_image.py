import tensorflow as tf
import numpy as np
from spectral import *
import matplotlib.pyplot as plt

IMAGE_PATH = "E:\\NISOY\\hod_hasron04_01_23\\f3\\swir\\100051_hodhasharon_04_01_23_2023_01_04_12_26_53\\raw_2570_rd_rf"
HDR_PATH = "E:\\NISOY\\hod_hasron04_01_23\\f3\\swir\\100051_hodhasharon_04_01_23_2023_01_04_12_26_53" \
           "\\raw_2570_rd_rf.hdr"
MODEL_PATH = "grass.h5"
CLASSES = [0, 1]
BANDS_NUM = 270
DICT_VAL_TO_COLOR = {0: (255, 0, 0),
                     1: (0, 255, 0)}


def find_closest(arr, val):
    idx = np.abs(arr - val).argmin()
    return arr[idx]


model = tf.keras.models.load_model(MODEL_PATH)
hdr = envi.open(HDR_PATH, IMAGE_PATH)
wvl = hdr.bands.centers
rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
meta = hdr.metadata
new_image = np.zeros((rows, cols, 3))

for x in range(30, 41):
    for y in range(498, 519):
        pixel = hdr.read_pixel(x, y).reshape(-1, BANDS_NUM)
        res = model.predict(pixel)
        closest_value = find_closest(np.array(CLASSES), res)
        new_image[x][y] = DICT_VAL_TO_COLOR[closest_value]

plt.imshow(new_image)
plt.show()
