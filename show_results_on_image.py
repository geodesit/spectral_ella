import os
from spectral import *
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from webcolors import rgb_to_name


def get_image_show(hdr):
    # 119 52 26
    rgb = np.stack([hdr.read_band(114), hdr.read_band(52),
                    hdr.read_band(26)], axis=-1)
    # Can be brighter by changing the number to multiply with
    rgb = rgb / rgb.max() * 1.5

    return rgb


def get_pixels(img):
    dict_of_pix = {}

    for x in range(len(img[1])):
        for y in range(len(img)):
            is_all_zero = np.all((img[y][x] == 0))
            if not is_all_zero:
                dict_of_pix[(x, y)] = reversed(img[y, x])

    return dict_of_pix


def plot_image(dict_of_pix, hdr):
    fig, ax = plt.subplots()
    ax.imshow(hdr)
    if len(dict_of_pix) > 0:
        x, y = zip(*dict_of_pix.keys())
        name_colors = [rgb_to_name(val) for val in dict_of_pix.values()]

        ax.scatter(x, y, color=name_colors)
        plt.show()


def main():
    file_or_folder = input("Please enter file or folder mode: ")
    hdr_path = input("Please enter hdr file/folder path: ")
    img_output_path = input("Please enter image output file/folder path: ")

    if file_or_folder == "file":
        hdr = envi.open(hdr_path)
        rgb_hdr = get_image_show(hdr)
        img = cv.imread(img_output_path)

        dict_of_pix = get_pixels(img)
        plot_image(dict_of_pix, rgb_hdr)

    else:
        for file in os.listdir(hdr_path):
            if file.endswith(".hdr"):
                hdr = envi.open(hdr_path + "\\" + file)
                rgb_hdr = get_image_show(hdr)
                name = file.split("_")[1] + ".png"
                img = cv.imread(img_output_path + "\\" + name)

                dict_of_pix = get_pixels(img)
                plot_image(dict_of_pix, rgb_hdr)


if __name__ == "__main__":
    main()
