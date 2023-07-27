import spectral
import numpy as np
import os
import cv2

THE_GOOD_LIST_SWIR = [[914, 1347], [1443, 1807], [1961, 2400]]


def creat_folders(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)


def optim(image, sig_path, angle_list, s_v):
    print(image)
    creat_folders("results_17.5\\" + os.path.split(image)[-1])
    pixel_array = []
    spec = spectral.envi.open(image)
    rows, cols = spec.nrows, spec.ncols
    spec_image = np.array(spec.asarray())
    if s_v == "swir":
        the_bad_list = np.concatenate((np.arange(0, 3), np.arange(76, 92), np.arange(152, 178), np.arange(252, 270)),
                                      axis=0)
        spec_image = np.delete(spec_image, the_bad_list, axis=2)
        pixel_array = spec_image.reshape(spec_image.shape[0] * spec_image.shape[1], spec_image.shape[2])
        norms = np.linalg.norm(pixel_array, axis=1)
        norms = norms.reshape(norms.shape[0], 1)
        pixel_array = np.multiply(pixel_array, 1 / norms)

    for el in os.listdir(sig_path):
        print(el)
        for sig in os.listdir(sig_path + "\\" + el):
            pixel = angle_find(get_sig(sig_path + "\\" + el + "\\" + sig), pixel_array, angle_list[el], cols, rows,
                               image, el)
            write_pixel(pixel, cols, rows,image, el)


def write_pixel(pixel_array, cols, rows, image, el):
    if len(pixel_array) == 0:
        pass
    else:
        pixel_array = pixel_array.reshape((rows, cols))  # convert back to picter size
        pixel_array = np.transpose(np.nonzero(pixel_array))
        np.set_printoptions(threshold=np.inf)
        pixel_array = np.flip(pixel_array)
        if len(pixel_array) > 0:
            np.savetxt("results_17.5\\" + os.path.split(image)[-1] + "\\" + str(el) + ".txt", pixel_array, fmt="%d")


def angle_find(sig, pixel_array, ang, cols, rows, image, el):
    band_array = np.reshape(sig, (1, 207))
    norms = np.linalg.norm(band_array, axis=1)
    bands = np.multiply(band_array, 1 / norms)
    cosine_distance = bands @ pixel_array.transpose()
    angles = np.arccos(cosine_distance) * 180 / np.pi
    angles[angles > ang] = 0
    pic = angles / angles.max()
    pic = pic.reshape((rows, cols))
    cv2.imwrite("results_17.5\\" + os.path.split(image)[-1] + "\\" + str(el) + ".png", np.int64(pic * 255))
    return angles


def angle_find_file(sig, pixel_array):
    band_array = np.reshape(sig, (1, 207))
    norms = np.linalg.norm(band_array, axis=1)
    bands = np.multiply(band_array, 1 / norms)
    cosine_distance = bands @ pixel_array.transpose()
    angles = np.arccos(cosine_distance) * 180 / np.pi
    return angles


def get_sig(text):
    total_dict = []
    with open(text, "r") as f:
        arr = f.read().split(",")
        arr = [float(i) for i in arr]
        total_dict.append(arr)

    return total_dict


def write_global(nz_list, igm_path):
    hdr = spectral.envi.open(igm_path)
    cor_image = np.array(hdr.asarray())
    cor_array = cor_image[tuple(nz_list)]
