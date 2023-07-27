import itertools
import math
from itertools import product
from tqdm.gui import tqdm
from matplotlib.widgets import PolygonSelector
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from spectral import *
import os
from matplotlib.backend_bases import MouseButton
import sys
from optim_code import *
from matplotlib.widgets import Slider
from webcolors import rgb_to_name
from webcolors import name_to_rgb
import cv2 as cv
import csv
import seaborn as sns

# List of non absorption bands
# [[914, 1347], [1443, 1807], [1961, 2400]]
THE_GOOD_LIST_SWIR = [[914, 1347], [1443, 1807], [1961, 2400]]
THE_GOOD_LIST_VNIR = [[400, 755], [773, 846], [853, 1000]]
locations = []
wvl = []
dict_angles = {}
dirs_colors = {}
# Flag for polygons selections
FLG_EXIT = False
# Colors for different materials in classification
COLORS = {1: name_to_rgb("lavender"),
          2: name_to_rgb("gray"),
          3: name_to_rgb("white"),
          4: name_to_rgb("brown"),
          5: name_to_rgb("red"),
          6: name_to_rgb("salmon"),
          7: name_to_rgb("chocolate"),
          8: name_to_rgb("gold"),
          9: name_to_rgb("yellow"),
          10: name_to_rgb("olive"),
          11: name_to_rgb("green"),
          12: name_to_rgb("aqua"),
          13: name_to_rgb("blue"),
          14: name_to_rgb("purple"),
          15: name_to_rgb("pink"),
          16: name_to_rgb("lime"),
          17: name_to_rgb("orange"),
          18: name_to_rgb("tomato"),
          19: name_to_rgb("tan"),
          20: name_to_rgb("teal")}
# Range for converting local pixels to global coordinates
RANGE = 0.000001


def open_wvl_file(file_name):
    """
    The function creates a wavelengths array
    :param file_name: The file which contains the wavelengths
    :return: The wavelengths array
    """
    with open(file_name, "r") as f:
        arr = f.read().split(",")

    return [float(i) for i in arr]


WVL_SWIR = open_wvl_file("C:\ella\FINELCODE\wvl_swir.txt")
WVL_VNIR = open_wvl_file("C:\ella\FINELCODE\wvl_vnir.txt")
WVL_ALL = [i for i in range(400, 2501)]


def find_pixel(lat, long):
    """
    The function checks if a certain coordinate exists
    :param lat: The lat value of the coordinate
    :param long: The long value of the coordinate
    :return: The coordinate in pixel which matches to the given global coordinate
    """
    global rows, cols

    # Iterate over each pixel in image until there is a match
    for x in range(cols):
        for y in range(rows):
            lat_hdr, long_hdr = hdr.read_pixel(y, x)[0], hdr.read_pixel(y, x)[1]
            if float(lat) - RANGE <= lat_hdr <= float(lat) + RANGE and float(long) - RANGE <= long_hdr <= float(
                    long) + RANGE:
                return x, y

    # If there is no match the function returns zero
    return 0, 0


def pixel_converter(lat, long, dir_path):
    """
    The function iterate over a folder of igm files to find the image and pixel of a certain input coordinate
    :param lat: The lat value of the coordinate
    :param long: The long value of the coordinate
    :param dir_path: The folder path of the igm files
    :return:
    """
    global hdr
    global rows, cols

    # Iterate over all files in the igm folder
    for file in os.listdir(dir_path):
        if file.endswith(".hdr"):
            name = file.split(".")[0]
            hdr = envi.open(dir_path + "\\" + file)
            rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
            x, y = find_pixel(lat, long)
            if x != 0 and y != 0:
                return name, x, y

    return "None", 0, 0


def pixel_to_dict(pixel_path):
    """
    The function iterates over a folder of text files with local coordinates from the classifying stage
    :param pixel_path: The path of the local coordinates text files
    :return: A dictionary key - text file name value - material, local coordinate
    """
    dict_pixels = {}
    for file in os.listdir(pixel_path):
        name = file.split(".")[0]
        dict_pixels[name] = []
        with open(pixel_path + "\\" + file, encoding='utf8') as f:
            for line in f:
                pixel = line.split(" ")
                dict_pixels[name].append([pixel[0], [int(pixel[1]), int(pixel[2])]])

    return dict_pixels


def get_coordinates(nz_list, igm_path, pic):
    """
    The function converts local coordinates to global coordinates
    :param dict_pixels: The dictionary of the local coordinates to convert
    :param igm_path: The folder of the igm files
    :return: A dictionary key - text file name value - material, global coordinate
    """
    cords_lst = []
    for file in os.listdir(igm_path):
        if pic.split("_")[1] == file.split("_")[1] and file.endswith(".hdr"):
            cor = envi.open(igm_path + "\\" + file)
            cor_image = np.array(cor.asarray())
            if len(nz_list[0]) == 1:
                cord = cor_image[int(nz_list[1])][int(nz_list[0])]
                cords_lst.append([cord[0], cord[1]])
            else:
                for nz in nz_list:
                    cord = cor_image[int(nz[1])][int(nz[0])]
                    cords_lst.append([cord[0], cord[1]])
    return np.array(cords_lst)


def write_dict_cords(list_cord, el):
    """
    Write the global found coordinates to text and csv files
    :param dict_cord: A dictionary containing the hdr names, materials and global coordinates
    """
    with open(el + ".csv", "a") as f:
        np.savetxt(f, list_cord, delimiter=" ", fmt='%f')


def convert_pixel_to_cord(pixel_path, igm_path):
    """
    The main function of converting and writing the results
    :param pixel_path: The path of the local coordinates text files
    :param igm_path: The folder of the igm files
    """
    for pic in os.listdir(pixel_path):
        print(pic)
        for el in os.listdir(pixel_path + "\\" + pic):
            if el.endswith(".txt"):
                pixels = np.loadtxt(pixel_path + "\\" + pic + "\\" + el, dtype='str')
                print(el)
                dict_cord = get_coordinates(pixels, igm_path, pic)
                write_dict_cords(dict_cord, el)
    sys.exit()


def get_signatures_db(path):
    """
    The function runs over all signatures in db and create a dictionary
    :param path: The path of all the folders of materials containing the different signatures
    :return: A dictionary key - material value - a list of all signatures
    """
    total_dict = {}
    dirs_name = [x[0] for x in os.walk(path)]

    # Iterates over all the folders in the given path
    for i, directory in enumerate(dirs_name):
        if i != 0:
            material = directory.split(path)[1].split("\\")[1]
            total_dict[material] = []
            for file in os.listdir(directory):
                with open(directory + "\\" + file, "r") as f:
                    arr = f.read().split(",")
                    arr = [float(i) for i in arr]
                    total_dict[material].append(arr)

    return total_dict


def get_non_empty_pixels():
    """
    The function saves only the non empty pixels
    :return: A dictionary key - location value - the normalized signature
    """
    global rows, cols
    dict_all = {}
    for x in range(cols):
        for y in range(rows):
            pixel = get_pixel(x, y)
            is_all_zero = np.all((pixel == 0))
            if not is_all_zero:
                pixel = normalize_the_data(pixel)
                dict_all[(x, y)] = pixel

    return dict_all


def get_dict_to_compare(path, s_v):
    """
    The function iterates over all the hdr files after ksvd and saves the data in a dictionary
    :param path: Path of all hdr files after ksvd
    :param s_v: Swir or Vnir
    :return: A dictionary key - name of file value - dictionary of non empty pixels
    """
    final_sigs = {}
    global hdr
    global rows, cols
    # Iterate over all hdr ksvd files in directory
    for i, file in enumerate(os.listdir(path)):
        if (i == 0 or i == 1) and file.endswith(".hdr"):
            name = file.split(".")[0]
            hdr = envi.open(path + "\\" + file)
            wvl_org = hdr.bands.centers
            global wvl
            wvl = get_wvl(wvl_org, s_v)
            rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
            global locations
            locations = get_locations(wvl_org, s_v)
            # Get all non empty pixels - anomaly
            dic_sig = get_non_empty_pixels()
            final_sigs[name] = dic_sig

        elif file.endswith(".hdr"):
            rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
            name = file.split(".")[0]
            hdr = envi.open(path + "\\" + file)
            # Get all non empty pixels - anomaly
            dic_sig = get_non_empty_pixels()
            final_sigs[name] = dic_sig

    return final_sigs


def compare_db_to_path(dict_of_sigs, dict_of_db):
    """
    The function finds the most similar material by angle to each non empty pixel in hdr ksvd
    :param dict_of_sigs: A dictionary key - hdr name value - dictionary key - location value - spectral signature
    :param dict_of_db: A dictionary key - material value - spectral signatures
    :return: A dictionary key - hdr name value - dictionary key - location value - angle, material
    """
    final_dict = {}
    # Iterate over all hdr ksvd
    for key_big, val_big in dict_of_sigs.items():
        compare_dict = {}
        # Iterate over each pixel in a single hdr ksvd and find it's most similar material by angle
        for key, val in val_big.items():
            angle, material = get_closest_angle(val, dict_of_db)
            compare_dict[key] = angle, material

        final_dict[key_big] = compare_dict

    return final_dict


def write_compare_dict_to_file(compare_dict, path_of_hdr):
    """
    Write to each hdr ksvd file a text file containing locations and materials
    :param compare_dict: A dictionary key - hdr name value - dictionary key - location value - angle, material
    :param path_of_hdr: Path of hdr ksvd files
    """
    os.mkdir(path_of_hdr + "_Output")
    for key_big, val_big in compare_dict.items():
        with open(path_of_hdr + "_Output\\" + key_big + ".txt", 'w') as f:
            for key, val in val_big.items():
                f.write('%s:%s\n' % (key, val))


def compare_signatures(path_of_hdr, path_of_db, s_v):
    """
    The main function of matching each pixel to material and writing the results
    :param path_of_hdr: Path of hdr ksvd files
    :param path_of_db: Path of all spectral signatures
    :param s_v: Swir or vnir
    """
    dict_of_sigs = get_dict_to_compare(path_of_hdr, s_v)
    dict_of_db = get_signatures_db(path_of_db)
    compare_dict = compare_db_to_path(dict_of_sigs, dict_of_db)
    write_compare_dict_to_file(compare_dict, path_of_hdr)


def view_slider():
    """
    The function creates a slider object
    """
    global slider_angle
    global rows, cols
    global ax
    # Creating the slider object
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Classification Result")
    plt.connect('button_press_event', on_press)
    ax.imshow(get_image_show(0.5))
    fig.subplots_adjust(left=0.15, bottom=0.2)
    angle_slider_ax = fig.add_axes([0.2, 0.1, 0.65, 0.02])
    angle_slider = Slider(
        ax=angle_slider_ax,
        label='Angle [deg]',
        valmin=0,
        valmax=30,
    )
    brightness_slider_ax = fig.add_axes([0.2, 0.05, 0.65, 0.02])
    brightness_slider = Slider(
        ax=brightness_slider_ax,
        label='Brightness',
        valmin=0,
        valmax=5,
    )
    # Call the update slider function
    angle_slider.on_changed(update_angle)
    brightness_slider.on_changed(update_brightness)
    plt.show()


def update_brightness(val_bright):
    """
    The function updates the image viewed by the brightness value on the slider
    :param val_bright: The slider parameter
    """
    global val_bright_global
    global ax
    val_bright_global = val_bright
    ax.imshow(get_image_show(val_bright))


# Can be changed to no color change on image
def update_angle(val):
    """
    The function updates the image viewed by the angle on the slider
    :param val: The slider parameter
    """
    global slider_angle
    slider_angle = val
    global val_bright_global
    global rows, cols
    global name_img
    global angles

    # Iterate over all the pixels in the hdr entered and their angles
    # Checks if the angle of the pixel is smaller than the slider parameter
    temp = np.where(angles < val, 0, angles)
    temp = temp.reshape((cnt, rows * cols))
    temp = temp.reshape((cnt, rows, cols))
    # check where are all the right pixel
    item_pixel = np.transpose(np.argwhere(temp == 0))
    x = np.flip(item_pixel[1])
    y = np.flip(item_pixel[2])

    global ax
    if len(item_pixel) > 0:
        ax.clear()
        ax.imshow(get_image_show(val_bright_global))
        # putt a dot on the lighted pixel
        ax.scatter(y, x, color='pink', s=10)
    else:
        ax.clear()
        ax.imshow(get_image_show(val_bright_global))


def file_classify(image, sig, s_v):
    """
    The main function of classifying a single hdr file
    :param image: The path of the hdr file
    :param sig: The path of the folder of spectral signatures
    :param s_v: Swir or vnir
    """
    global hdr
    hdr = envi.open(image)
    # Get all bands
    wvl_org = hdr.bands.centers
    global wvl
    # Get good bands only
    wvl = get_wvl(wvl_org, s_v)
    global rows, cols
    rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
    global locations
    # Get location in bands array of good bands
    locations = get_locations(wvl_org, s_v)
    # Get dictionary key - material value - spectral signatures
    sig_dict = get_sig_dict(sig)
    # Classify the hdr
    classify_image_by_angles(sig_dict, image)


def compare_vectors(v, v_to_compare):
    """
    The function returns the angle between 2 vectors
    :param v: The first vector (spectral signature)
    :param v_to_compare: The second vector (spectral signature)
    :return: The angle in degrees
    """
    # Make sure that the 2 vectors have the same length
    if len(v) < len(v_to_compare):
        for i in range(len(v_to_compare) - len(v)):
            v.append(0)

    elif len(v_to_compare) < len(v):
        for i in range(len(v) - len(v_to_compare)):
            v_to_compare.append(0)
    d = np.dot(v, v_to_compare)

    try:
        rad_angle = math.acos(d)
        degrees_angle = rad_angle * (180.0 / math.pi)

    except ValueError:
        degrees_angle = 0

    return degrees_angle


def check_angle(v, v_to_compare):
    """
    The function calls the function which returns the angle between the vectors
    :param v: The first vector (spectral signature)
    :param v_to_compare: The second vector (spectral signature)
    :return: The angle between the vectors
    """
    angle = compare_vectors(v, v_to_compare)

    return angle


def get_closest_angle(pixel, sig_dict):
    """
    The function looks for the closest material by angle for a given spectral signature
    :param pixel: The vector (spectral signature) to classify
    :param sig_dict: A dictionary key- material value - spectral signatures
    :return:
    """
    min_angle = sys.maxsize
    lowest_angle_dict = {}

    # Iterate over sig_dict
    for key, value in sig_dict.items():
        for val in value:
            angle = check_angle(pixel, val)
            if angle <= min_angle:
                min_angle = angle

        # Insert the lowest angle for each material to a dictionary
        lowest_angle_dict[key] = min_angle

    # Get the lowest material and angle
    material = min(lowest_angle_dict, key=lowest_angle_dict.get)

    return lowest_angle_dict[material], material


def get_key(val):
    """
    The function looks for a color in the colors dictionary
    :param val: The color entered
    :return: The key of the color (material)
    """
    global dirs_colors
    for key, value in dirs_colors.items():
        if list(val) == list(value):
            return key


def write_nz(name_img_str):
    """
    The function writes all the pixels found in the algorithm
    :param name_img_str: The name of the image
    """

    global dirs_colors
    global img
    # global lst_final
    lst_final = []
    # Get all non empty pixels from image and their locations
    for x in range(len(img[1])):
        for y in range(len(img)):
            is_all_zero = np.all((img[y][x] == 0))
            if not is_all_zero:
                key = get_key(img[y][x])
                lst_final.append([key, (x, y)])

    # Writes all the locations found to a text file
    # orgenize()
    with open("results_17.5\\" + name_img_str + ".txt", "w") as f:
        for i in range(len(lst_final)):
            if i == len(lst_final) - 1:
                f.write(str(lst_final[i][0]) + " " + str(lst_final[i][1][0]) + " " + str(lst_final[i][1][1]))
            else:
                f.write(str(lst_final[i][0]) + " " + str(lst_final[i][1][0]) + " " + str(lst_final[i][1][1]) + "\n")


# cols than rows
def classify_image_by_angles(image, sig_path, s_v):
    """
    The main function of classifying the hdr files
    :param sig_dict: A dictionary key - material value - spectral signatures
    :param image: The hdr path
    """
    all_pixel = []
    global rows, cols
    global dict_angles
    global name_img
    global hdr
    global angles
    spec = envi.open(image)
    hdr = spec
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
        global cnt
        cnt = 0
        for sig in os.listdir(sig_path):
            cnt = cnt + 1
            pixel = angle_find_file(get_sig(sig_path + "\\" + sig), pixel_array)
            all_pixel.append(pixel)
        all_pixel = np.array(all_pixel)
        angles = np.array(all_pixel)
    # Send to angle slider
    view_slider()


# cols than rows
def classify_image_by_angles_folder(sig_dict, image, angles):
    """
    Same as the "classify_image_by_angle" function without the slider view
    :param sig_dict: A dictionary key - material value - spectral signatures
    :param image: The hdr path
    """
    global rows, cols
    global dict_angles
    global img
    global slider_angle
    global name_img
    dict_angles = {}
    heat_angles = np.empty((rows, cols), dtype=float)
    heat_materials = np.empty((rows, cols), dtype=object)
    xy = list(product(range(cols), range(rows)))
    for dot in tqdm(xy, desc="Classification Progress"):
        pixel = get_pixel(dot[0], dot[1])
        pixel = list(normalize_the_data(pixel))
        angle, material = get_closest_angle(pixel, sig_dict)
        if angle < 6:
            heat_angles[dot[1]][dot[0]] = angle
        else:
            heat_angles[dot[1]][dot[0]] = 6
        heat_materials[dot[1]][dot[0]] = material
        dict_angles[dot] = [angle, material]

    # print(dict_angles)

    # Classify pixel if it's angle is lower than angle entered above
    img = np.zeros((rows, cols, 3))
    for key, val_dict in dict_angles.items():
        if val_dict[0] <= angles[val_dict[1]]:
            img[(key[1], key[0])] = dirs_colors[val_dict[1]]

    im_float = np.float32(img)
    im_rgb = cv.cvtColor(im_float, cv.COLOR_BGR2RGB)
    name_img_str = ""
    for i in os.path.basename(image):
        if i.isdigit():
            name_img_str += i

    name_img = name_img_str
    create_heat_map(heat_angles, heat_materials, name_img_str)
    cv.imwrite("results_17.5\\" + name_img_str + ".png", im_rgb)
    write_nz(name_img_str)


# def orgenize():
#     global lst_final
#     global image_name
#     path = "results_17.5\\" + str(image_name) + "\\"
#     # Writes all the locations found to a text file
#     for alement in lst_final:
#         if os.path.isfile(path + str(alement[0]) + ".txt"):
#             f = open(path + str(alement[0]) + ".txt", 'a')
#             f.write(str(alement) + "\n")
#         else:
#             f = open(path + str(alement[0]) + ".txt", 'a')
#             f.write(str(alement) + "\n")


def write_nz_angle(dicct_angle, name_img_str, angle):
    """
    The function writes all the pixels found in the algorithm
    :param name_img_str: The name of the image
    """
    global file
    # path = os.path.join("results_17.5\\",name_img_str)
    path = "results_17.5\\" + file
    # Writes all the locations found to a text file
    if (len(dicct_angle) > 0):
        with open(path + "\\" + str(angle) + ".txt", "w") as f:
            for i in range(len(dicct_angle)):
                if i == len(dicct_angle) - 1:
                    f.write(str(dicct_angle[i][0]) + " " + str(dicct_angle[i][1][0]) + " " + str(dicct_angle[i][1][1]))
                else:
                    f.write(
                        str(dicct_angle[i][0]) + " " + str(dicct_angle[i][1][0]) + " " + str(
                            dicct_angle[i][1][1]) + "\n")


def create_heat_map(heat_angles, heat_materials, name_img_str):
    """
    The function creates and saves a heat map by angles
    :param heat_angles: A numpy array containing all angles
    :param heat_materials: A numpy array containing all materials
    :param name_img_str: The name of thr image
    """
    fig = plt.figure()
    # Adding annotations
    # formatted_text = (np.asarray(["{0}\n{1:.2f}".format(
    #     text, data) for text, data in zip(heat_angles.flatten(), heat_materials.flatten())])).reshape(cols, rows)
    # fig = sns.heatmap(heat_angles, annot=formatted_text, fmt="", cmap="autumn")

    # Create the heat map
    fig = sns.heatmap(heat_angles, cmap="autumn")
    # Save the heat map
    plt.savefig("results_17.5\\" + name_img_str + "_heat.png")


def create_perfect_signature(list_of_signatures):
    """
    The function converts a list of signatures into one signature
    :param list_of_signatures: A list of spectral signatures
    :return: A summarized single signature
    """
    final_array = [sum(elem) for elem in zip(*list_of_signatures)]
    final_array = [x / len(list_of_signatures) for x in final_array]

    return final_array


def normalize_the_data(signature):
    """
    The function normalize a spectral signature
    :param signature: A not normalized spectral signature
    :return: Normalized spectral signature
    """
    normal = signature / np.linalg.norm(signature)

    return normal


def smooth(signature, box_pts):
    box = np.ones(box_pts) / box_pts
    signature = np.convolve(signature, box, mode='same')

    return signature


def get_perfect_signature(file_name):
    """
    The function calls all necessary functions in order to create a final spectral signatures from file
    :param file_name: A file with spectral signatures of one material
    :return: The final single normalized spectral signature
    """
    return list(normalize_the_data(np.array((create_perfect_signature(list(read_data(file_name)))))))


def create_sigs_from_polygon(sig_path_pol, mode):
    """
    The main function of creating a signature from a polygon
    :param sig_path_pol: Path of files of signatures from polygon
    :param mode: Mode of how to take the points from file
    """
    os.mkdir(sig_path_pol + "\\finalPolSig")
    for file in os.listdir(sig_path_pol):
        if sig_path_pol + "\\" + file != sig_path_pol + "\\finalPolSig":
            if mode == "all":
                points = list(read_data(sig_path_pol + "\\" + file))
                for i, point in enumerate(points):
                    point = list(normalize_the_data(np.array(point)))
                    splited = file.split(".")
                    with open(sig_path_pol + "\\finalPolSig\\" + splited[0] + "_" + str(i) + "." + splited[1],
                              'w') as fp:
                        for j, item in enumerate(point):
                            if j != len(point) - 1:
                                fp.write("%s, " % item)
                            else:
                                fp.write("%s" % item)

            else:
                sig = get_perfect_signature(sig_path_pol + "\\" + file)

                with open(sig_path_pol + "\\finalPolSig\\" + file, "w") as fp:
                    for i, item in enumerate(sig):
                        if i != len(sig) - 1:
                            fp.write("%s, " % item)
                        else:
                            fp.write("%s" % item)


def closest_arg(original, target):
    """
    The function looks for the same wavelengths between 2 wavelengths
    :param original: The original wavelengths
    :param target: The target wavelengths
    :return: A list containing the similar wavelengths
    """
    rounded = [round(num) for num in target]
    return [original[i] for i in range(len(original)) if original[i] in rounded]


def get_locations_of_rf(original, target):
    """
    The function returns the locations of the wanted wavelengths
    :param original: The original wavelengths
    :param target: The target wavelengths
    :return: A list of locations of wanted wavelengths
    """
    return [i for i in range(len(original)) if original[i] in target]


def get_rf_by_locations(loc, rf):
    """
    The function returns the final spectral signature
    :param loc: The locations of the wanted wavelengths
    :param rf: All the data of the spectral signature
    :return: A list of the wanted bands data (spectral signature)
    """
    return [rf[i] for i in loc]


def get_sig_from_db(file_path, swir_or_vnir):
    """
    The function converts an online or spectrometer output signature to a signature with non corrupted bands
    :param file_path: Path of spectral signature file to convert
    :param swir_or_vnir: Swir or vnir
    :return: A list of the new signature
    """
    with open(file_path, "r") as f:
        rf = f.readlines()
        rf = rf[1:]
        for i in range(len(rf)):
            if "-" in rf[i]:
                rf[i] = float(rf[i].split("\n")[0])
            else:
                rf[i] = float(rf[i].split(" ")[1].split("\n")[0])

    if swir_or_vnir == "swir":
        wvl = WVL_SWIR.copy()

    else:
        wvl = WVL_VNIR.copy()

    final_wv = closest_arg(np.array(WVL_ALL), wvl)
    loc = get_locations_of_rf(WVL_ALL, final_wv)
    final_rf = get_rf_by_locations(loc, rf)

    return normalize_the_data(np.array(final_rf))


def create_sigs_from_online_db(sig_path_db, swir_or_vnir):
    """
    The main function of creating spectral signature from online db or spectrometer
    :param sig_path_db: A folder with all spectral signatures to convert
    :param swir_or_vnir: Swir or vnir
    """
    # Creating an output directory
    os.mkdir(sig_path_db + "\\finalDBSig")
    # Iterate over the folder path entered
    for file in os.listdir(sig_path_db):
        if sig_path_db + "\\" + file != sig_path_db + "\\finalDBSig":
            sig = get_sig_from_db(sig_path_db + "\\" + file, swir_or_vnir)

            # Write output signature to new file
            with open(sig_path_db + "\\finalDBSig\\" + file, "w") as fp:
                for i, item in enumerate(sig):
                    if i != len(sig) - 1:
                        fp.write("%s, " % item)
                    else:
                        fp.write("%s" % item)


def get_sig_dict(sig_loc):
    """
    The function creates a dictionary of signatures
    :param sig_loc: Path to folder of signatures
    :return: A dictionary key - material value - spectral signatures
    """
    total_dict = {}
    # Get all directories names
    dirs_name = [x[0] for x in os.walk(sig_loc)]

    global dirs_colors

    # Iterate over all directories
    for i, directory in enumerate(dirs_name):
        if i != 0:
            # Get each material a color
            dirs_colors[os.path.basename(directory)] = COLORS[i]
            material = directory.split(sig_loc)[1].split("\\")[1]
            total_dict[material] = []
            # Iterate over files of signatures in directory
            for file in os.listdir(directory):
                # Append each signature to list of signatures by materials
                with open(directory + "\\" + file, "r") as f:
                    arr = f.read().split(",")
                    arr = [float(i) for i in arr]
                    total_dict[material].append(arr)

    return total_dict


def remove_atmospheric_absorption_bands(bands):
    """
    The function removes all absorption bands from bands list
    :param bands: List of all bands
    :return: A new list without the corrupted bands
    """
    global locations
    return np.delete(bands, locations)


def get_pixel(x, y):
    """
    The function returns a spectral signature by a given x and y location
    :param x: X location on hdr
    :param y: Y location on hdr
    :return: Spectral signature
    """
    global hdr
    pixel = remove_atmospheric_absorption_bands(hdr.read_pixel(y, x))

    return pixel


def file_show(image, s_v):
    """
    The main function of shows a hdr
    :param image: Path of hdr
    :param s_v: Swir or vnir
    """
    global hdr
    hdr = envi.open(image)
    wvl_org = hdr.bands.centers
    global wvl
    wvl = get_wvl(wvl_org, s_v)
    global locations
    locations = get_locations(wvl_org, s_v)
    view_slider_image()


def view_slider_image():
    """
    The function creates a slider object to show an image
    """
    # Creating the slider object
    global ax
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("RGB Image")
    plt.connect('button_press_event', on_press)
    ax.imshow(get_image_show(0.5))
    fig.subplots_adjust(left=0.15, bottom=0.2)
    brightness_slider_ax = fig.add_axes([0.2, 0.1, 0.65, 0.02])
    brightness_slider = Slider(
        ax=brightness_slider_ax,
        label='Brightness',
        valmin=0,
        valmax=5,
    )
    # Call the update slider function
    brightness_slider.on_changed(update_brightness_image)
    plt.show()


def update_brightness_image(val):
    """
    The function updates the image viewed by the brightness value on the slider
    :param val: The slider parameter
    """
    global ax
    ax.imshow(get_image_show(val))


def get_image_show(bright):
    """
    The function returns the rgb image from hdr by 3 bands and brightness value
    :return: The rgb image
    """
    global hdr
    # 119 52 26
    rgb = np.stack([hdr.read_band(114), hdr.read_band(52),
                    hdr.read_band(26)], axis=-1)
    # Can be brighter by changing the number to multiply with
    rgb = rgb / rgb.max() * bright

    return rgb


def file_create(image, list_of_create_data, s_v):
    """
    The main function of creating a material signature by polygon drawing
    :param image: Path of hdr
    :param list_of_create_data: A list [w/a, name of material]
    :param s_v: Swir or vnir
    """
    global hdr
    hdr = envi.open(image)
    wvl_org = hdr.bands.centers
    global wvl
    wvl = get_wvl(wvl_org, s_v)
    global locations
    locations = get_locations(wvl_org, s_v)
    create_data(list_of_create_data)


def create_data(list_of_types):
    """
    The function calls the drawing polygon function and writes all spectral signatures from the selected polygons
    :param list_of_types: A list [w/a, name of material]
    """
    # Iterate over list_of_types
    for obj in range(1, len(list_of_types) + 1):
        # Call drawing multiple_polygons
        points = multiple_polygons()
        global FLG_EXIT
        FLG_EXIT = False

        # Write output points into a text file by name and text mode entered
        with open(list_of_types[obj - 1][1] + ".txt", list_of_types[obj - 1][0]) as f:
            for i, value in enumerate(points.values()):
                value = list(value)
                if i != len(points.values()) - 1:
                    s = ", ".join(map(str, value))
                    f.write(f"%s\n" % s)
                else:
                    s = ", ".join(map(str, value))
                    f.write(f"%s" % s)


def multiple_polygons():
    """
    The function calls the drawing polygon function ands gets all points in the selected polygon
    :return: The final list of all points selected in polygons
    """
    points_in_polygon = {}
    while FLG_EXIT is False:
        # Call drawing polygon function
        points = set_polygon()
        if len(points) > 0:
            # Get all points inside the selected polygon and update the final points_in_polygon dictionary
            polygon = Polygon(np.array(points))
            points_in_polygon.update(get_all_points_inside_polygon(polygon))

    return points_in_polygon


def set_polygon():
    """
    The function allows the user draw a polygon on the hdr entered
    :return: The location of points inside the selected polygon
    """
    global ax
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Polygon Selector")
    selector = PolygonSelector(ax, lambda *args: None)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.tight_layout()
    brightness_slider_ax = fig.add_axes([0.2, 0.01, 0.65, 0.02])
    brightness_slider = Slider(
        ax=brightness_slider_ax,
        label='Brightness',
        valmin=0,
        valmax=5,
    )
    ax.imshow(get_image_show(0.5))
    brightness_slider.on_changed(update_brightness_image)
    plt.show()
    return selector.verts


def get_all_points_inside_polygon(polygon):
    """
    The function returns all the spectral signatures by locations inside polygons
    :param polygon: The selected polygon
    :return: A dictionary key - location value - spectral signature
    """
    points = {}
    path = polygon.get_path()
    # Get polygon extents (square surrounding the polygon)
    xmin, ymin, xmax, ymax = path.get_extents().extents
    for x in range(round(xmin), round(xmax) + 1):
        for y in range(round(ymin), round(ymax) + 1):
            if polygon.contains_point((x, y)):
                global locations
                points[(x, y)] = get_pixel(x, y)

    return points


def get_image():
    """
    The function returns the rgb image from hdr by 3 bands
    :return: The rgb image
    """
    global hdr
    rgb = np.stack([hdr.read_band(114), hdr.read_band(52),
                    hdr.read_band(26)], axis=-1)
    # Can be brighter by changing the number to multiply with
    rgb = rgb / rgb.max() * 1.5

    return rgb


def on_key(event):
    """
    The function changes polygon exit flag when the 'e' key is pressed
    :param event: Event
    """
    if event.key == 'e':
        global FLG_EXIT
        FLG_EXIT = True
        plt.close()


def read_data(file_name):
    """
    The function returns all the spectral signatures in a single text file
    :param file_name: Path of signatures text file
    :return: list of all spectral signatures
    """
    with open(file_name, "r") as f:
        all_lines = f.readlines()
        points = [(line.strip()).split(", ") for line in all_lines]

    points = [list(map(float, sublist)) for sublist in points]

    return points


def plot_vectors(path, s_v_a):
    """
    The function plots spectral signatures on graph
    :param path: File or folder path containing spectral signatures
    :param s_v_a: Swir or vnir or all
    """
    list_of_vectors = []
    if s_v_a == "swir":
        wvl_plot = WVL_SWIR.copy()

    elif s_v_a == "vnir":
        wvl_plot = WVL_VNIR.copy()

    else:
        wvl_plot = WVL_ALL.copy()

    # Insert spectral signature from file to list
    if os.path.isfile(path):
        with open(path, "r") as f:
            arr = f.read().split(",")
            list_of_vectors.append([float(i) for i in arr])
    else:
        # Insert all spectral signatures in folder to list of lists
        for root, dirs, files in os.walk(path):
            for file in files:
                with open(os.path.join(root, file), "r") as f:
                    arr = f.read().split(",")
                    list_of_vectors.append([float(i) for i in arr])

    fig = plt.figure()
    fig.canvas.manager.set_window_title("Spectral Signature")
    max_len = len(max(list_of_vectors, key=len))
    final_lst = []
    for lst in list_of_vectors:
        if len(lst) != max_len:
            for _ in range(max_len - len(lst)):
                lst.append(0)
        final_lst.append(lst)
    if len(wvl_plot) != len(final_lst[0]):
        for i in range(len(final_lst[0]) - len(wvl_plot)):
            wvl_plot.append(wvl_plot[len(wvl_plot) - 1] + (i + 1))
    # Draw all signatures on graph
    for vector in final_lst:
        plt.plot(wvl_plot, vector)
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')

    plt.show()


def read_angle_file(path):
    """
    The function creates angles and materials dictionary
    :param path: Path of file with materials and angles
    :return: A dictionary key - material value - angle
    """

    angle_dictionary = {}
    with open(path, "r") as f:
        lines = f.read()
        lines = lines.split("\n")
        for line in lines:
            l = line.split(" ")
            angle_dictionary[l[0]] = float(l[1])

    return angle_dictionary


def folder_classify(image, sig, s_v, angle):
    """
    The main function of classifying a folder with hdr
    :param image: Path of folder with hdrs
    :param sig: Folder of spectral signatures
    :param s_v: Swir or vnir
    :param angle: Path of file with materials and angles
    """
    global hdr
    global rows, cols
    angles = read_angle_file(angle)
    for i, file in enumerate(os.listdir(image)):
        if (i == 0 or i == 1) and file.endswith(".hdr"):
            optim(image + "\\" + file, sig, angles, s_v)
        elif file.endswith(".hdr"):
            # Classify the hdr
            optim(image + "\\" + file, sig, angles, s_v)
    sys.exit()


def get_wvl(wvl_org, s_v):
    """
    The function returns only not corrupted wavelengths
    :param wvl_org: Original wavelengths values
    :param s_v: Swir or vnir
    :return: The non corrupted wavelengths values
    """
    if s_v == "swir":
        list_to_delete = THE_GOOD_LIST_SWIR
    else:
        list_to_delete = THE_GOOD_LIST_VNIR

    global wvl
    # Check if a value from the original wavelengths list is in the "good list"
    wvl = [x for x in wvl_org if
           list_to_delete[0][0] <= round(x) <= list_to_delete[0][1] or list_to_delete[1][0] <= round(x) <=
           list_to_delete[1][1] or list_to_delete[2][0] <= round(x) <= list_to_delete[2][1]]

    return wvl


def on_press(event):
    """
    The function plots on right click press the spectral signature of the pressed pixel
    :param event: Event
    """
    # Check if right click in mouse is pressed
    if event.button is MouseButton.RIGHT:
        # Draw the graph
        graph = plt.figure(figsize=(5, 5))
        graph.canvas.manager.set_window_title("Spectral Signature")
        plt.title("x = " + str(int(event.xdata)) + ", y = " + str(int(event.ydata)))
        pixel = get_pixel(int(event.xdata), int(event.ydata))
        pixel = normalize_the_data(pixel)
        global wvl
        plt.plot(wvl, pixel)
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        graph.show()


def show_img():
    """
    The function shows a hdr - not in use currently
    """
    # Get a rgb image
    rgb = get_image()

    # Plot the image
    figure = plt.figure()
    figure.canvas.manager.set_window_title("RGB Image")
    plt.connect('button_press_event', on_press)
    plt.imshow(rgb)
    plt.show()


def get_locations(wvl_org, s_v):
    """
    The function returns all the locations of the non corrupted bands
    :param wvl_org: Original wavelengths
    :param s_v: Swir or vnir
    :return: A list of the non corrupted locations
    """
    if s_v == "swir":
        list_to_delete = THE_GOOD_LIST_SWIR
    else:
        list_to_delete = THE_GOOD_LIST_VNIR

    global locations
    # Check if a value from the original wavelengths list is in the "good list" and get it's location
    for loc in range(len(wvl_org)):
        if list_to_delete[0][0] <= round(wvl_org[loc]) <= list_to_delete[0][1] or list_to_delete[1][0] <= round(
                wvl_org[loc]) <= list_to_delete[1][1] or list_to_delete[2][0] <= round(wvl_org[loc]) <= \
                list_to_delete[2][1]:
            continue
        else:
            locations.append(loc)

    return locations


# TODO: Need to be checked
def change_len(sig_dict):
    new_dict = {}
    longest_list = len(max(sig_dict.values(), key=len))
    for key, value in sig_dict.items():
        vals = []
        for val in value:
            if len(val) != longest_list:
                for _ in range(longest_list - len(val)):
                    val.append(0)
                vals.append(val)
            else:
                vals.append(val)

        new_dict[key] = vals

    return new_dict


def get_tags(sig_dict):
    tags = []
    for i, (key, value) in enumerate(sig_dict.items()):
        for _ in value:
            tags.append(i)

    return np.array(tags)
