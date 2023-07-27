import math
from scipy.signal import correlate
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
from matplotlib.widgets import PolygonSelector
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Polygon
import pickle
from scipy.signal import find_peaks
from matplotlib.widgets import Slider

dict_angles = {}
# HDR_PATH = "F:\\NISOY\\elykim_29_03_23\\DATA\\2\\swir\\process\\raw_8208_rd_rf.hdr"
# IMAGE_PATH = "F:\\NISOY\\elykim_29_03_23\\DATA\\2\\swir\\process\\raw_8208_rd_rf"
HDR_PATH = "F:\\NISOY\\elykim_06_02_23\\2\\swir\\new\\raw_0_rdk_rd_rf.hdr"
IMAGE_PATH = "F:\\NISOY\\elykim_06_02_23\\2\\swir\\new\\raw_0_rdk_rd_rf"
HDR = envi.open(HDR_PATH, IMAGE_PATH)
# HDR_LOAD = HDR.load()
wvl = HDR.bands.centers
ROWS, COLS, BANDS = HDR.nrows, HDR.ncols, HDR.nbands
FLG_EXIT = False
SIG = [0.0671333, 0.0656209, 0.0675903, 0.0749856, 0.0739124, 0.0743378, 0.0735712, 0.0722691, 0.0698626, 0.0672761,
       0.0673151, 0.0661039, 0.0649313, 0.0644676, 0.0643208, 0.064355, 0.063927, 0.0638993, 0.0636286, 0.0635837,
       0.0632653, 0.0632858, 0.063188, 0.0632282, 0.0633507, 0.063391, 0.0633566, 0.0633249, 0.0635548, 0.0639207,
       0.0646545, 0.0663289, 0.0675355, 0.0707764, 0.0777706, 0.081141, 0.0778523, 0.0748463, 0.0765254, 0.0763524,
       0.0730458, 0.0699901, 0.0685159, 0.0680265, 0.0681061, 0.0685354, 0.0680648, 0.0685097, 0.0684059, 0.0681271,
       0.0679329, 0.0676514, 0.0672223, 0.0667138, 0.0664228, 0.0665255, 0.0667285, 0.0673277, 0.0682948, 0.0683242,
       0.0671226, 0.0668281, 0.066604, 0.0672118, 0.0678399, 0.0690082, 0.0702792, 0.0714775, 0.0725991, 0.0762245,
       0.0772818, 0.0785533, 0.0864714, 0.1029616, 0.1012082, 0.0991311, 0.0990865, 0.1014755, 0.1001617, 0.0997886,
       0.0980745, 0.0945215, 0.0921133, 0.0910171, 0.0892119, 0.0881725, 0.0869545, 0.0858077, 0.0847803, 0.0839781,
       0.0830872, 0.0821455, 0.0814169, 0.0805644, 0.0800098, 0.0781238, 0.0789846, 0.0781697, 0.0775057, 0.0763535,
       0.0756451, 0.0748031, 0.0741239, 0.072993, 0.0714209, 0.069308, 0.0672171, 0.0656044, 0.0644053, 0.0641113,
       0.064152, 0.0625824, 0.061989, 0.0620084, 0.0625905, 0.062881, 0.0638476, 0.0646101, 0.0642777, 0.0640113,
       0.0643012, 0.0649461, 0.0659247, 0.0665013, 0.067015, 0.0673741, 0.06777, 0.0692436, 0.0705989, 0.0719948,
       0.0725185, 0.0741385, 0.0772607, 0.0798134, 0.0916111, 0.0897871, 0.0871526, 0.0862531, 0.0851938, 0.0871679,
       0.0954778, 0.0970935, 0.0904781, 0.0837292, 0.079586, 0.0775363, 0.076276, 0.0756953, 0.0767674, 0.0756207,
       0.0742227, 0.073123, 0.071624, 0.0688954, 0.0683029, 0.0669944, 0.0650736, 0.064978, 0.0639046, 0.0629045,
       0.0613598, 0.0606216, 0.0592429, 0.0556743, 0.0541148, 0.0556946, 0.0573179, 0.0551673, 0.0528469, 0.0539609,
       0.0558997, 0.0548438, 0.0538022, 0.053484, 0.0539185, 0.0530279, 0.0532421, 0.0531264, 0.0529571, 0.0511488,
       0.0500494, 0.0476515, 0.0477028, 0.0467411, 0.0449732, 0.0441599, 0.0441172, 0.0438423, 0.0453956, 0.0443437,
       0.0444645, 0.0443496, 0.0436078, 0.0458906, 0.0444947, 0.0469991, 0.045009, 0.0457618, 0.0450346, 0.0465116,
       0.040992, 0.0479089, 0.0441993, 0.0499964, 0.0480346, 0.0516228, 0.0525556]

FABRIC_PEAKS = {7: 934,
                10: 952,
                17: 994,
                39: 1125,
                40: 1131,
                43: 1149,
                96: 1466,
                129: 1664,
                208: 2137,
                219: 2203,
                228: 2256}

COLORS = {0: (255, 0, 0),
          1: (0, 255, 0),
          2: (0, 0, 255),
          3: (255, 255, 0),
          4: (0, 255, 255)}

THE_BLACK_LIST = [[914, 1347], [1443, 1807], [1961, 2400]]

WVL = [x for x in wvl if
       THE_BLACK_LIST[0][0] <= round(x) <= THE_BLACK_LIST[0][1] or THE_BLACK_LIST[1][0] <= round(x) <=
       THE_BLACK_LIST[1][1] or THE_BLACK_LIST[2][0] <= round(x) <= THE_BLACK_LIST[2][1]]


locations = []

for loc in range(len(wvl)):
    if THE_BLACK_LIST[0][0] <= round(wvl[loc]) <= THE_BLACK_LIST[0][1] or THE_BLACK_LIST[1][0] <= round(
            wvl[loc]) <= THE_BLACK_LIST[1][1] or THE_BLACK_LIST[2][0] <= round(wvl[loc]) <= THE_BLACK_LIST[2][1]:
        continue
    else:
        locations.append(loc)


def on_key(event):
    if event.key == 'e':
        global FLG_EXIT
        FLG_EXIT = True
        plt.close()


def on_press(event):
    if event.button is MouseButton.RIGHT:
        graph = plt.figure(figsize=(5, 5))
        graph.canvas.manager.set_window_title("Spectral Signature")
        plt.title("x = " + str(int(event.xdata)) + ", y = " + str(int(event.ydata)))
        pixel = get_pixel(int(event.xdata), int(event.ydata))
        plt.plot(WVL, pixel)
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        graph.show()


def remove_atmospheric_absorption_bands(bands):
    return np.delete(bands, locations)


def data(x1, y1, x2, y2):
    arr = []
    for row in range(x1, x2):
        for col in range(y1, y2):
            arr.append(get_pixel(col, row))

    arr = np.array(arr)
    np.random.shuffle(arr)

    return arr


def tags_1(times):
    tags_arr = []
    for i in range(times):
        tags_arr.append(1)

    return np.array(tags_arr)


def tags_0(times):
    tags_arr = []
    for i in range(times):
        tags_arr.append(0)

    return np.array(tags_arr)


def get_pixel(x, y):
    pixel = remove_atmospheric_absorption_bands(HDR.read_pixel(y, x))

    return pixel


def compare_data(x1, y1, x2, y2):
    data_to_compare = {}
    for row in range(x1, x2):
        for col in range(y1, y2):
            data_to_compare[row, col] = get_pixel(col, row)

    return data_to_compare


def get_wavelengths():
    return WVL


def show_img():
    rgb = get_image()

    figure = plt.figure()
    figure.canvas.manager.set_window_title("RGB Image")
    plt.connect('button_press_event', on_press)
    plt.imshow(rgb)
    plt.show()

    return rgb


def show_bands(list_of_bands):
    if len(list_of_bands) == 1:
        rgb = np.array(HDR.read_band(list_of_bands[0]))
    else:
        rgb = np.stack([HDR.read_band(list_of_bands[0]), HDR.read_band(list_of_bands[1]),
                        HDR.read_band(list_of_bands[2])], axis=-1)
    rgb = rgb / rgb.max() * 1.5

    figure = plt.figure()
    figure.canvas.manager.set_window_title("RGB Image")
    plt.connect('button_press_event', on_press)
    plt.imshow(rgb)
    plt.show()


def view_pixel_graph(x, y):
    pixel = get_pixel(x, y)
    graph = plt.figure(figsize=(5, 5))
    graph.canvas.manager.set_window_title("Spectral Signature")
    plt.title("x = " + str(int(x)) + ", y = " + str(int(y)))
    plt.plot(WVL, pixel)
    plt.xlabel('Wavelength')
    plt.ylabel('Reflectance')
    plt.show()


def view_graphs_of_image():
    graph = plt.figure()
    graph.canvas.manager.set_window_title("Spectral Signature")
    for x in range(ROWS):
        for y in range(COLS):
            pixel = get_pixel(x, y)
            plt.title("x = " + str(int(x)) + ", y = " + str(int(y)))
            plt.plot(WVL, pixel)
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            plt.show()


def get_hdr():
    return HDR


# def get_hdr_load():
#     return HDR_LOAD


def insert_label(arr, num, x1, y1, x2, y2):
    for i in range(y1, y2):
        for j in range(x1, x2):
            arr[i][j] = num

    return arr


def get_bands():
    bands = []
    for x in range(ROWS):
        for y in range(COLS):
            bands.append(get_pixel(x, y))

    return np.array(bands)


def get_image():
    rgb = np.stack([HDR.read_band(119), HDR.read_band(52),
                    HDR.read_band(26)], axis=-1)
    rgb = rgb / rgb.max() * 1.5

    return rgb


def set_polygon():
    rgb = get_image()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Polygon Selector")
    selector = PolygonSelector(ax, lambda *args: None)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.tight_layout()
    ax.imshow(rgb)
    plt.show()
    return selector.verts


def view_polygon(list_of_points):
    rgb = get_image()
    points = np.array(list_of_points)
    polygon = Polygon(points)
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Polygon")
    ax.add_patch(polygon)
    ax.imshow(rgb)
    plt.show()


def get_all_points_inside_polygon(polygon):
    points = {}
    path = polygon.get_path()
    xmin, ymin, xmax, ymax = path.get_extents().extents
    for x in range(round(xmin), round(xmax) + 1):
        for y in range(round(ymin), round(ymax) + 1):
            if polygon.contains_point((x, y)):
                points[(x, y)] = get_pixel(x, y)

    return points


def multiple_polygons():
    points_in_polygon = {}
    while FLG_EXIT is False:
        points = set_polygon()
        if len(points) > 0:
            polygon = Polygon(np.array(points))
            points_in_polygon.update(get_all_points_inside_polygon(polygon))

    return points_in_polygon


def classification_array(list_of_locations):
    new_data = np.zeros((ROWS, COLS, 3))
    for i, l in enumerate(list_of_locations):
        for location in l:
            new_data[location[0], location[1]] = COLORS[i]

    return new_data


def create_data(list_of_types):
    list_of_locations = []
    for obj in range(1, len(list_of_types) + 1):
        points = multiple_polygons()
        list_of_locations.append(points.keys())
        global FLG_EXIT
        FLG_EXIT = False

        with open(list_of_types[obj - 1][1] + ".txt", list_of_types[obj - 1][0]) as f:
            pickle.dump(points, f)
            # for key, value in points.items():
            #     f.write('%s:%s\n' % (key, value))


def read_data(list_of_files_names):
    points = {}
    for files_names in list_of_files_names:
        with open(files_names, "rb") as f:
            try:
                while True:
                    points.update(pickle.load(f))

            except EOFError:
                pass

    return points


def compare_vectors(v, v_to_compare):
    d = np.dot(v_to_compare, v)
    try:
        rad_angle = math.acos(d)
        degrees_angle = rad_angle * (180.0 / math.pi)

    except ValueError:
        degrees_angle = 100

    return degrees_angle


def check_angle(v, v_to_compare):
    angle = compare_vectors(v, v_to_compare)

    return angle


def classify_image_by_angles():
    for x in range(COLS):
        for y in range(ROWS):
            pixel = get_pixel(x, y)
            pixel = normalize_the_data(pixel)
            angle = check_angle(pixel, SIG)
            global dict_angles
            dict_angles[(x, y)] = angle

    return dict_angles


# def cor_func(x, y):
#     mean_x = np.mean(x)
#     mean_y = np.mean(y)
#     std_x = np.std(x)
#     std_y = np.std(y)
#     n = len(x)
#     # For numpy function use this line instead of the line below
#     # return np.correlate(x - mean_x, y - mean_y, mode='valid')[0] / n / (std_x * std_y)
#     return correlate(x - mean_x, y - mean_y, mode='valid')[0] / n / (std_x * std_y)


# def compare_curves(v, v_to_compare):
#     curve_1 = v_to_compare / v_to_compare.sum()
#
#     if v.sum() == 0:
#         match_score = 0
#     else:
#         curve_2 = v / v.sum()
#
#         # match_score = spearmanr(curve_1, curve_2)[0]
#         # match_score = pearsonr(curve_1, curve_2)[0]
#         # match_score = correlate(curve_1, curve_2, mode='same') / (np.std(curve_1) * np.std(curve_2) * len(curve_1))
#         # match_score = correlate(curve_1, curve_2, mode='full')
#         # match_score = np.corrcoef(curve_1, curve_2)[0, 1]
#         match_score = cor_func(v, v_to_compare)
#
#     return match_score


# def compare_scores(v, v_to_compare):
#     is_similar = False
#     score = compare_curves(v, v_to_compare)
#     if score * 100 >= 85:
#         is_similar = True
#
#     return is_similar


# def classify_image_by_match_score():
#     classified_image = np.zeros((ROWS, COLS))
#     places = []
#     for x in range(COLS):
#         for y in range(ROWS):
#             pixel = get_pixel(x, y)
#             pixel = normalize_the_data(pixel)
#             is_similar = compare_scores(np.array(pixel), np.array(SIG))
#             if is_similar:
#                 classified_image[y][x] = 255
#                 places.append((x, y))
#
#     return classified_image, places


def create_perfect_signature(list_of_signatures):
    final_array = [sum(elem) for elem in zip(*list_of_signatures)]
    final_array = [float("{:.7f}".format(x / len(list_of_signatures))) for x in final_array]

    return final_array


def plot_vectors(list_of_vectors):
    for vector in list_of_vectors:
        plt.plot(WVL, vector)
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')

    plt.show()


def normalize_the_data(signature):
    normal = signature / np.linalg.norm(signature)

    return [float("{:.7f}".format(x)) for x in normal]


def get_perfect_signature(file_name):
    return list(normalize_the_data(np.array((create_perfect_signature(list(read_data([file_name]).values()))))))


# def peaks(signature):
#     return find_peaks(signature)
#
#
# def compare_peaks(peak1, peak2):
#     percentage = cor_func(peak1, peak2)
#     print(percentage)
#     if percentage * 100 > 85:
#         return True
#     return False


def closest_arg(original, target):
    rounded = [round(num) for num in target]
    return [original[i] for i in range(len(original)) if original[i] in rounded]


def get_locations_of_rf(original, target):
    return [i for i in range(len(original)) if original[i] in target]


def get_rf_by_locations(loc, rf):
    return [rf[i] for i in loc]


def get_sig_from_db(wv_file, rf_file):
    with open(wv_file, "r") as f:
        wv = f.readlines()
        wv = [round(float(wv[i].split(" ")[1].split("\n")[0]) * 1000) for i in range(len(wv)) if i != 0]

    with open(rf_file, "r") as f:
        rf = f.readlines()
        rf = rf[1:]
        for i in range(len(rf)):
            if "-" in rf[i]:
                rf[i] = float(rf[i].split("\n")[0])
            else:
                rf[i] = float(rf[i].split(" ")[1].split("\n")[0])
        # rf = [float(rf[i].split(" ")[1].split("\n")[0]) for i in range(len(rf)) if i != 0]

    final_wv = closest_arg(np.array(wv), WVL)
    loc = get_locations_of_rf(wv, final_wv)
    final_rf = get_rf_by_locations(loc, rf)

    return normalize_the_data(np.array(final_rf))


def view_slider():
    img = np.zeros((ROWS, COLS))
    global ax
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(get_image())
    fig.subplots_adjust(left=0.15, bottom=0.2)
    axfreq = fig.add_axes([0.2, 0.1, 0.65, 0.03])
    angle_slider = Slider(
        ax=axfreq,
        label='Angle [deg]',
        valmin=0,
        valmax=30,
    )
    angle_slider.on_changed(update_angle)
    plt.show()


def update_angle(val):
    places = []
    img = np.zeros((ROWS, COLS))
    global dict_angles
    for key, val_dict in dict_angles.items():
        if val_dict <= val:
            img[(key[1], key[0])] = 255
            places.append(key)

    global ax
    if len(places) > 0:
        ax[0].clear()
        ax[0].imshow(img)
        ax[1].clear()
        ax[1].imshow(get_image())
        ax[1].scatter(*zip(*places))
    else:
        ax[0].clear()
        ax[0].imshow(img)
        ax[1].clear()
        ax[1].imshow(get_image())
