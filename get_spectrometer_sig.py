import csv
import os
import numpy as np

final_lst = []


def open_wvl_file(file_name):
    with open(file_name, "r") as f:
        arr = f.read().split(",")

    return [float(i) for i in arr]


WVL_SWIR = open_wvl_file("C:\ella\FINELCODE\wvl_swir.txt")
WVL_VNIR = open_wvl_file("C:\ella\FINELCODE\wvl_vnir.txt")


def get_locations_of_rf(original, target):
    return [i for i in range(len(original)) if original[i] in target]


def closest_arg(original, target):
    rounded = [round(num) for num in target]
    return [original[i] for i in range(len(original)) if original[i] in rounded]


def get_rf_by_locations(loc, rf):
    return [rf[i] for i in loc]


def normalize_the_data(signature):
    normal = signature / np.linalg.norm(signature)

    return [float("{:.7f}".format(x)) for x in normal]


def get_sig_from_db(wvl_of_db, s_v):
    if s_v == "swir":
        wvl = WVL_SWIR
    else:
        wvl = WVL_VNIR

    sig_lst = []
    for lst in final_lst:
        final_wv = closest_arg(np.array(wvl_of_db), np.array(wvl))
        loc = get_locations_of_rf(wvl_of_db, final_wv)
        final_rf = get_rf_by_locations(loc, lst)

        sig_lst.append(normalize_the_data(np.array(final_rf)))

    return sig_lst


def extract(lst, position):
    lst = list(list(zip(*lst))[position])
    return [float(i) for i in lst]


def main():
    file_path = input("Please enter file path ")
    db_type = input("Please enter db type path ")
    s_v = input("Swir or Vnir ")

    wvl_of_db = open_wvl_file(db_type)

    with open(file_path) as f:
        all_lst = []
        reader = csv.reader(f)
        cnt = 0
        for row in reader:
            all_lst.append(row)
            cnt += 1

        for i in range(len(all_lst[0])):
            final_lst.append(extract(all_lst, i))

    sig_lst = get_sig_from_db(wvl_of_db, s_v)
    name_dir = "db"
    os.mkdir(name_dir)
    for i in range(len(sig_lst)):
        with open(name_dir + "\\" + str(i) + ".txt", 'w') as fp:
            for j, item in enumerate(sig_lst[i]):
                if j != len(sig_lst[i]) - 1:
                    fp.write("%s, " % item)
                else:
                    fp.write("%s" % item)


if __name__ == "__main__":
    main()
