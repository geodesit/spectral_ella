import tkinter as tk
from tkinter import *
from tkinter import messagebox
from important_func import *
from ksvd import anomaly_detection

list_of_create_data = []
type_entries = []
name_entries = []

TYPES = ['w', 'a']
MODES = ['all', 'average']


def run_find():
    global lat
    global long
    global igm_find_path

    lat_val = lat.get()
    long_val = long.get()
    find_dir_path = igm_find_path.get()

    if check_folder(find_dir_path):
        img, x, y = pixel_converter(lat_val, long_val, find_dir_path)
        messagebox.showinfo(title="Pixel Info",
                            message="The image number is " + img + " The pixel number is " + str(x) + " " + str(y))

    else:
        messagebox.showerror(title="Error", message="You need to insert a folder path")


def view_find_pixel():
    global find_wid
    find_wid = Toplevel(root)
    find_wid.wm_iconbitmap("drone.ico")
    find_wid.title("Find Pixel")
    find_wid.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter lat:")
    label_set = Label(find_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5, sticky=W)

    global lat
    lat = StringVar(None)
    lat = Entry(find_wid, textvariable=lat, width=50)
    lat.grid(row=0, column=1, padx=5, sticky=W)

    label_text = StringVar()
    label_text.set("Enter long:")
    label_set = Label(find_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=1, column=0, padx=5, sticky=W)

    global long
    long = StringVar(None)
    long = Entry(find_wid, textvariable=long, width=50)
    long.grid(row=1, column=1, padx=5, sticky=W)

    label_text = StringVar()
    label_text.set("Enter path of igm:")
    label_set = Label(find_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=2, column=0, padx=5, sticky=W)

    global igm_find_path
    igm_find_path = StringVar(None)
    igm_find_path = Entry(find_wid, textvariable=igm_find_path, width=50)
    igm_find_path.grid(row=2, column=1, padx=5, sticky=W)

    enter_btn = tk.Button(find_wid, bg="seashell3", text="Enter", command=run_find, width=5, height=1)
    enter_btn.grid(row=3, column=0, padx=20, pady=5, sticky=W)


def run_convert():
    global igm_path
    global pixel_path

    igm = igm_path.get()
    pixel = pixel_path.get()

    if os.path.isdir(pixel) and os.path.isdir(igm):
        convert_pixel_to_cord(pixel, igm)

    else:
        messagebox.showerror(title="Error", message="You need to insert a folder path")


def view_pixel_to_cord():
    global pixel_wid
    pixel_wid = Toplevel(root)
    pixel_wid.wm_iconbitmap("drone.ico")
    pixel_wid.title("Pixel To Coordinates")
    pixel_wid.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter igm folder path:")
    label_set = Label(pixel_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5, sticky=W)

    global igm_path
    igm_path = StringVar(None)
    igm_path = Entry(pixel_wid, textvariable=igm_path, width=50)
    igm_path.grid(row=0, column=1, padx=5, sticky=W)

    label_text = StringVar()
    label_text.set("Enter pixels folder path:")
    label_set = Label(pixel_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=1, column=0, padx=5, sticky=W)

    global pixel_path
    pixel_path = StringVar(None)
    pixel_path = Entry(pixel_wid, textvariable=pixel_path, width=50)
    pixel_path.grid(row=1, column=1, padx=5, sticky=W)

    enter_btn = tk.Button(pixel_wid, bg="seashell3", text="Enter", command=run_convert, width=5, height=1)
    enter_btn.grid(row=2, column=0, padx=20, pady=5, sticky=W)


def run_ksvd():
    global ksvd_path
    path = ksvd_path.get()
    global s_v
    swir_or_vnir = s_v

    if check_folder(path):
        anomaly_detection(path, swir_or_vnir)

    else:
        messagebox.showerror(title="Error", message="You need to insert a folder path")


def view_ksvd():
    global ksvd_wid
    ksvd_wid = Toplevel(root)
    ksvd_wid.wm_iconbitmap("drone.ico")
    ksvd_wid.title("Ksvd")
    ksvd_wid.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter signatures of flight path:")
    label_set = Label(ksvd_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5)

    global ksvd_path
    ksvd_path = StringVar(None)
    ksvd_path = Entry(ksvd_wid, textvariable=ksvd_path, width=50)
    ksvd_path.grid(row=0, column=1, padx=5)

    enter_btn = tk.Button(ksvd_wid, bg="seashell3", text="Enter", command=run_ksvd, width=5, height=1)
    enter_btn.grid(row=0, column=2, padx=20, pady=5, sticky=W)


def compare_sigs():
    global sig_compare_path
    compare_path = sig_compare_path.get()
    global sig_db_compare_path
    db_compare_path = sig_db_compare_path.get()
    global s_v
    swir_or_vnir = s_v

    if os.path.isdir(compare_path) and os.path.isdir(db_compare_path):
        compare_signatures(compare_path, db_compare_path, swir_or_vnir)

    else:
        messagebox.showerror(title="Error", message="You need to insert a folder path")


def view_sigs_compare():
    global sigs_compare_wid
    sigs_compare_wid = Toplevel(root)
    sigs_compare_wid.wm_iconbitmap("drone.ico")
    sigs_compare_wid.title("Compare Signatures")
    sigs_compare_wid.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter anomaly result path:")
    label_set = Label(sigs_compare_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5)

    global sig_compare_path
    sig_compare_path = StringVar(None)
    sig_compare_path = Entry(sigs_compare_wid, textvariable=sig_compare_path, width=50)
    sig_compare_path.grid(row=0, column=1, padx=5)

    label_text = StringVar()
    label_text.set("Enter folder of signatures:")
    label_set = Label(sigs_compare_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=1, column=0, padx=5, sticky=W)

    global sig_db_compare_path
    sig_db_compare_path = StringVar(None)
    sig_db_compare_path = Entry(sigs_compare_wid, textvariable=sig_db_compare_path, width=50)
    sig_db_compare_path.grid(row=1, column=1, padx=5)

    enter_btn = tk.Button(sigs_compare_wid, bg="seashell3", text="Enter", command=compare_sigs, width=5, height=1)
    enter_btn.grid(row=2, column=0, padx=20, pady=5, sticky=W)


def create_sig_pol():
    global sig_pol_mode
    global sig_pol_path
    sig_path_pol = sig_pol_path.get()
    sig_mode_pol = sig_pol_mode.get()

    if sig_mode_pol != MODES[0] and sig_mode_pol != MODES[1]:
        messagebox.showerror(title="Error", message="You need to insert all/average")

    elif not os.path.isdir(sig_path_pol):
        messagebox.showerror(title="Error", message="You need to insert a folder path")

    else:
        create_sigs_from_polygon(sig_path_pol, sig_mode_pol)


def view_sigs_pol():
    global sigs_pol_wid
    sigs_pol_wid = Toplevel(root)
    sigs_pol_wid.wm_iconbitmap("drone.ico")
    sigs_pol_wid.title("Create DB")
    sigs_pol_wid.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter signatures polygon path:")
    label_set = Label(sigs_pol_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5)

    global sig_pol_path
    sig_pol_path = StringVar(None)
    sig_pol_path = Entry(sigs_pol_wid, textvariable=sig_pol_path, width=50)
    sig_pol_path.grid(row=0, column=1, padx=5)

    label_text = StringVar()
    label_text.set("Enter signatures polygon mode:")
    label_set = Label(sigs_pol_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=1, column=0, padx=5)

    global sig_pol_mode
    sig_pol_mode = StringVar(None)
    sig_pol_mode = Entry(sigs_pol_wid, textvariable=sig_pol_mode, width=50)
    sig_pol_mode.grid(row=1, column=1, padx=5)

    enter_btn = tk.Button(sigs_pol_wid, bg="seashell3", text="Enter", command=create_sig_pol, width=5, height=1)
    enter_btn.grid(row=2, column=0, padx=20, pady=5, sticky=W)


def create_sig_db():
    global sig_db_path
    sig_path_db = sig_db_path.get()
    global s_v
    swir_or_vnir = s_v

    if os.path.isdir(sig_path_db):
        create_sigs_from_online_db(sig_path_db, swir_or_vnir)

    else:
        messagebox.showerror(title="Error", message="You need to insert a folder path")


def view_sigs_db():
    global sigs_db_wid
    sigs_db_wid = Toplevel(root)
    sigs_db_wid.wm_iconbitmap("drone.ico")
    sigs_db_wid.title("Create DB")
    sigs_db_wid.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter signatures db path:")
    label_set = Label(sigs_db_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5)

    global sig_db_path
    sig_db_path = StringVar(None)
    sig_db_path = Entry(sigs_db_wid, textvariable=sig_db_path, width=50)
    sig_db_path.grid(row=0, column=1, padx=5)

    enter_btn = tk.Button(sigs_db_wid, bg="seashell3", text="Enter", command=create_sig_db, width=5, height=1)
    enter_btn.grid(row=0, column=2, padx=20, pady=5, sticky=W)


def get_s_or_v():
    global s_v
    swir_or_vnir = s_v
    if swir_or_vnir == "swir" or swir_or_vnir == "vnir" or swir_or_vnir == "all":
        view_main()

    else:
        messagebox.showerror(title="Error", message="You need to write swir/vnir/all")


def get_view_image_path():
    global view_image_path
    path = view_image_path.get()
    global s_v
    swir_or_vnir = s_v

    if os.path.isfile(path) and path.endswith(".hdr"):
        file_show(path, swir_or_vnir)
    else:
        messagebox.showerror(title="Error", message="You need to insert a file path")


def get_image_path():
    global image_path_wid
    image_path_wid = Toplevel(root)
    image_path_wid.wm_iconbitmap("drone.ico")
    image_path_wid.title("View Image")
    image_path_wid.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter image path:")
    label_set = Label(image_path_wid, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5)

    global view_image_path
    view_image_path = StringVar(None)
    view_image_path = Entry(image_path_wid, textvariable=view_image_path, width=50)
    view_image_path.grid(row=0, column=1, padx=5)

    enter_btn = tk.Button(image_path_wid, bg="seashell3", text="Enter", command=get_view_image_path, width=5, height=1)
    enter_btn.grid(row=0, column=2, padx=20, pady=5, sticky=W)


def check_folder(path):
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(".hdr"):
                return True

    return False


def get_vectors():
    global vectors_path
    path = vectors_path.get()

    global s_v
    swir_or_vnir_or_all = s_v
    if os.path.isfile(path) and path.endswith(".txt") or os.path.isdir(path):
        plot_vectors(path, swir_or_vnir_or_all)

    else:
        messagebox.showerror(title="Error", message="You need a file/folder path")


def view_vectors():
    global vectors_window
    vectors_window = Toplevel(root)
    vectors_window.wm_iconbitmap("drone.ico")
    vectors_window.title("View Signatures")
    vectors_window.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter file/folder of signatures:")
    label_set = Label(vectors_window, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5)

    global vectors_path
    vectors_path = StringVar(None)
    vectors_path = Entry(vectors_window, textvariable=vectors_path, width=50)
    vectors_path.grid(row=0, column=1, padx=5)

    enter_btn = tk.Button(vectors_window, bg="seashell3", text="Enter", command=get_vectors, width=5, height=1)
    enter_btn.grid(row=0, column=2, padx=20, pady=5, sticky=W)


def get_folders():
    global images_path, sig_path, angle_path
    image, sig, angle = images_path.get(), sig_path.get(), angle_path.get()
    global s_v
    swir_or_vnir = s_v

    if (os.path.isfile(image) and image.endswith("hdr")) or check_folder(image) and os.path.isdir(
            sig) and os.path.isfile(angle):
        if os.path.isfile(image):
            print(image)
            print(sig)
            classify_image_by_angles(image, sig, swir_or_vnir)

        else:
            folder_classify(image, sig, swir_or_vnir, angle)

    else:
        messagebox.showerror(title="Error", message="You need a file/folder path")


def enter_folders_locations():
    global folders_window
    folders_window = Toplevel(root)
    folders_window.wm_iconbitmap("drone.ico")
    folders_window.title("Classify Data")
    folders_window.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter folder of images:")
    label_set = Label(folders_window, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5, sticky=W)

    global images_path
    images_path = StringVar(None)
    images_path = Entry(folders_window, textvariable=images_path, width=50)
    images_path.grid(row=0, column=1, padx=5)

    label_text = StringVar()
    label_text.set("Enter folder of signatures:")
    label_set = Label(folders_window, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=1, column=0, padx=5, sticky=W)

    global sig_path
    sig_path = StringVar(None)
    sig_path = Entry(folders_window, textvariable=sig_path, width=50)
    sig_path.grid(row=1, column=1, padx=5)

    label_text = StringVar()
    label_text.set("Enter file of angles:")
    label_set = Label(folders_window, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=2, column=0, padx=5, sticky=W)

    global angle_path
    angle_path = StringVar(None)
    angle_path = Entry(folders_window, textvariable=angle_path, width=50)
    angle_path.grid(row=2, column=1, padx=5)

    enter_btn = tk.Button(folders_window, bg="seashell3", text="Enter", command=get_folders, width=5, height=1)
    enter_btn.grid(row=3, column=0, padx=20, pady=5, sticky=W)


def get_list():
    global type_entries, name_entries
    global list_of_create_data
    list_of_create_data.clear()
    global create_path
    path = create_path.get()
    global s_v
    swir_or_vnir = s_v

    for i in range(len(type_entries)):
        if type_entries[i].get() in TYPES:
            list_of_create_data.append([type_entries[i].get(), name_entries[i].get()])

    if len(list_of_create_data) == 0:
        messagebox.showerror(title="Error", message="You need to enter at least one object")

    else:
        file_create(path, list_of_create_data, swir_or_vnir)


def set_input_type_entry():
    global create_data_window
    for widget in create_data_window.grid_slaves():
        if int(widget.grid_info()["row"]) > 2:
            widget.grid_forget()

    row = 1
    global obj_entr
    num_entered = obj_entr.get()
    global type_entries, name_entries
    type_entries, name_entries = [], []

    global create_path
    path = create_path.get()

    try:
        if int(num_entered) > 0 and (os.path.isfile(path) and path.endswith(".hdr")):
            label_text = StringVar()
            label_text.set("Enter type of information:")
            label_set = Label(create_data_window, bg="seashell2", textvariable=label_text, height=2)
            label_set.grid(row=2, column=0, padx=5, sticky=W)

            for i in range(int(num_entered)):
                row += 2

                label_text = StringVar()
                label_text.set("Enter file type:")
                label_set = Label(create_data_window, bg="seashell2", textvariable=label_text, height=2)
                label_set.grid(row=row, column=0, padx=20, sticky=W)

                type_entr = StringVar(None)
                type_entr = Entry(create_data_window, textvariable=type_entr, width=15)
                type_entr.grid(row=row, column=1, padx=5)

                type_entries.append(type_entr)

                label_text = StringVar()
                label_text.set("Enter file name:")
                label_set = Label(create_data_window, bg="seashell2", textvariable=label_text, height=2)
                label_set.grid(row=row + 1, column=0, padx=20, pady=(0, 20), sticky=W)

                name_entr = StringVar(None)
                name_entr = Entry(create_data_window, textvariable=name_entr, width=15)
                name_entr.grid(row=row + 1, column=1, padx=5, pady=(0, 20))

                name_entries.append(name_entr)

            enter_btn = tk.Button(create_data_window, bg="seashell3", text="Enter", command=get_list, width=5, height=1)
            enter_btn.grid(row=row + 2, column=0, padx=20, pady=5, sticky=W)

            enter_btn = tk.Button(create_data_window, bg="seashell3", text="Quit", command=create_data_window.destroy,
                                  width=5, height=1)
            enter_btn.grid(row=row + 2, column=1, padx=20, pady=5, sticky=W)

    except ValueError:
        pass


def get_sensor(sensor):
    global s_v
    s_v = sensor
    swir_or_vnir = s_v
    if swir_or_vnir == "swir" or swir_or_vnir == "vnir" or swir_or_vnir == "all":
        view_main()

    else:
        messagebox.showerror(title="Error", message="You need to write swir/vnir/all")


def enter_data_window():
    global create_data_window
    create_data_window = Toplevel(root)
    create_data_window.wm_iconbitmap("drone.ico")
    create_data_window.title("Create Data")
    create_data_window.configure(bg="seashell2")

    label_text = StringVar()
    label_text.set("Enter file path:")
    label_set = Label(create_data_window, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=0, column=0, padx=5, sticky=W)

    global create_path
    create_path = StringVar(None)
    create_path = Entry(create_data_window, textvariable=create_path, width=30)
    create_path.grid(row=0, column=1, padx=5, sticky=W)

    label_text = StringVar()
    label_text.set("Enter number of objects:")
    label_set = Label(create_data_window, bg="seashell2", textvariable=label_text, height=2)
    label_set.grid(row=1, column=0, padx=5, sticky=W)

    global obj_entr
    obj_entr = StringVar(None)
    obj_entr = Entry(create_data_window, textvariable=obj_entr, width=15)
    obj_entr.grid(row=1, column=1, padx=5, sticky=W)

    enter_btn = tk.Button(create_data_window, bg="seashell3", text="Enter", command=set_input_type_entry, width=5,
                          height=1)
    enter_btn.grid(row=1, column=2, padx=5)


def view_main():
    for widget in frame.grid_slaves():
        if int(widget.grid_info()["row"]) == 0:
            widget.grid_forget()

    create_data_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="Create Data",
                                command=enter_data_window, width=12, height=3)
    create_data_btn.grid(row=0, column=0, padx=10, pady=10)

    classify_data_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="Classify Data",
                                  command=enter_folders_locations, width=12, height=3)
    classify_data_btn.grid(row=0, column=1, padx=10, pady=10)

    show_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="View Image", command=get_image_path,
                         width=12, height=3)
    show_btn.grid(row=1, column=0, padx=10, pady=10)

    vectors_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="View Signatures",
                            command=view_vectors,
                            width=12, height=3)
    vectors_btn.grid(row=1, column=1, padx=10, pady=10)

    sig_btn_db = tk.Button(frame, bg="seashell3", activebackground="gray99", text="DB Sig", command=view_sigs_db,
                           width=12, height=3)
    sig_btn_db.grid(row=2, column=0, padx=10, pady=10)

    sig_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="Polygon Sig", command=view_sigs_pol,
                        width=12, height=3)
    sig_btn.grid(row=2, column=1, padx=10, pady=10)

    compare_sig_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="Compare Sig",
                                command=view_sigs_compare, width=12, height=3)
    compare_sig_btn.grid(row=3, column=0, padx=10, pady=10)

    compare_sig_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="Ksvd", command=view_ksvd,
                                width=12, height=3)
    compare_sig_btn.grid(row=3, column=1, padx=10, pady=10)

    compare_sig_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="Pixel Converter",
                                command=view_pixel_to_cord, width=12, height=3)
    compare_sig_btn.grid(row=4, column=0, padx=10, pady=10)

    compare_sig_btn = tk.Button(frame, bg="seashell3", activebackground="gray99", text="Find Pixel",
                                command=view_find_pixel, width=12, height=3)
    compare_sig_btn.grid(row=4, column=1, padx=10, pady=10)


if __name__ == "__main__":
    if os.path.isdir("results_17.5"):
        pass
    else:
        os.mkdir("results_17.5")
    root = tk.Tk()
    root.title("Hyper")
    root.wm_iconbitmap("drone.ico")
    frame = tk.Frame(root)
    frame.configure(bg="seashell2")
    frame.pack()

    swir_btn = tk.Button(frame, bg="seashell3", text="Swir", command=lambda: get_sensor("swir"), width=5, height=1)
    swir_btn.grid(row=0, column=0, padx=20, pady=5, sticky=W)
    vnir_btn = tk.Button(frame, bg="seashell3", text="Vnir", command=lambda: get_sensor("vnir"), width=5, height=1)
    vnir_btn.grid(row=0, column=1, padx=20, pady=5, sticky=W)
    all_btn = tk.Button(frame, bg="seashell3", text="All", command=lambda: get_sensor("all"), width=5, height=1)
    all_btn.grid(row=0, column=2, padx=20, pady=5, sticky=W)

    root.mainloop()


