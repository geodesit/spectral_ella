import matplotlib.pyplot as plt
import numpy as np
from spectral import *
from get_data_sets import *

# hdr_path = "E:\\NISOY\\Hod_hasharon_part_4\\swir\\100034_hodhasharon_exp2_part4_2022_12_05_12_51_00" \
#            "\\raw_12000_rd_rf.hdr"
# file_path = "E:\\NISOY\\Hod_hasharon_part_4\\swir\\100034_hodhasharon_exp2_part4_2022_12_05_12_51_00\\raw_12000_rd_rf"
# hdr = envi.open(hdr_path, file_path).load()

# (m, c) = kmeans(hdr, 20, 30)
# plt.imshow(m)
# plt.show()

show_img()
hdr = get_hdr()
hdr_load = get_hdr_load()
gt = np.zeros((hdr.shape[0], hdr.shape[1]))
gt = insert_label(gt, 1, 2, 401, 246, 938)
gt = insert_label(gt, 2, 215, 310, 256, 345)
gt = insert_label(gt, 3, 383, 649, 415, 709)

pc = principal_components(hdr_load)
pc_0999 = pc.reduce(fraction=0.999)
img_pc = pc_0999.transform(hdr_load)

classes = create_training_classes(img_pc, gt)
gmlc = GaussianClassifier(classes)
clmap = gmlc.classify_image(img_pc)
clmap_training = clmap * (gt != 0)
v = imshow(classes=clmap)
plt.pause(1000000)

