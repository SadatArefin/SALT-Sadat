import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import csv
from pathlib import Path
# import read_shp
import exifread

import geopandas as gpd                     # 导入包
import descartes
import random
import shutil
# import utils

## 用于把含有annotation files的图像隔离出来
all_images = os.listdir('new_patches')
all_labels = os.listdir('new_write_labels')

write_split_train = open('train22.txt', 'w')
write_split_test = open('test22.txt', 'w')
write_split_val = open('val22.txt', 'w')

for labels_i in all_labels:
    labels_path = os.path.join('new_write_labels', labels_i)
    old_image_path = labels_path.replace('.txt', '.png').replace('new_write_labels', 'new_patches')


    new_image_path = old_image_path.replace('new_patches', 'annotated_images')
    # print(labels_path)
    # print(old_image_path)
    # print(new_image_path)

    old_exif_path = old_image_path.replace('.png', '.txt').replace('new_patches', 'exif_temp')
    new_exif_path = old_exif_path.replace('exif_temp', 'annotated_exif')

    shutil.copy(old_image_path, new_image_path)
    shutil.copy(old_exif_path, new_exif_path)


    ## 在几个split里搜索
    train_open = open('train11.txt', 'r')
    for lines in train_open.readlines():
        if lines.find(labels_i.replace('.txt', '.png')) != -1:
            write_str = 'data/' + 'images/' + labels_i.replace('.txt', '.jpg') + '\n'
            write_split_train.write(write_str)
    train_open.close()

    test_open = open('test11.txt', 'r')
    for lines in test_open.readlines():
        if lines.find(labels_i.replace('.txt', '.png')) != -1:
            write_str = 'data/' + 'images/' + labels_i.replace('.txt', '.jpg') + '\n'
            write_split_test.write(write_str)
    test_open.close()

    val_open = open('val11.txt', 'r')
    for lines in val_open.readlines():
        if lines.find(labels_i.replace('.txt', '.png')) != -1:
            write_str = 'data/' + 'images/' + labels_i.replace('.txt', '.jpg') + '\n'
            write_split_val.write(write_str)
    val_open.close()

write_split_train.close()
write_split_test.close()
write_split_val.close()





