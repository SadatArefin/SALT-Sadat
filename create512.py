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
import math

import geopandas as gpd                     # 导入包
import descartes
import random
import shutil
# import utils


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

all_train_txt = os.listdir('new_train')
all_test_txt = os.listdir('new_test')
all_val_txt = os.listdir('new_val')
current_count = 0
current_count_all = 0

write_split_train = open('train11.txt', 'w')
for i in all_train_txt:
    # print(i)
    train_txt = os.path.join('new_train', i)
    original_name = train_txt.replace('.txt', '.jpg').replace('new_train', 'VOC2007')
    img = cv2.imread(original_name)

    # 读取当前train.txt文件
    for b in range(48):
        patch_name = 'D:\\pytorch_project\\file_process\\new_patches\\' + '%06d' % int(current_count_all * 48 + b + 1) + '.jpg'

        cv2.imwrite(patch_name, img[int(500 * math.floor(b/8)) : int(500 * math.floor(b/8) + 500), int(500 * (b % 8)) : int(500 * (b % 8) + 500), :])

        write_split = 'data/' + 'images/' + str('%06d' % int(current_count_all * 48 + b + 1)) + '.jpg' + '\n'
        # write_split = os.path.join('data', 'images', str('%06d' % int(current_count_all * 48 + b + 1)) + '.jpg' + '\n')
        write_split_train.write(write_split)

    train_open = open(train_txt, 'r')



    for lines in train_open.readlines():
        items = lines.strip().split(' ')
        coord_x = float(items[0]) + float(items[2]) / 2
        coord_y = float(items[1]) + float(items[3]) / 2
        rename_x_index = math.floor(coord_x/500)
        rename_y_index = math.floor(coord_y/500)

        new_write_label = 'D:\\pytorch_project\\file_process\\new_write_labels\\' + '%06d' % \
                          int(current_count_all * 48 + rename_x_index + rename_y_index * 8 + 1) + '.txt'

        float_write_label = 'D:\\pytorch_project\\file_process\\newww_write_labels\\' + '%06d' % \
                          int(current_count_all * 48 + rename_x_index + rename_y_index * 8 + 1) + '.txt'

        file_write = open(new_write_label, 'a')

        float_write = open(float_write_label, 'a')

        # 更新到现在小patches的坐标
        new_coord_left = max(float(items[0]) - 500 * rename_x_index, 0)
        new_coord_top = max(float(items[1]) - 500 * rename_y_index, 0)
        new_coord_right = min(float(items[0]) + float(items[2]) - 500 * rename_x_index, 500)
        new_coord_bottom = min(float(items[1]) + float(items[3]) - 500 * rename_y_index, 500)

        current_lines = ['0', str(new_coord_left), str(new_coord_top), str(new_coord_right), str(new_coord_bottom)]

        for nums in current_lines:
            file_write.write(nums + ' ')
        file_write.write('\n')
        file_write.close()

        box = (new_coord_left, new_coord_right, new_coord_top, new_coord_bottom)
        bb = convert((500, 500), box)
        float_write.write(str('0') + " " + " ".join([str(a) for a in bb]) + '\n')
        float_write.close()


        print(items)
    train_open.close()

    current_count_all = current_count_all + 1
write_split_train.close()


write_split_test = open('test11.txt', 'w')
for i in all_test_txt:
    # print(i)
    test_txt = os.path.join('new_test', i)
    original_name = test_txt.replace('.txt', '.jpg').replace('new_test', 'VOC2007')
    img = cv2.imread(original_name)

    # 读取当前test.txt文件
    for b in range(48):
        patch_name = 'D:\\pytorch_project\\file_process\\new_patches\\' + '%06d' % int(current_count_all * 48 + b + 1) + '.jpg'

        cv2.imwrite(patch_name, img[int(500 * math.floor(b/8)) : int(500 * math.floor(b/8) + 500), int(500 * (b % 8)) : int(500 * (b % 8) + 500), :])

        write_split = 'data/' + 'images/' + str('%06d' % int(current_count_all * 48 + b + 1)) + '.jpg' + '\n'
        # write_split = os.path.join('data', 'images', str('%06d' % int(current_count_all * 48 + b + 1)) + '.jpg' + '\n')
        write_split_test.write(write_split)


    test_open = open(test_txt, 'r')
    for lines in test_open.readlines():
        items = lines.strip().split(' ')
        coord_x = float(items[0]) + float(items[2]) / 2
        coord_y = float(items[1]) + float(items[3]) / 2
        rename_x_index = math.floor(coord_x/500)
        rename_y_index = math.floor(coord_y/500)

        new_write_label = 'D:\\pytorch_project\\file_process\\new_write_labels\\' + '%06d' % \
                          int(current_count_all * 48 + rename_x_index + rename_y_index * 8 + 1) + '.txt'
        float_write_label = 'D:\\pytorch_project\\file_process\\newww_write_labels\\' + '%06d' % \
                            int(current_count_all * 48 + rename_x_index + rename_y_index * 8 + 1) + '.txt'

        file_write = open(new_write_label, 'a')

        float_write = open(float_write_label, 'a')
        # 更新到现在小patches的坐标
        new_coord_left = max(float(items[0]) - 500 * rename_x_index, 0)
        new_coord_top = max(float(items[1]) - 500 * rename_y_index, 0)
        new_coord_right = min(float(items[0]) + float(items[2]) - 500 * rename_x_index, 500)
        new_coord_bottom = min(float(items[1]) + float(items[3]) - 500 * rename_y_index, 500)

        current_lines = ['0', str(new_coord_left), str(new_coord_top), str(new_coord_right), str(new_coord_bottom)]

        for nums in current_lines:
            file_write.write(nums + ' ')
        file_write.write('\n')
        file_write.close()

        box = (new_coord_left, new_coord_right, new_coord_top, new_coord_bottom)
        bb = convert((500, 500), box)
        float_write.write(str('0') + " " + " ".join([str(a) for a in bb]) + '\n')
        float_write.close()

        print(items)
    test_open.close()

    current_count_all = current_count_all + 1
write_split_test.close()


write_split_val = open('val11.txt', 'w')
for i in all_val_txt:
    # print(i)
    val_txt = os.path.join('new_val', i)
    original_name = val_txt.replace('.txt', '.jpg').replace('new_val', 'VOC2007')
    img = cv2.imread(original_name)

    # 读取当前val.txt文件
    for b in range(48):
        patch_name = 'D:\\pytorch_project\\file_process\\new_patches\\' + '%06d' % int(current_count_all * 48 + b + 1) + '.jpg'

        cv2.imwrite(patch_name, img[int(500 * math.floor(b/8)) : int(500 * math.floor(b/8) + 500), int(500 * (b % 8)) : int(500 * (b % 8) + 500), :])

        write_split = 'data/' + 'images/' + str('%06d' % int(current_count_all * 48 + b + 1)) + '.jpg' + '\n'
        # write_split = os.path.join('data', 'images', str('%06d' % int(current_count_all * 48 + b + 1)) + '.jpg' + '\n')
        write_split_val.write(write_split)


    val_open = open(val_txt, 'r')
    for lines in val_open.readlines():
        items = lines.strip().split(' ')
        coord_x = float(items[0]) + float(items[2]) / 2
        coord_y = float(items[1]) + float(items[3]) / 2
        rename_x_index = math.floor(coord_x/500)
        rename_y_index = math.floor(coord_y/500)

        new_write_label = 'D:\\pytorch_project\\file_process\\new_write_labels\\' + '%06d' % \
                          int(current_count_all * 48 + rename_x_index + rename_y_index * 8 + 1) + '.txt'
        float_write_label = 'D:\\pytorch_project\\file_process\\newww_write_labels\\' + '%06d' % \
                            int(current_count_all * 48 + rename_x_index + rename_y_index * 8 + 1) + '.txt'

        file_write = open(new_write_label, 'a')

        float_write = open(float_write_label, 'a')

        # 更新到现在小patches的坐标
        new_coord_left = max(float(items[0]) - 500 * rename_x_index, 0)
        new_coord_top = max(float(items[1]) - 500 * rename_y_index, 0)
        new_coord_right = min(float(items[0]) + float(items[2]) - 500 * rename_x_index, 500)
        new_coord_bottom = min(float(items[1]) + float(items[3]) - 500 * rename_y_index, 500)

        current_lines = ['0', str(new_coord_left), str(new_coord_top), str(new_coord_right), str(new_coord_bottom)]

        for nums in current_lines:
            file_write.write(nums + ' ')
        file_write.write('\n')
        file_write.close()

        box = (new_coord_left, new_coord_right, new_coord_top, new_coord_bottom)
        bb = convert((500, 500), box)
        float_write.write(str('0') + " " + " ".join([str(a) for a in bb]) + '\n')
        float_write.close()

        print(items)
    val_open.close()

    current_count_all = current_count_all + 1
write_split_val.close()


print('current_count_all = ', current_count_all)







