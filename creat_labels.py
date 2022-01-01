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

connection_csv = 'labels_to_Zenodo.csv'
connection_data = []
with open(connection_csv) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    # birth_header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        connection_data.append(row)


# birth_data = [[float(x) for x in row] for row in birth_data]  # 将数据从string形式转换为float形式


connection_data = np.array(connection_data)  # 将list数组转化成array数组便于查看数据结构

print('shape = ', connection_data.shape)  # 利用.shape查看结构。
print('[0] = ', connection_data[0])
print('[1] = ', connection_data[1])
print('[1][1] = ', connection_data[1][1])

for rows in range(connection_data.shape[0]):
    current_zenodo = connection_data[rows][0].strip()
    print('current_zenodo = ', current_zenodo)
    current_origin = connection_data[rows][1].strip()
    print('current_origin = ', current_origin)

    if current_origin == '-1':
        continue

    else:
        # 在train.txt中搜索
        train_open = open('RSE2018/train.txt', 'r')
        for lines in train_open.readlines():
            if lines.find(current_origin) != -1:
                print('Find in train ', rows)
                new_train_path = str(Path('new_train') / Path(current_zenodo.replace('.JPG', '.txt')))
                old_train_path = str(current_origin)
                shutil.copy(old_train_path, new_train_path)
                continue
        train_open.close()

        # 在test.txt中搜索
        test_open = open('RSE2018/test.txt', 'r')
        for lines in test_open.readlines():
            if lines.find(current_origin) != -1:
                print('Find in test ', rows)
                new_test_path = str(Path('new_test') / Path(current_zenodo.replace('.JPG', '.txt')))
                old_test_path = str(current_origin)
                shutil.copy(old_test_path, new_test_path)
                continue
        test_open.close()

        # 在val.txt中搜索
        val_open = open('RSE2018/val.txt', 'r')
        for lines in val_open.readlines():
            if lines.find(current_origin) != -1:
                print('Find in val ', rows)
                new_val_path = str(Path('new_val') / Path(current_zenodo.replace('.JPG', '.txt')))
                old_val_path = str(current_origin)
                shutil.copy(old_val_path, new_val_path)
                continue
        val_open.close()






