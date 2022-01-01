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

all_exif_txt = os.listdir('annotated_exif')

num_all = []
for i in all_exif_txt:
    f_exif = open(os.path.join('annotated_exif', i), 'r')
    for lines in f_exif.readlines():
        num_exif = float((lines.strip().split())[0])
        print(num_exif)
        num_all.append(num_exif)


print('min = ', min(num_all))
print('max = ', max(num_all))


