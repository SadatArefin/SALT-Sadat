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

## 用于缩放图像
all_images = os.listdir('rename')

scale_factor = 4

for images_i in all_images:
    images_path = os.path.join('rename', images_i)
    new_image_path = images_path.replace('rename', 'renamed').replace('_HR_x2_SR.png', '.jpg')
    # old_image_path = images_path.replace('annotated_images', 'resized_lr').replace('.png', '_LR.png')




    shutil.copy(images_path, new_image_path)







