# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:34:22 2022

@author: Amirhossein
"""

import os
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


data_reduction_scale = 5
results_reduction_scale = 3

data_images_path = []
results_images_path = []

for (root,dirs,files) in os.walk("./"):
    for file_name in files:
        if file_name[-3:] == "png":
            if file_name[:-4].isnumeric():
                data_images_path.append(os.path.join(root, file_name))
            else:
                results_images_path.append(os.path.join(root, file_name))
            
        


for image_path in tqdm(data_images_path):
    img = io.imread(image_path)
    if np.max(img.shape) > 100:
        resized_image = resize(img, (img.shape[0] // data_reduction_scale,
                                     img.shape[1] // data_reduction_scale))
    io.imsave(image_path, resized_image)




for image_path in tqdm(results_images_path):
    img = io.imread(image_path)
    if np.max(img.shape) > 100:
        resized_image = resize(img, (img.shape[0] // results_reduction_scale,
                                     img.shape[1] // results_reduction_scale))
    io.imsave(image_path, resized_image)




