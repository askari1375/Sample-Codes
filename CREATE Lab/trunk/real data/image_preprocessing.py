# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:42:16 2021

@author: Amirhossein
"""

import cv2
import os
import numpy as np
from tqdm import tqdm


def input_image_processor(img, center_percentage):
    if np.max(img) > 1:
        img = img / np.max(img)
    img = square_image(img, center_percentage)
    img = cv2.flip(img, 1)
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img = np.mean(img, axis = 2)
    return img


def output_image_processor(img, center_percentage):
    if np.max(img) > 1:
        img = img / np.max(img)
    img = square_image(img, center_percentage)
    img = cv2.flip(img, 0)    
    img = np.mean(img, axis = 2)
    return img
    


def square_image(img, center_percentage):
    
    min_center = (img.shape[0] / 2) / img.shape[1]
    max_center = 1 - min_center
    
    center = max(min_center, center_percentage)
    center = min(center, max_center)
    
    left_border = int(center * img.shape[1] - img.shape[0] / 2)
    right_border = int(center * img.shape[1] + img.shape[0] / 2)
    img = img[:, left_border:right_border,:]
    return img


def main():
    input_center_percentage = 0.5
    output_center_percentage = 0.53
    keyword = "validation"
    
    input_names = list(os.walk("data/raw {} data/input".format(keyword)))[0][2]
    output_names = list(os.walk("data/raw {} data/output".format(keyword)))[0][2]
    
    for input_name in tqdm(input_names):
        img = cv2.imread("data/raw {} data/input/".format(keyword) + input_name)    
        img = input_image_processor(img, input_center_percentage)
        cv2.imwrite("{} data/input/".format(keyword) + input_name, img)
    
    for output_name in tqdm(output_names):
        img = cv2.imread("data/raw {} data/output/".format(keyword) + output_name)    
        img = output_image_processor(img, output_center_percentage)
        cv2.imwrite("{} data/output/".format(keyword) + output_name, img)


if __name__ == "__main__":
    main()