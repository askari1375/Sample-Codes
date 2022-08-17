# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 12:02:45 2021

@author: Amirhossein
"""

import cv2
import numpy as np
import os
import time


def fill_region(img, point):
    x = point[0]
    y = point[1]
    result = np.copy(img)
    
    candidates = [(x, y)]
    
    while len(candidates) > 0:
        x, y = candidates.pop()
        if result[x, y] == 0:
            result[x, y] = 1
        for i in [1, 0, -1]:
            for j in [1, 0, -1]:
                if x + i >= 0 and x + i < result.shape[0] and y + j >= 0 and y + j < result.shape[1]:
                    if result[x + i, y + j] == 0:
                        candidates.append((x + i, y + j))
    return result

def find_background(image):
    
    start_time = time.time()
    max_x = image.shape[0] - 1
    max_y = image.shape[1] - 1
    
    raw_background = fill_region(image, (0, 0))
    raw_background = fill_region(raw_background, (max_x, 0))
    raw_background = fill_region(raw_background, (0, max_y))
    raw_background = fill_region(raw_background, (max_x, max_y))
    
    background = raw_background - image
    
    #print("Background find in \t{} seconds".format(time.time() - start_time))
    return background


def create_trunk_mask(image, threshold = 0.9):
    
    img = cv2.GaussianBlur(image, (11, 11), cv2.BORDER_REFLECT)
    img[img < threshold] = 0
    img[img >= threshold] = 1
    filled_image = 1 - find_background(img)
    return filled_image


def main():
    image_names = os.listdir("targets")
    
    threshold = 0.9
    
    idx = 25
    
    image_name = image_names[idx]
    image = cv2.imread("targets/{}".format(image_name)) / 255
    image = np.mean(image, axis = 2)
    
    filled_image = create_trunk_mask(image, threshold)
    
    
    cv2.imshow("original image", image)
    cv2.imshow("filled image", filled_image)
    
    
if __name__ == "__main__":
    main()









