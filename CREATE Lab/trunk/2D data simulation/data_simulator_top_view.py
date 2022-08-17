# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:18:24 2021

@author: Amirhossein
"""

import numpy as np
import cv2
from functions import find_curves, find_x_list



def create_ellipse_mask(c_x, c_y, r_x, r_y, image_shape):
    mask = np.zeros((image_shape[0], image_shape[1], 1))
    mask = cv2.ellipse(mask, (c_x, c_y), (r_x, r_y),
                       0, 0, 360, color = (255, 255, 255), thickness = -1)
    mask /= np.max(mask)
    mask = np.repeat(mask, 3, axis = 2)
    return mask


def create_top_view_images(p_coef, length, r_0, n, image_shape, thickness):
    
    scale = image_shape[0] / 2
    r = np.linspace(r_0, 0, n)
    dist_list = np.linspace(0, 1, n)
    x = find_x_list(p_coef, dist_list, epsilon = 1e-4)
    
    middle, up, down = find_curves(x, p_coef, r)
    
    
    ellipses = np.zeros((n, 4))
    for k in range(n - 1):
        c_x = 0
        c_y = middle[k, 1]
        width = r[k] * 2
        xx = np.abs(up[k, 0] - down[k, 0])
        yy = np.abs(up[k, 1] - down[k, 1])    
        height = 2 * r[k] * (yy) / (np.sqrt(np.power(xx, 2) + np.power(yy, 2)))
        
        ellipses[k, :] = c_x, c_y, width, height
    
    
    image_1 = np.zeros(image_shape)
    image_2 = np.zeros(image_shape)
    
    for k in list(reversed(range(ellipses.shape[0]))):
        c_x = int(ellipses[k, 0] * scale + image_shape[1] / 2)
        c_y = int(image_shape[0] / 2 - ellipses[k, 1] * scale)
        r_x = int(ellipses[k, 2] * scale / 2)
        r_y = int(ellipses[k, 3] * scale / 2)
        
        mask = create_ellipse_mask(c_x, c_y, r_x, r_y, image_shape)
        image_1 = np.multiply(image_1, mask)
        image_1 = cv2.ellipse(image_1, (c_x, c_y), (r_x, r_y),
                              0, 0, 360, color = (0, 255, 0), thickness = thickness)
        image_2 = cv2.ellipse(image_2, (c_x, c_y), (r_x, r_y),
                              0, 0, 360, color = (0, 255, 0), thickness = thickness)
    
    image_1 = image_1.astype('uint8')
    image_2 = image_2.astype('uint8')
    rotation_matrix = cv2.getRotationMatrix2D((image_1.shape[1] // 2, image_1.shape[0] // 2), -90, 1.0)
    image_1 = cv2.warpAffine(image_1, rotation_matrix, (image_1.shape[1], image_1.shape[0]))
    image_2 = cv2.warpAffine(image_2, rotation_matrix, (image_2.shape[1], image_2.shape[0]))
  
    return image_1, image_2


def main():
    p_coef = [0, 0, 1]
    n = 50
    length = 1
    r_0 = 0.2
    thickness = 1
    image_shape = (1000, 800, 3)
    scale = 500
    
    image_1, image_2 = create_top_view_images(p_coef, length, r_0, n,
                                              image_shape, thickness)

    cv2.imshow("result 1", image_1)
    cv2.imshow("result 2", image_2)

if __name__ == "__main__":
    main()