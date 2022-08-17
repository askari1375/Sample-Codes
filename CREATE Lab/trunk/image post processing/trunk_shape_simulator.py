# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:47:09 2021

@author: Amirhossein
"""

import numpy as np
import cv2
    

def create_central_points(angles_change, x_0, y_0, length = 1):
    n = angles_change.shape[0]
    segment_length = length / n
    
    central_points = np.zeros((n + 1, 2))
    central_points[0, 0] = x_0
    central_points[0, 1] = y_0
    
    alpha = 0
    
    for k in range(n):
        alpha += angles_change[k]
        
        x = central_points[k, 0] + segment_length * np.cos(alpha)
        y = central_points[k, 1] + segment_length * np.sin(alpha)
        
        central_points[k + 1, 0] = x
        central_points[k + 1, 1] = y
    
    return central_points


def create_side_points(central_points, angles_change, r_0):
    n = angles_change.shape[0]
    
    r = np.linspace(r_0, 0, n + 1)
    
    side_points = np.zeros((n + 1, 4))
    side_points[0, 0] = central_points[0, 0]
    side_points[0, 1] = central_points[0, 1] + r_0
    side_points[0, 2] = central_points[0, 0]
    side_points[0, 3] = central_points[0, 1] - r_0
    
    alpha = 0
    
    for k in range(n):
        alpha += angles_change[k]
        
        alpha_1 = alpha + np.pi / 2
        alpha_2 = alpha - np.pi / 2
        
        x_0 = central_points[k + 1, 0]
        y_0 = central_points[k + 1, 1]
        
        x_1 = r[k + 1] * np.cos(alpha_1) + x_0
        y_1 = r[k + 1] * np.sin(alpha_1) + y_0
        x_2 = r[k + 1] * np.cos(alpha_2) + x_0
        y_2 = r[k + 1] * np.sin(alpha_2) + y_0
        
        side_points[k + 1, 0] = x_1
        side_points[k + 1, 1] = y_1
        side_points[k + 1, 2] = x_2
        side_points[k + 1, 3] = y_2
    
    return side_points



def create_image_from_side_points(side_points, image_shape):
    img = np.zeros(image_shape)
    p_1 = side_points[:, 0:2]
    p_2 = side_points[::-1, 2:4]
    points = np.concatenate((p_1[:, ::-1], p_2[:, ::-1]), axis = 0)
    points = points.astype('int32')
    img = cv2.drawContours(img, [points], 0, (255, 255, 255), -1)
    return img / 255



def create_trunk_image(angles_change):
    
    aspect_ratio = 0.06
    x_0 = 0.14
    y_0 = 0.505
    image_shape = (480, 480)
    scale = 395
    
    
    x_0 = image_shape[0] * x_0
    y_0 = image_shape[1] * y_0
    r_0 = aspect_ratio * scale
    
    central_points = create_central_points(angles_change, x_0, y_0, scale)
    side_points = create_side_points(central_points, angles_change, r_0)
    img = create_image_from_side_points(side_points, image_shape)
    return img, side_points

def main():
    n = 20
    
    angles_change = np.ones((n,)) * np.pi * (-1 / 180)
    
    image, _ = create_trunk_image(angles_change)
    
    cv2.imshow("result", image)


if __name__ == "__main__":
    main()




