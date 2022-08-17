# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:27:24 2021

@author: Amirhossein
"""

import cv2
import numpy as np
from tqdm import tqdm
from data_simulator_top_view import create_top_view_images
from data_simulator_side_view import create_side_view_image


def generate_random_polinomial(max_n = 10):
    p_coef = [0, 0, 0]
    #n = np.random.randint(1, max_n)
    n = max_n
    for k in range(n):
        p_coef.append(np.random.uniform(-5, 5))
    return p_coef


for k in tqdm(range(200)):
    p_coef = generate_random_polinomial()
    length = 1
    r_0 = 0.25
    thickness = 1
    
    n = 50
    image_shape = (512, 512, 3)
    
    image_1, image_2 = create_top_view_images(p_coef, length, r_0, n, image_shape, thickness)
    image_3 = create_side_view_image(p_coef, length, r_0, n, image_shape, thickness)    
    
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("simulated data/val1/{}.png".format(k), image_1)
    cv2.imwrite("simulated data/val2/{}.png".format(k), image_3)
    cv2.imwrite("simulated data/val3/{}.png".format(k), image_2)

#cv2.imshow("result 1", np.concatenate((image_1, image_3), axis = 1))
#cv2.imshow("result 2", np.concatenate((image_2, image_3), axis = 1))


