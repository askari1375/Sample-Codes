# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 12:09:14 2021

@author: Amirhossein
"""

from simple_3d_structure import Volume, create_side_top_images
import numpy as np
import cv2
from tqdm import tqdm


class RandomFunction:
    
    def __init__(self):        
        
        self.p1 = np.polynomial.Polynomial(self.generate_random_polinomial())
        self.p2 = np.polynomial.Polynomial(self.generate_random_polinomial())
        
        
        
    def generate_random_polinomial(self, max_n = 10):
        p_coef = [0, 0, 0]
        #n = np.random.randint(1, max_n)
        n = max_n
        for k in range(n):
            p_coef.append(np.random.uniform(-5, 5))
        return p_coef
    
    def __call__(self, t):
        x = t
         
        y = self.p1(t)
        z = self.p2(t)
     
        return np.array([x, y, z])



central_curve_points = 50
points_on_each_circle = 1000

image_shape = (400, 400)
scale = 200

random_function = RandomFunction()

for k in tqdm(range(200)):
    
    random_function = RandomFunction()
    volume = Volume(random_function,
                    length = 1,
                    radious = 0.2,
                    central_curve_points = central_curve_points,
                    circle_points_number = points_on_each_circle)
    
    
    top_image, side_x_image, side_y_image = create_side_top_images(volume.points, image_shape, scale)
    
    cv2.imwrite("simulated data/without occlusion/validation/input/{}.png".format(k), top_image * 255)
    cv2.imwrite("simulated data/without occlusion/validation/output 1/{}.png".format(k), side_x_image * 255)
    cv2.imwrite("simulated data/without occlusion/validation/output 2/{}.png".format(k), side_y_image * 255)