# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:58:18 2021

@author: Amirhossein
"""

import numpy as np
import cv2
import os
from target_image_processor import create_trunk_mask
from trunk_shape_simulator import create_trunk_image
from tqdm import tqdm



def load_image(image_name):    
    image = cv2.imread("targets/{}".format(image_name)) / 255
    image = np.mean(image, axis = 2)
    image = create_trunk_mask(image)
    return image


class Trunk:
    def __init__(self, image, n):        
        self.image = image
        self.angles_change = np.zeros((n,))
    
    def create_result_image(self, name = None):
        trunk_image = self.create_trunk_mask()
        res = np.repeat(np.expand_dims(self.image, axis = 2), 3, axis = 2)
        res[:, :, 2] = 0.5 * res[:, :, 2] + 0.5 * trunk_image        
        return res
    
    def create_trunk_mask(self, angles_change = None):
        if angles_change is None:
            trunk_image, _ = create_trunk_image(self.angles_change)
        else:
             trunk_image, _ = create_trunk_image(angles_change)
        return trunk_image
    
    def find_gradient(self, epsilon = 1e-2):
        trunk_mask = self.create_trunk_mask()
        u_0 = np.sum(np.multiply(self.image, trunk_mask))
        n = self.angles_change.shape[0]
        u_array = np.zeros((n,))
        gradients = np.zeros((n,))
        for k in range(n):
            new_angles = np.copy(self.angles_change)
            new_angles[k] += epsilon
            new_trunk_mask = self.create_trunk_mask(new_angles)
            u_array[k] = np.sum(np.multiply(new_trunk_mask, self.image))
            gradients[k] = (u_array[k] - u_0) / epsilon
        
        return gradients
    
    def fit(self, iterations, step_size = 1e-6):
        
        best_u = 0
        best_angles_change = None
        
        for k in tqdm(range(iterations)):
            gradients = self.find_gradient()
            self.angles_change = self.angles_change + gradients * step_size
            trunk_mask = self.create_trunk_mask()
            u = np.sum(np.multiply(self.image, trunk_mask))
            if u > best_u:
                best_u = u
                best_angles_change = np.copy(self.angles_change)            
            
        self.angles_change = np.copy(best_angles_change)


n = 20
image_indices = [3, 4, 5, 6, 7,
                 11, 14, 15, 17, 18,
                 20, 21, 22, 24, 25,
                 27, 28, 29, 46, 47,
                 48, 49, 50, 51, 56,
                 58, 59, 60, 62, 63,
                 67, 68, 69, 70, 71,
                 72, 73, 74]
image_names = ['{}.png'.format(idx) for  idx in image_indices]

for idx in range(len(image_names)):

    image_name = image_names[idx]
    print(image_name)
    image = load_image(image_name)
    
    trunk = Trunk(image, n)
    trunk.angles_change = np.loadtxt("fit trunk/best angles/{}.txt".format(image_name[:-4]))
    
    trunk.fit(1000)
    result_image = trunk.create_result_image()
    cv2.imwrite("fit trunk/{}".format(image_name), result_image * 255)
    np.savetxt("fit trunk/best angles/{}.txt".format(image_name[:-4]), trunk.angles_change)




