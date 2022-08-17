# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:53:00 2021

@author: Amirhossein
"""


import numpy as np
import os



def load_data_segment(k):
    
    
    glove_data = np.load("captured data/glove data {}.npy".format(k))
    camera_data = np.load("captured data/camera data {}.npy".format(k))
    glove_times = np.load("captured data/glove times {}.npy".format(k))
    camera_times = np.load("captured data/camera times {}.npy".format(k))
    
    print(glove_data.shape)
    print(camera_data.shape)
    print(glove_times.shape)
    print(camera_times.shape)
    
    glove_data_temp = []
    for x in glove_data:
        if len(x) == 16:
            glove_data_temp.append(x)
    glove_data = np.array(glove_data_temp)
    
    print(glove_data.shape)


load_data_segment(6)