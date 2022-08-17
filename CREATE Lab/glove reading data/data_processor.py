# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:56:09 2021

@author: Amirhossein
"""

import numpy as np
import os


def load_data_segment(k, time_diff_threshold):
    
    
    glove_data = np.load("captured data/glove data {}.npy".format(k))
    camera_data = np.load("captured data/camera data {}.npy".format(k))
    glove_times = np.load("captured data/glove times {}.npy".format(k))
    camera_times = np.load("captured data/camera times {}.npy".format(k))
    
    
    
    time_diff = np.abs(glove_times - camera_times)
    
    glove_data = glove_data[time_diff < time_diff_threshold, ...].astype("float64")
    camera_data = camera_data[time_diff < time_diff_threshold, ...]
    glove_times = glove_times[time_diff < time_diff_threshold]
    camera_times = camera_times[time_diff < time_diff_threshold]
    
    return glove_data, camera_data, glove_times, camera_times


def load_data(time_diff_threshold):
    file_names = os.listdir("captured data")
    file_names = [name for name in file_names if name[:11] == "camera data"]
    file_names.sort()
    name = file_names[-1]
    last_file_number = int(name.split(" ")[-1].split(".")[0])
    
    data = load_data_segment(1, time_diff_threshold)
    
    glove_data = data[0]
    camera_data = data[1]
    glove_times = data[2]
    camera_times = data[3]
    
    data_part_length = [glove_data.shape[0]]
    
    for k in range(1, last_file_number):
        
        data = load_data_segment(k + 1, time_diff_threshold)
        
        new_glove_data = data[0]
        new_camera_data = data[1]
        new_glove_times = data[2]
        new_camera_times = data[3]
                
        glove_data = np.concatenate((glove_data, new_glove_data))
        camera_data = np.concatenate((camera_data, new_camera_data))
        glove_times = np.concatenate((glove_times, new_glove_times))
        camera_times = np.concatenate((camera_times, new_camera_times))
        
        data_part_length.append(new_glove_data.shape[0])
    
    data_part_length = np.array(data_part_length)
    
    return glove_data, camera_data, glove_times, camera_times, data_part_length
    



time_diff_threshold = 0.02

data = load_data(time_diff_threshold)

glove_data = data[0]
camera_data = data[1]
glove_times = data[2]
camera_times = data[3]
data_part_length  = data[4]

correct_rows = np.max(glove_data, 1) < 100

glove_data = glove_data[correct_rows, :]
camera_data = camera_data[correct_rows, :]
glove_times = glove_times[correct_rows]
camera_times = camera_times[correct_rows]


np.save("processed data/glove data.npy", glove_data)
np.save("processed data/camera data.npy", camera_data)
np.save("processed data/glove times.npy", glove_times)
np.save("processed data/camera times.npy", camera_times)



