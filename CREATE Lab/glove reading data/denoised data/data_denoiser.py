# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 12:29:28 2021

@author: Amirhossein
"""

import numpy as np
import os


def find_mean(data_list):
    n = len(data_list)
    m = data_list[0].shape[0]
    
    data = np.zeros((n, m))
    for k in range(n):
        data[k, :] = data_list[k]
    
    return np.mean(data, axis = 0)


group_size = 3
time_threshold = 0.5

data_type = 2
folder_path = "../processed data type {} angles".format(data_type)


angles_data = np.load(os.path.join(folder_path, "angles data.npy"))
glove_data = np.load(os.path.join(folder_path, "glove data.npy"))
times = np.load(os.path.join(folder_path, "camera times.npy"))

print(angles_data.shape)
print(glove_data.shape)
print(times.shape)




n = times.shape[0]

new_angles = []
new_glove_data = []
new_times = []

temp_angles = []
temp_glove_data = []
temp_times = []

for k in range(n):
    if k > 0:   
        if times[k] - times[k - 1] < time_threshold:
            if len(temp_times) == group_size:
                new_angles.append(find_mean(temp_angles))
                new_glove_data.append(find_mean(temp_glove_data))
                new_times.append(find_mean(temp_times))
                
                temp_angles = []
                temp_glove_data = []
                temp_times = []
        else:            
            new_angles.append(find_mean(temp_angles))
            new_glove_data.append(find_mean(temp_glove_data))
            new_times.append(find_mean(temp_times))
            
            
            temp_angles = []
            temp_glove_data = []
            temp_times = []
    
    temp_angles.append(angles_data[k, :])
    temp_glove_data.append(glove_data[k, :])
    temp_times.append(np.array([times[k]]))
    


np.save("denoised angles.npy", np.array(new_angles))
np.save("denoised glove data.npy", np.array(new_glove_data))
np.save("denoised times.npy", np.array(new_times))

