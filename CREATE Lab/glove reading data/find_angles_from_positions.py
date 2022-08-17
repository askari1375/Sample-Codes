# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:04:48 2021

@author: Amirhossein
"""

import numpy as np
from data_plotter import DataPlotter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def position_to_angle(position_data):
    connections = [(1, 2, 3, 4),
               (5, 6, 7, 8),
               (9, 10, 11, 12),
               (13, 14, 15, 16),
               (17, 18, 19, 20)]
    
    angles = np.zeros((10,))
    c = 0
    for finger in connections:
        for k in range(1, 3):
            p1 = position_data[finger[k - 1], :]
            p2 = position_data[finger[k], :]
            p3 = position_data[finger[k + 1], :]
            
            angles[c] = find_angle(p1, p2, p3)
            c += 1
    
    return angles


def angle_to_position(angles):
    
    positions = np.zeros((20, 3))
    
    for k in range(5):
        alpha_1 = angles[2 * k]
        alpha_2 = angles[2 * k + 1]
        
        positions[4*k : 4*k + 4, 2] = k
        
        y1 = 1
        y2 = 1 - np.cos(alpha_1)
        y3 = y2 + np.cos(alpha_1 + alpha_2)
        x1 = 0
        x2 = np.sin(alpha_1)
        x3 = x2 - np.sin(alpha_1 + alpha_2) 
        
        
        positions[4*k + 1, :2] = np.array([x1, y1])
        positions[4*k + 2, :2] = np.array([x2, y2])
        positions[4*k + 3, :2] = np.array([x3, y3])
        
    return positions
        


def find_angle(p1, p2, p3):
    v_1 = p1 - p2
    v_2 = p3 - p2
    
    vec_dot = np.dot(v_1, v_2)
    cos_alpha = vec_dot / (np.linalg.norm(v_1) * np.linalg.norm(v_2))
    alpha = np.arccos(cos_alpha)
    
    return alpha




all_raw_camera_data = np.load("processed data type 2 validation/camera data.npy")    
angles = np.zeros((all_raw_camera_data.shape[0], 10))

for k in range(all_raw_camera_data.shape[0]):
    angles[k, :] = position_to_angle(all_raw_camera_data[k, ...])



np.save("processed data type 2 validation angles/angles data.npy", angles)

idx = 0
raw_positions = all_raw_camera_data[idx, ...]
positions = angle_to_position(angles[idx, :])


data_plotter = DataPlotter()
data_plotter.plot_hands(np.expand_dims(raw_positions, axis = 0))


figure = plt.figure(figsize = (10, 10))
ax = Axes3D(figure, adjustable = 'datalim')
connections = [(0, 1, 2, 3),
               (4, 5, 6, 7),
               (8, 9, 10, 11),
               (12, 13, 14, 15),
               (16, 17, 18, 19)]

for finger in connections:
    f = np.zeros((4, 3))
    for k in range(len(finger)):
        f[k, :] = positions[finger[k], :]
    ax.plot(f[:, 0], f[:, 1], f[:, 2], marker = 'o', color = 'r')



