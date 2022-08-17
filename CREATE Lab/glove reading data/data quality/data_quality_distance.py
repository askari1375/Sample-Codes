# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:14:32 2021

@author: Amirhossein
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt



def find_pairwise_distance(a):
    
    start_time = time.time()
    b = a.reshape(a.shape[0], 1, a.shape[1])
    c = np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))
    print("process time :\t{}".format(time.time() - start_time))
    return c



data_limit = 3000
folder_path = "data"

type_names = ['type 1', 'type 2', 'denoised']


for k_nn in [10, 20, 50, 100]:
    for idx, name in enumerate(type_names):
        
        angles_data = np.load(os.path.join(folder_path, "angles data {}.npy".format(name)))
        glove_data = np.load(os.path.join(folder_path, "glove data {}.npy".format(name)))
        
        correct_rows = np.max(glove_data, 1) < 100
        glove_data = glove_data[correct_rows, :]
        angles_data = angles_data[correct_rows, :]
        glove_data = glove_data / 99
        
        """ ------------------------------------------- """
        glove_data = glove_data[:data_limit, :]
        angles_data = angles_data[:data_limit, :]
        """ ------------------------------------------- """
        
        
        glove_dist = find_pairwise_distance(glove_data)
        angles_dist = find_pairwise_distance(angles_data)
        
        n = glove_dist.shape[0]
        
        for k in range(n):
            glove_dist[k, k] = np.inf
            angles_dist[k, k] = np.inf
        
        quality = np.zeros((n,))
        
        for k in range(n):
            indices_glove = np.argpartition(glove_dist[k, :], k_nn)[:k_nn]
            indices_angles = np.argpartition(angles_dist[k, :], k_nn)[:k_nn]
            common_values = np.intersect1d(indices_glove, indices_angles)
            quality[k] = common_values.shape[0] / k_nn
        
        
        fig = plt.figure(figsize = (10, 10))
        plt.hist(quality, bins = 10, density = True)
        plt.xlim(0, 1)
        plt.ylim(0, 5)
        fig.savefig("{} histogram for {} {}.png".format(k_nn, idx, name))

