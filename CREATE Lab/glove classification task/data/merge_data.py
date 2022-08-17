# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:42:55 2021

@author: Amirhossein
"""

import numpy as np
import os



for phase in ['train', 'validation', 'test']:

    raw_data_path = "raw data {}".format(phase)
    
    sub_folders = os.listdir(raw_data_path)
    
    if not os.path.exists("data {}".format(phase)):
        os.mkdir("data {}".format(phase))
    for sub_folder in sub_folders:
        data = None
        np_file_names = os.listdir("raw data {}/{}".format(phase, sub_folder))
        for name in np_file_names:
            new_data = np.load("raw data {}/{}/{}".format(phase, sub_folder, name))
            if data is None:
                data = new_data
            else:
                data = np.concatenate((data, new_data), axis = 0)
    
        print("------------------\t{} - {}\t------------------".format(phase, sub_folder))
        print(data.shape)
        if not os.path.exists("data {}/{}".format(phase, sub_folder)):
            os.mkdir("data {}/{}".format(phase, sub_folder))
        save_path = "data {}/{}/{}.npy".format(phase, sub_folder, sub_folder)
        np.save(save_path, data)
