# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:35:40 2021

@author: Amirhossein
"""

import torch
import numpy as np
import os


class GloveDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
            self.glove_data = np.load(os.path.join(data_path, "glove data.npy"))
            self.camera_data = np.load(os.path.join(data_path, "camera data.npy"))            
            
    def __getitem__(self, idx):
        x = self.glove_data[idx, ...]
        y = self.camera_data[idx, ...]
        
        normalization_parameter = np.linalg.norm(y[0, :] - y[5, :])        
        y = y / normalization_parameter
                
        x = torch.Tensor(x.reshape((-1)))
        y = torch.Tensor(y.reshape((-1)))
        
        return x, y
        
    def __len__(self):
        return self.glove_data.shape[0]


def main():

    glove_dataset = GloveDataset("data")
    
    x, y = glove_dataset.__getitem__(5)
    print(x.shape)
    print(y.shape)
    
    print("length :")
    
    print(glove_dataset.glove_data.shape)
    print(glove_dataset.__len__())

if __name__ == "__main__":
    main()