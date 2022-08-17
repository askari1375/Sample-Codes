# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:35:40 2021

@author: Amirhossein
"""

import torch
import numpy as np
import os


class GloveDataset(torch.utils.data.Dataset):
    def __init__(self, phase):

        sub_folders = os.listdir("../data/data {}".format(phase))
        
        data_list = []
        labels_list = []
        for folder_name in sub_folders:
            file_path = "../data/data {}/{}/{}.npy".format(phase, folder_name, folder_name)
            data_list.append(np.load(file_path))
            labels_list.append(np.ones((data_list[-1].shape[0],)) * int(folder_name))
        
        X = np.concatenate(data_list, axis = 0)
        Y = np.concatenate(labels_list, axis = 0).astype('int32')
        
        self.X = torch.Tensor(X)
        self.Y = torch.from_numpy(Y)
            
    def __getitem__(self, idx):
        x = self.X[idx, ...]
        y = self.Y[idx]
        
        y = torch.tensor(y, dtype=torch.long)
        
        return x, y
        
    def __len__(self):
        return self.X.shape[0]


def main():

    glove_dataset = GloveDataset("train")
    
    x, y = glove_dataset.__getitem__(7500)
    print(x.shape)
    print(y.shape)
    print(y.numpy())
    
    print("length :")
    
    print(glove_dataset.X.shape)
    print(glove_dataset.__len__())

if __name__ == "__main__":
    main()