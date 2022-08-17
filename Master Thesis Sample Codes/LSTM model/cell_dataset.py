# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 21:51:26 2022

@author: Amirhossein
"""

import torch
import numpy as np
import os
from PIL import Image
from skimage.transform import resize
from skimage import exposure


class CellDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset_path, centroids_path, crop_shape, model_input_shape):
        
        self.dataset_path = dataset_path
        self.centroids_path = centroids_path
        self.image_names = os.listdir(dataset_path)
        self.crop_shape = crop_shape
        self.model_input_shape = model_input_shape
        
        self.max_candidate_cells = self.find_max_candidate_cells()
        
        
    def find_max_candidate_cells(self):
        res = 0
        for image_name in self.image_names:
            centroids = np.load(os.path.join(self.centroids_path + image_name[:-4] + ".npy"))
            res = max(res, centroids.shape[0])

        return res
    
    
    def __getitem__(self, idx):
        
        image_name = self.image_names[idx]
        img = Image.open(os.path.join(self.dataset_path, image_name))
        img = np.array(img) / 255
        
        label = int(image_name[-5])
        
        centroids = np.load(os.path.join(self.centroids_path + image_name[:-4] + ".npy"))
        
        pad_x = np.zeros((self.crop_shape[0], img.shape[1], 3))
        pad_image = np.concatenate((pad_x, img, pad_x), axis = 0)
        pad_y = np.zeros((pad_image.shape[0], self.crop_shape[1], 3))
        pad_image = np.concatenate((pad_y, pad_image, pad_y), axis = 1)
        
        cell_images = np.zeros((self.max_candidate_cells, 3) + self.model_input_shape)
    
        for k in range(centroids.shape[0]):
            x1 = int(centroids[k, 0] - self.crop_shape[0] / 2) + self.crop_shape[0]
            x2 = int(centroids[k, 0] + self.crop_shape[0] / 2) + self.crop_shape[0]
            y1 = int(centroids[k, 1] - self.crop_shape[1] / 2) + self.crop_shape[1]
            y2 = int(centroids[k, 1] + self.crop_shape[1] / 2) + self.crop_shape[1]
            
            patch = pad_image[x1:x2, y1:y2, :]
            patch = resize(patch, self.model_input_shape)
            
            patch = exposure.equalize_hist(patch)
            
            patch = np.moveaxis(patch, 2, 0)
            
            cell_images[k, ...] = patch
        
        
        shuffled_indices = list(range(self.max_candidate_cells))
        np.random.shuffle(shuffled_indices)
        cell_images = cell_images[shuffled_indices, ...]
        
        cell_images = torch.Tensor(cell_images)
        
        return cell_images, label
    
    def __len__(self):
        return len(self.image_names)
    

def main():

    dataset_path = "D:/university/bioinformatics/master project/dataset/ALL-IDB/ALL_IDB1/im"
    centroids_path = "D:/university/bioinformatics/master project/codes/codes - object detection/two stage - CNN - Transformer - gray scale/ALL IDB 1 results/Middle Results/Candidate Centroids/"
    
    crop_shape = (250, 250)
    model_input_shape = (256, 256)
    
    dataset = CellDataset(dataset_path, centroids_path, crop_shape, model_input_shape)
    
    print(len(dataset))
    x, y = dataset[2]
    print(x.shape)
    print(y)

if __name__ == "__main__":
    main()
