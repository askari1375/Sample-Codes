# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:34:15 2021

@author: Amirhossein
"""

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    
    def __init__(self, input_path, output_path, image_shape = (256, 256), size = -1):
        
        self.input_path = input_path
        self.output_path = output_path
        
        transforms = []
        transforms.append(torchvision.transforms.Resize(image_shape))
        transforms.append(torchvision.transforms.ToTensor())
        
        self.transforms = torchvision.transforms.Compose(transforms)
        
        self.input_image_names = list(os.walk(self.input_path))[0][2]
        self.output_image_names = list(os.walk(self.output_path))[0][2]
        
        self.size = size if size > 0 else len(self.input_image_names)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        input_img_loc = os.path.join(self.input_path, self.input_image_names[index])
        output_img_loc = os.path.join(self.output_path, self.output_image_names[index])
        
        x = self.transforms(Image.open(input_img_loc))
        y = self.transforms(Image.open(output_img_loc))
        
        return x, y



def main():
    input_path = "data/trunk without occlusion/train/input"
    output_path = "data/trunk without occlusion/train/output"
    map_dataset = ImageDataset(input_path, output_path, size = 500)
    
    print("Dataset length : {}".format(len(map_dataset)))
    x, y = map_dataset.__getitem__(5)
    print("type X : {}".format(type(x)))
    print("type Y : {}".format(type(y)))
    print("shape X : {}".format(x.shape))
    print("shape Y : {}".format(y.shape))
    
    """ ---------------------------------- """
    train_input_path = "data/trunk without occlusion/train/input"
    train_output_path = "data/trunk without occlusion/train/output"
    train_dataset = ImageDataset(train_input_path, train_output_path, size = 500)
    
    val_input_path = "data/trunk without occlusion/validation/input"
    val_output_path = "data/trunk without occlusion/validation/output"
    val_dataset = ImageDataset(val_input_path, val_output_path, size = 200)
    
    BATCH_SIZE = 16
    DATA_LOADER_KWARGS = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'pin_memory': True
    }
    
    train_loader = DataLoader(train_dataset, **DATA_LOADER_KWARGS)
    val_loader = DataLoader(val_dataset, **DATA_LOADER_KWARGS)
    


if __name__ == "__main__":
    main()
        