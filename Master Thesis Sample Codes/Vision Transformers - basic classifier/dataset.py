# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 14:14:29 2021

@author: Amirhossein
"""
from torchvision.datasets import ImageFolder
import torchvision
import os


def create_datasets(dataset_path, image_shape):
    
    train_transforms = []
    val_transforms = []
    
    train_transforms.append(torchvision.transforms.Resize(image_shape))
    train_transforms.append(torchvision.transforms.RandomVerticalFlip())
    train_transforms.append(torchvision.transforms.RandomHorizontalFlip())
    train_transforms.append(torchvision.transforms.RandomRotation(180))
    train_transforms.append(torchvision.transforms.ToTensor())
    train_transforms = torchvision.transforms.Compose(train_transforms)
    
    val_transforms.append(torchvision.transforms.Resize(image_shape))
    val_transforms.append(torchvision.transforms.ToTensor())
    val_transforms = torchvision.transforms.Compose(val_transforms)
    
    train_dataset = ImageFolder(os.path.join(dataset_path, "train"),
                                transform = train_transforms)
    val_dataset = ImageFolder(os.path.join(dataset_path, "validation"),
                                transform = val_transforms)
    test_dataset = ImageFolder(os.path.join(dataset_path, "test"),
                                transform = val_transforms)
    
    return train_dataset, val_dataset, test_dataset


def main():
    
    image_shape = (256, 256)
    dataset_path = "D:/university/bioinformatics/master project/codes/dataset/train validation test"
    
    train_dataset, val_dataset, test_dataset = create_datasets(dataset_path, image_shape)
    print(train_dataset.class_to_idx)

if __name__ =="__main__":
    main()