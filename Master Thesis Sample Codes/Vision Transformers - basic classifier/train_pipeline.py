# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 13:40:15 2021

@author: Amirhossein
"""
import torch
from neural_network import VisionTransormer
from dataset import create_datasets
from torch.utils.data import DataLoader
from train_functions import train
import numpy as np
import os



""" ------------------------------- Config ------------------------------- """


image_shape = (224, 224)
dataset_path = "D:/university/bioinformatics/master project/five class codes/prepare dataset/dataset"

BATCH_SIZE = 16
learning_rate = 1e-3

DATA_LOADER_KWARGS = {
    'batch_size': BATCH_SIZE,
    'shuffle': True,
    'pin_memory': True
}

np.random.seed(0)

save_results_folder = "results/"


def create_model_name(backbone_name, last_layers_number):
    model_name = "{}__{}.pt".format(backbone_name, last_layers_number)
    return model_name

""" ----------------------------- data loader ---------------------------- """

train_dataset, val_dataset, test_dataset = create_datasets(dataset_path, image_shape)

train_loader = DataLoader(train_dataset,
                          **DATA_LOADER_KWARGS)
val_loader = DataLoader(val_dataset,
                        **DATA_LOADER_KWARGS)


""" --------------------------- Neural Network --------------------------- """


num_classes = 5
backbone_name = 'vit_base_patch16_224'
last_layers_number = 4

network_input_shape = (3,) + image_shape
print('Input Shape =', network_input_shape)
print('Num of Classes =', num_classes)


model_name = create_model_name(backbone_name, last_layers_number)

model = VisionTransormer(backbone_name, num_classes, last_layers_number)

save_folder = save_results_folder + model_name[:-3]
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

model.summary(network_input_shape)

""" ---------------------------- Train Process ---------------------------- """

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

train(model,
      train_loader,
      val_loader,
      optimizer,
      num_epochs = 100,
      device = device,
      save_folder = save_folder,
      model_name = model_name)


