# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 13:40:15 2021

@author: Amirhossein
"""
import torch
from leukemia_network import LeukemiaNet
from meta import create_model_name
from cell_dataset import CellDataset
from train_functions import train
import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler



""" ------------------------------- Config ------------------------------- """


dataset_path = "D:/university/bioinformatics/master project/dataset/ALL-IDB/ALL_IDB1/im"
centroids_path = "D:/university/bioinformatics/master project/codes/codes - object detection/two stage - CNN - Transformer - gray scale/ALL IDB 1 results/Middle Results/Candidate Centroids/"

batch_size = 8
learning_rate = 1e-3



np.random.seed(0)

save_results_folder = "results/"
if not os.path.exists(save_results_folder):
    os.mkdir(save_results_folder)


""" ----------------------------- data loader ---------------------------- """

train_val_ratio = 0.7

crop_shape = (250, 250)
dataset_output_shape = (256, 256)
dataset = CellDataset(dataset_path, centroids_path, crop_shape, dataset_output_shape)

num_train = len(dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(train_val_ratio * num_train))

train_index, val_index = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_index)
val_sampler = SubsetRandomSampler(val_index)

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size = batch_size,
                                           sampler = train_sampler)


val_loader = torch.utils.data.DataLoader(dataset,
                                         batch_size = batch_size,
                                         sampler = val_sampler)




""" --------------------------- Neural Network --------------------------- """

num_classes = 2
lstm_nodes = 256
final_fcs = [64]
fc_drop_rate = 0.5
    
network_input_shape = (3,) + dataset_output_shape

print('Input Shape =', network_input_shape)
print('Num of Classes =', num_classes)


model_name = create_model_name(num_classes,
                               lstm_nodes,
                               final_fcs,
                               fc_drop_rate)

model = LeukemiaNet(network_input_shape,
                    num_classes,
                    lstm_nodes,
                    final_fcs,
                    fc_drop_rate)


save_folder = save_results_folder + model_name[:-3]
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

#model.get_summary(network_input_shape, "cuda")

""" ---------------------------- Train Process ---------------------------- """

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train(model,
      train_loader,
      val_loader,
      optimizer,
      num_epochs = 100,
      device = device,
      save_folder = save_folder,
      model_name = model_name)


