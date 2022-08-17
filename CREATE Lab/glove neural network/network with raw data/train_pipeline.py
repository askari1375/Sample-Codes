# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 18:50:46 2021

@author: Amirhossein
"""

import torch
from neural_network import Network
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from glove_dataset import GloveDataset
from train_functions import train
import numpy as np
import os


""" ------------------------------- Config ------------------------------- """

BATCH_SIZE = 16
val_ratio = 0.3
learning_rate = 1e-3

DATA_LOADER_KWARGS = {
    'batch_size': BATCH_SIZE,
    'pin_memory': True
}


save_results_folder = "results/"
data_path = "data"


def create_model_name(fc_layers):
    model_name = "network_"
    for fc in fc_layers:
        model_name += "_{}".format(fc)
    model_name += ".pt"
    return model_name


""" ----------------------------- data loader ----------------------------- """


glove_dataset = GloveDataset(data_path)

num_data = len(glove_dataset)
indices = list(range(num_data))
split = int(np.floor((1 - val_ratio) * num_data))

train_index, val_index = indices[split:], indices[:split]
np.random.shuffle(train_index)
np.random.shuffle(val_index)

train_sampler = SubsetRandomSampler(train_index)
val_sampler = SubsetRandomSampler(val_index)

train_loader = DataLoader(glove_dataset,
                          sampler = train_sampler,
                          **DATA_LOADER_KWARGS)
val_loader = DataLoader(glove_dataset,
                        sampler = val_sampler,
                        **DATA_LOADER_KWARGS)

""" --------------------------- Neural Network --------------------------- """


fc_layers = [20, 20]

model = Network(fc_layers, input_nodes = 16, output_nodes = 63, drop_rate = 0.5)
model_name = create_model_name(fc_layers)

save_folder = save_results_folder + model_name[:-3]
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

model.summary((16,))

""" ---------------------------- Train Process ---------------------------- """

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train(model,
      train_loader,
      val_loader,
      optimizer,
      num_epochs = 100,
      device = device,
      save_folder = save_folder,
      model_name = model_name,
      loss_plot_limit = 0.5)
