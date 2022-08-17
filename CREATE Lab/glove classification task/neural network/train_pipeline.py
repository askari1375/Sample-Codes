# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 18:50:46 2021

@author: Amirhossein
"""

import torch
from neural_network import Network
from torch.utils.data import DataLoader
from glove_dataset import GloveDataset
from train_functions import train
import numpy as np
import os


""" ------------------------------- Config ------------------------------- """

BATCH_SIZE = 16
learning_rate = 1e-3

DATA_LOADER_KWARGS = {
    'batch_size': BATCH_SIZE,
    'shuffle': True,
    'pin_memory': True
}

np.random.seed(0)

save_results_folder = "results/"


def create_model_name(fc_layers):
    model_name = "network_"
    for fc in fc_layers:
        model_name += "_{}".format(fc)
    model_name += ".pt"
    return model_name


""" ----------------------------- data loader ----------------------------- """


train_dataset = GloveDataset('train')
val_dataset = GloveDataset('validation')


train_loader = DataLoader(train_dataset,
                          **DATA_LOADER_KWARGS)
val_loader = DataLoader(val_dataset,
                        **DATA_LOADER_KWARGS)

""" --------------------------- Neural Network --------------------------- """


fc_layers = [64, 64, 8]

model = Network(fc_layers, input_nodes = 16, num_classes = 6, drop_rate = 0.5)
model_name = create_model_name(fc_layers)

save_folder = save_results_folder + model_name[:-3]
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

model.summary((16,))

""" ---------------------------- Train Process ---------------------------- """

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

train(model,
      train_loader,
      val_loader,
      optimizer,
      num_epochs = 500,
      device = device,
      save_folder = save_folder,
      model_name = model_name)

