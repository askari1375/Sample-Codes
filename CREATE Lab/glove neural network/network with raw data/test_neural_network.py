# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:10:59 2021

@author: Amirhossein
"""

import torch
import numpy as np
from data_plotter import DataPlotter
from neural_network import Network
from glove_dataset import GloveDataset
import os



class NetworkEvaluator():
    def __init__(self, fc_layers):
        save_results_folder = "results/"
        data_path = "data"
        self.glove_dataset = GloveDataset(data_path)
        self.fc_layers = fc_layers
        
        self.model = Network(fc_layers, input_nodes = 16, output_nodes = 63, drop_rate = 0.5)
        model_name = self.create_model_name()
        model_path = os.path.join(save_results_folder, model_name[:-3], model_name)
        self.model.load_state_dict(torch.load(model_path))
        
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.data_plotter = DataPlotter()
        
    def create_model_name(self):
        model_name = "network_"
        for fc in self.fc_layers:
            model_name += "_{}".format(fc)
        model_name += ".pt"
        return model_name
    
    def get_prediction(self, idx):
        x, y = self.glove_dataset.__getitem__(idx)
        x = torch.unsqueeze(x, dim = 0).to(self.device)
        y = torch.unsqueeze(y, dim = 0).to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(x)
        y_pred = y_pred.to(torch.device("cpu")).numpy()
        
        y_pred = y_pred.reshape((21, 3))
        y = y.to(torch.device("cpu")).numpy().reshape((21, 3))
        
        return y_pred, y
    
    def plot_result(self, idx):
        y_pred, y = self.get_prediction(idx)
        a = np.expand_dims(y, axis = 0)
        b = np.expand_dims(y_pred, axis = 0)
        c = np.concatenate((a, b), axis = 0)            
        self.data_plotter.plot_hands(c)
        



fc_layers = [20, 20]

network_evaluator = NetworkEvaluator(fc_layers)


for idx in range(200):
    network_evaluator.plot_result(idx)







