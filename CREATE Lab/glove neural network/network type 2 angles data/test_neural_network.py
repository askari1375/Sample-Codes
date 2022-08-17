# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:10:59 2021

@author: Amirhossein
"""

import torch
import numpy as np
from neural_network import Network
from glove_angles_dataset import GloveDataset
import os
from data_plotter import AnglesDataPlotter
import matplotlib.pyplot as plt


class NetworkEvaluator():
    def __init__(self, fc_layers):
        save_results_folder = "results/"
        data_path = "data"
        self.glove_dataset = GloveDataset(data_path)
        self.fc_layers = fc_layers
        
        self.model = Network(fc_layers, input_nodes = 16, output_nodes = 10, drop_rate = 0.5)
        model_name = self.create_model_name()
        model_path = os.path.join(save_results_folder, model_name[:-3], model_name)
        self.model.load_state_dict(torch.load(model_path))
        
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.data_plotter = AnglesDataPlotter()
        
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
        y_pred = y_pred.to(torch.device("cpu")).numpy().reshape((-1))
        
        y = y.to(torch.device("cpu")).numpy().reshape((-1))
        
        return y_pred, y
    
    
    @staticmethod
    def angle_to_position(angles):
    
        positions = np.zeros((20, 3))
        
        for k in range(5):
            alpha_1 = angles[2 * k]
            alpha_2 = angles[2 * k + 1]
            
            positions[4*k : 4*k + 4, 2] = k
            
            y1 = 1
            y2 = 1 - np.cos(alpha_1)
            y3 = y2 + np.cos(alpha_1 + alpha_2)
            x1 = 0
            x2 = np.sin(alpha_1)
            x3 = x2 - np.sin(alpha_1 + alpha_2) 
            
            
            positions[4*k + 1, :2] = np.array([x1, y1])
            positions[4*k + 2, :2] = np.array([x2, y2])
            positions[4*k + 3, :2] = np.array([x3, y3])
            
        return positions
    
    
    def plot_result(self, idx):
        y_pred, y = self.get_prediction(idx)
        
        y_pred_positions = self.angle_to_position(y_pred)
        y_positions = self.angle_to_position(y)
        
        plot_data = np.concatenate((np.expand_dims(y_positions, axis = 0),
                                    np.expand_dims(y_pred_positions, axis = 0)),
                                   axis=0)
        
        #self.data_plotter.plot_hands(plot_data)
        return y_pred, y_pred_positions



fc_layers = [16, 8, 8, 10]

network_evaluator = NetworkEvaluator(fc_layers)

angles = []
positions = []

for idx in range(2942):
     a, p = network_evaluator.plot_result(idx)
     angles.append(a)
     positions.append(p)

angles = np.array(angles)
positions = np.array(positions)

for k in range(angles.shape[1]):
    plt.plot(angles[:, k])




