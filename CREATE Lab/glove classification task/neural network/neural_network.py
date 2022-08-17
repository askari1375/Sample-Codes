# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:09:00 2021

@author: Amirhossein
"""

import torch.nn as nn
from torchsummary import summary as torch_summary


class Network(nn.Module):
    def __init__(self, fcs, input_nodes, num_classes, drop_rate = None):
        
        super(Network, self).__init__()
        
        fc_layers = []
        last_nodes_number = input_nodes
        
        for nodes_number in fcs:
            fc_layers.append(nn.Linear(last_nodes_number, nodes_number))
            last_nodes_number = nodes_number
            fc_layers.append(nn.BatchNorm1d(last_nodes_number))
            fc_layers.append(nn.ReLU())
            if drop_rate is not None:
                fc_layers.append(nn.Dropout(drop_rate))
            
        fc_layers.append(nn.Linear(last_nodes_number, num_classes))
        fc_layers.append(nn.Softmax())
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.fc_layers(x)
    
    def get_loss(self, output, target):
        loss = self.cross_entropy(output, target)
        return loss
    
    def summary(self, input_size):
        torch_summary(self, input_size)
        
            
        