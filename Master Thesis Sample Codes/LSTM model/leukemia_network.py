# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 23:00:22 2022

@author: Amirhossein
"""

import torch
import torch.nn as nn
import torchvision
from cell_dataset import CellDataset
import time
from meta import create_model_name


class LeukemiaNet(nn.Module):
    
    def __init__(self,
                 input_shape,
                 num_classes,
                 lstm_nodes,
                 final_fcs,
                 fc_drop_rate):
        
        super().__init__()
        alexnet_pretrained = torchvision.models.alexnet(pretrained=True)
        self.conv_layers = alexnet_pretrained.features
        
        feature_extractor_parameters = list(self.conv_layers.parameters())
        for param in feature_extractor_parameters:
            param.requires_grad = False
        
        self.flatten = nn.Flatten()
        
        sample_input = torch.randn(1, *input_shape)
        conv_layers_output_shape = self.conv_layers(sample_input).view(1, -1).shape[1]
        
        
        self.lstm = nn.LSTM(conv_layers_output_shape, lstm_nodes, batch_first = True)
        
        
        lstm_output_shape = lstm_nodes
        self.final_fc_block = FullyConnectedBlock(lstm_output_shape,
                                                  num_classes,
                                                  final_fcs,
                                                  fc_drop_rate,
                                                  add_softmax = True)
        
        self.cross_entropy = nn.CrossEntropyLoss()
        
    
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        x = x.view(batch_size * seq_len, x.shape[2], x.shape[3], x.shape[4])
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = x.view(batch_size, seq_len, -1)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.final_fc_block(x)
        
        return x
    
    def get_loss(self, outputs, targets):
        loss = self.cross_entropy(outputs, targets)
        return loss




class FullyConnectedBlock(nn.Module):
    
    def __init__(self, in_features,
                 out_features,
                 hidden_nodes,
                 fc_drop_rate,
                 add_softmax = False):
        
        super(FullyConnectedBlock, self).__init__()
        
        fc_layers = []
        
        last_nodes_number = in_features
        for nodes_number in hidden_nodes:
            fc_layers.append(nn.Linear(last_nodes_number, nodes_number))
            last_nodes_number = nodes_number
            fc_layers.append(nn.ReLU())
            if fc_drop_rate is not None:
                fc_layers.append(nn.Dropout(fc_drop_rate))
        
        fc_layers.append(nn.Linear(last_nodes_number, out_features))
        if add_softmax:
            fc_layers.append(nn.Softmax())
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
    
    def forward(self, x):
        
        output = self.fc_layers(x)
        return output


def main():

    model_input_shape = (3, 256, 256)
    num_classes = 2
    lstm_nodes = 256
    final_fcs = [64]
    fc_drop_rate = 0.5
    
    dataset_path = "D:/university/bioinformatics/master project/dataset/ALL-IDB/ALL_IDB1/im"
    centroids_path = "D:/university/bioinformatics/master project/codes/codes - object detection/two stage - CNN - Transformer - gray scale/ALL IDB 1 results/Middle Results/Candidate Centroids/"
    
    crop_shape = (250, 250)
    
    dataset_output_shape = (model_input_shape[1], model_input_shape[2])
    
    
    dataset = CellDataset(dataset_path, centroids_path, crop_shape, dataset_output_shape)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 8, shuffle = True)
    
    
    model = LeukemiaNet(model_input_shape,
                        num_classes,
                        lstm_nodes,
                        final_fcs,
                        fc_drop_rate)
    
    
    
    
    
    
    start_time = time.time()
    x, y = next(iter(data_loader))
    print("Load Batch Time : {}".format(time.time() - start_time))
    
    
    start_time = time.time()
    y_pred = model(x)
    print("process Batch Time : {}".format(time.time() - start_time))
    
    print(x.shape)
    print(y.shape)
    print(y_pred.shape)
    
    
    model_name = create_model_name(num_classes,
                                   lstm_nodes,
                                   final_fcs,
                                   fc_drop_rate)
    
    print(model_name)


if __name__ == "__main__":
    main()
