# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:10:59 2021

@author: Amirhossein
"""

import torch
import numpy as np
from neural_network import Network
from glove_dataset import GloveDataset
import os
from tqdm import tqdm


class NetworkEvaluator():
    def __init__(self, fc_layers):
        save_results_folder = "results/"
        self.fc_layers = fc_layers
        
        self.model = Network(fc_layers, input_nodes = 16, num_classes = 6, drop_rate = 0.5)
        model_name = self.create_model_name()
        self.final_folder = os.path.join(save_results_folder, model_name[:-3])
        model_path = os.path.join(self.final_folder, model_name)
        self.model.load_state_dict(torch.load(model_path))
        
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        
    def create_model_name(self):
        model_name = "network_"
        for fc in self.fc_layers:
            model_name += "_{}".format(fc)
        model_name += ".pt"
        return model_name
    
    def get_prediction(self, idx, dataset):
        x, y = dataset.__getitem__(idx)
        x = torch.unsqueeze(x, dim = 0).to(self.device)
        y = torch.unsqueeze(y, dim = 0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
        outputs = outputs.to(torch.device("cpu")).numpy()
        y_pred = np.argmax(outputs, axis=-1)
        
        y_true = y.to(torch.device("cpu")).numpy().reshape((-1))
        
        return y_pred, y_true
    
    def find_accuracy(self, data_phase):
        
        dataset = GloveDataset(data_phase)
        n = len(dataset)

        y_pred = np.zeros((n,))
        y_true = np.zeros((n,))
        
        for idx in tqdm(range(n)):
             y_pred[idx], y_true[idx] = self.get_prediction(idx, dataset)
        
        accuracy = np.sum(y_pred == y_true) / n
        
        return accuracy
    




fc_layers = [1000]


network_evaluator = NetworkEvaluator(fc_layers)
train_accuracy = network_evaluator.find_accuracy('train')
validation_accuracy = network_evaluator.find_accuracy('validation')
test_accuracy = network_evaluator.find_accuracy('test')

report = ""
report += "train accuracy\t:\t{}\n".format(train_accuracy)
report += "validation accuracy\t:\t{}\n".format(validation_accuracy)
report += "test accuracy\t:\t{}".format(test_accuracy)

with open(network_evaluator.final_folder + "/accuracy.txt", 'w') as f:
    f.write(report)

print(report)

