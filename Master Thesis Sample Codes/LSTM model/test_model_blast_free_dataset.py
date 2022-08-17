# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:12:52 2021

@author: Amirhossein
"""

import torch
import numpy as np
from leukemia_network import LeukemiaNet
from cell_dataset import CellDataset
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from meta import create_model_name


class NetworkEvaluator():
    def __init__(self,
                 network_input_shape,
                 num_classes,
                 lstm_nodes,
                 final_fcs,
                 fc_drop_rate):
        
        
        save_results_folder = "results/"
        
        self.class_names = ['Normal', 'Cancer']
        
        self.network_input_shape = network_input_shape
        self.num_classes = num_classes
        self.lstm_nodes = lstm_nodes
        self.final_fcs = final_fcs
        self.final_fcs = final_fcs
        self.fc_drop_rate = fc_drop_rate
        
        self.model = LeukemiaNet(network_input_shape,
                                 num_classes,
                                 lstm_nodes,
                                 final_fcs,
                                 fc_drop_rate)
        
        model_name = self.create_model_name()
        self.final_folder = os.path.join(save_results_folder, model_name[:-3])
        model_path = os.path.join(self.final_folder, model_name)
        self.model.load_state_dict(torch.load(model_path))
        
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.features_save_path = os.path.join(self.final_folder, "extracted features")
        if not os.path.exists(self.features_save_path):
            os.mkdir(self.features_save_path)
        
        

    def create_model_name(self):
        model_name = create_model_name(self.num_classes,
                                       self.lstm_nodes,
                                       self.final_fcs,
                                       self.fc_drop_rate)
        return model_name
    
    def get_prediction(self, idx, dataset):
        x, y = dataset.__getitem__(idx)
        x = torch.unsqueeze(x, dim = 0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
        outputs = outputs.to(torch.device("cpu")).numpy()
        y_pred = np.argmax(outputs, axis=-1)
        
        y_true = 0
        
        return y_pred, y_true
    
    def find_performance(self, dataset):
        
        n = len(dataset)

        y_pred = np.zeros((n,))
        y_true = np.zeros((n,))
        
        for idx in tqdm(range(n)):
             y_pred[idx], y_true[idx] = self.get_prediction(idx, dataset)
             
        
        accuracy = np.sum(y_pred == y_true) / n
        cm_normalized = confusion_matrix(y_true, y_pred, normalize = 'true')
        disp_normalized = ConfusionMatrixDisplay(confusion_matrix = cm_normalized,
                                                 display_labels = self.class_names)
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm,
                                      display_labels = self.class_names)
       
        
        return accuracy, disp, disp_normalized
    
    
    
    
        
    

def save_displays(disps, labels, save_folder):
    n = len(disps)
    fig, axs = plt.subplots(1, n, figsize = (7 * n, 7))
    fig.suptitle("Confusion Matrices")
    for idx, disp in enumerate(disps):
        if n > 1:
            disp.plot(ax = axs[idx])
            axs[idx].set_title("{}".format(labels[idx]))
        else:
            disp.plot(ax = axs)
            axs.set_title("{}".format(labels[idx]))
    
    save_path = os.path.join(save_folder, "confusion matrix blast free.png")
    fig.savefig(save_path)
        
        
        



crop_shape = (250, 250)
dataset_output_shape = (256, 256)


num_classes = 2
lstm_nodes = 256
final_fcs = [64]
fc_drop_rate = 0.5
    
network_input_shape = (3,) + dataset_output_shape


dataset_path = "D:/university/bioinformatics/master project/codes/dataset/ALL IDB blast free dataset"
centroids_path = "D:/university/bioinformatics/master project/codes/codes - object detection/two stage - CNN - Transformer - gray scale/ALL IDB 1 results/Middle Results/Candidate Centroids/"

batch_size = 8


network_evaluator = NetworkEvaluator(network_input_shape,
                                     num_classes,
                                     lstm_nodes,
                                     final_fcs,
                                     fc_drop_rate)

dataset = CellDataset(dataset_path, centroids_path, crop_shape, dataset_output_shape)

data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)



accuracy, cm_n, cm = network_evaluator.find_performance(dataset)


save_displays([cm, cm_n], ['Normalized', 'Non - Normalized'], network_evaluator.final_folder)

report = ""
report += "all data accuracy\t:\t{}\n".format(accuracy)

with open(network_evaluator.final_folder + "/blast free accuracy.txt", 'w') as f:
    f.write(report)

print(report)
