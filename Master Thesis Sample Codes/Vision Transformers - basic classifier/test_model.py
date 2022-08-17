# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:12:52 2021

@author: Amirhossein
"""

import torch
import numpy as np
from neural_network import VisionTransormer
from dataset import create_datasets
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class NetworkEvaluator():
    def __init__(self,
                 backbone_name,
                 num_classes,
                 last_layers_number,
                 network_input_shape):
        
        save_results_folder = "results/"
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.last_layers_number = last_layers_number
        
        self.model = VisionTransormer(backbone_name, num_classes, last_layers_number)
        
        model_name = self.create_model_name()
        self.final_folder = os.path.join(save_results_folder, model_name[:-3])
        model_path = os.path.join(self.final_folder, model_name)
        self.model.load_state_dict(torch.load(model_path))
        
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        

    def create_model_name(self):
        model_name = "{}__{}.pt".format(self.backbone_name, self.last_layers_number)
        return model_name
    
    def get_prediction(self, idx, dataset):
        x, y = dataset.__getitem__(idx)
        x = torch.unsqueeze(x, dim = 0).to(self.device)
        #y = torch.unsqueeze(y, dim = 0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x)
        outputs = outputs.to(torch.device("cpu")).numpy()
        y_pred = np.argmax(outputs, axis=-1)
        
        #y_true = y.to(torch.device("cpu")).numpy().reshape((-1))
        y_true = y
        
        return y_pred, y_true
    
    def find_performance(self, dataset):
        
        n = len(dataset)

        y_pred = np.zeros((n,))
        y_true = np.zeros((n,))
        
        for idx in tqdm(range(n)):
             y_pred[idx], y_true[idx] = self.get_prediction(idx, dataset)
        
        accuracy = np.sum(y_pred == y_true) / n
        cm = confusion_matrix(y_true, y_pred, normalize = 'true')
        disp = ConfusionMatrixDisplay(confusion_matrix = cm,
                                      display_labels = dataset.classes)
       
        
        return accuracy, disp
    

def save_displays(train_disp, val_disp, test_disp, save_folder):
    disps = [train_disp, val_disp, test_disp]
    labels = ["Train", "Validation", "Test"]
    fig, axs = plt.subplots(1, 3, figsize = (21, 7))
    fig.suptitle("Confusion Matrices")
    for idx, disp in enumerate(disps):
        disp.plot(ax = axs[idx])
        axs[idx].set_title("{}".format(labels[idx]))
    
    save_path = os.path.join(save_folder, "confusion matrix.png")
    fig.savefig(save_path)
    



num_classes = 5
backbone_name = 'vit_base_patch16_224'
last_layers_number = 4

image_shape = (224, 224)

network_input_shape = (3,) + image_shape

dataset_path = "D:/university/bioinformatics/master project/codes/dataset/train validation test"

network_evaluator = NetworkEvaluator(backbone_name,
                                     num_classes,
                                     last_layers_number,
                                     network_input_shape)

train_dataset, val_dataset, test_dataset = create_datasets(dataset_path, image_shape)


train_accuracy, train_matrix = network_evaluator.find_performance(train_dataset)
validation_accuracy, validation_matrix = network_evaluator.find_performance(val_dataset)
test_accuracy, test_matrix = network_evaluator.find_performance(test_dataset)

save_displays(train_matrix,
              validation_matrix,
              test_matrix,
              network_evaluator.final_folder)

report = ""
report += "train accuracy\t:\t{}\n".format(train_accuracy)
report += "validation accuracy\t:\t{}\n".format(validation_accuracy)
report += "test accuracy\t:\t{}".format(test_accuracy)

with open(network_evaluator.final_folder + "/accuracy.txt", 'w') as f:
    f.write(report)

print(report)
