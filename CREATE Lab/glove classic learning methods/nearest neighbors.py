# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 17:26:16 2021

@author: Amirhossein
"""

import numpy as np
from sklearn.decomposition import PCA
from data_plotter import AnglesDataPlotter
import time


def load_data(train_ratio, val_ratio):
    glove_data = np.load("data/glove data.npy")
    angles_data = np.load("data/angles data.npy")
    glove_data = glove_data / 99
    n = glove_data.shape[0]
    all_indices = list(range(n))
    np.random.shuffle(all_indices)
    split_train = int(np.floor(n * train_ratio))
    split_val = int(np.floor(n * (train_ratio + val_ratio)))
    train_indiecs = all_indices[:split_train]
    val_indices = all_indices[split_train:split_val]
    test_indices = all_indices[split_val:]
    
    train_glove = glove_data[train_indiecs, :]
    train_angles = angles_data[train_indiecs, :]
    val_glove = glove_data[val_indices, :]
    val_angles = angles_data[val_indices, :]
    test_glove = glove_data[test_indices, :]
    test_angles = angles_data[test_indices, :]
    
    return train_glove, train_angles, val_glove, val_angles, test_glove, test_angles


def apply_pca(train_data, val_data, test_data, pca_n):
    pca = PCA(n_components = pca_n)
    pca.fit(train_data)
    
    train_data_pca = pca.transform(train_data)
    val_data_pca = pca.transform(val_data)
    test_data_pca = pca.transform(test_data)
    
    return train_data_pca, val_data_pca, test_data_pca
    

def get_prediction(input_data, train_inputs, train_outputs, k_nn):
    dist = (train_inputs - input_data)**2
    dist = np.sum(dist, axis = 1)
    dist = np.sqrt(dist)
    best_idx = np.argpartition(dist, k_nn)[:k_nn]
    best_angles = train_outputs[best_idx, :]
    pred = np.mean(best_angles, axis = 0)
    return pred



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


def plot_single_result(y, y_pred, data_plotter):
    
    y_pred_positions = angle_to_position(y_pred)
    y_positions = angle_to_position(y)
    
    plot_data = np.concatenate((np.expand_dims(y_positions, axis = 0),
                                np.expand_dims(y_pred_positions, axis = 0)),
                               axis=0)
    
    data_plotter.plot_hands(plot_data)
    time.sleep(0.3)


def find_loss(glove_data, angles_data, train_glove, train_angles, k_nn):
    all_loss = np.zeros((glove_data.shape[0],))
    loss_weight = np.zeros((glove_data.shape[0],))
    for idx in range(glove_data.shape[0]):
        input_glove = glove_data[idx, :]    
        pred = get_prediction(input_glove, train_glove, train_angles, k_nn)    
        true_angles = angles_data[idx, :]
        all_loss[idx] = np.linalg.norm(pred - true_angles) / true_angles.shape[0]
        loss_weight[idx] = np.linalg.norm(true_angles)
    
    loss_weight = loss_weight / np.sum(loss_weight)
    mean_loss = np.dot(all_loss, loss_weight)
    #mean_loss = np.mean(all_loss)
    return mean_loss


def plot_all_data(glove_data, angles_data, train_glove, train_angles, k_nn):
    data_plotter = AnglesDataPlotter()
    for idx in range(glove_data.shape[0]):
        input_glove = glove_data[idx, :]    
        pred = get_prediction(input_glove, train_glove, train_angles, k_nn)    
        true_angles = angles_data[idx, :]
        plot_single_result(true_angles, pred, data_plotter)



pca_n = 12
k_nn = 3
use_plotter = False


train_ratio = 0.5
val_ratio = 0.3



np.random.seed(6)

data = load_data(train_ratio, val_ratio)
train_glove = data[0]
train_angles = data[1]
val_glove = data[2]
val_angles = data[3]
test_glove = data[4]
test_angles = data[5]


print("Train size :\t{}".format(train_glove.shape[0]))
print("Validation size :\t{}".format(val_glove.shape[0]))
print("Test size :\t{}\n\n".format(test_glove.shape[0]))



train_glove, val_glove, test_glove = apply_pca(train_glove, val_glove, test_glove, pca_n)



mean_train_loss = find_loss(train_glove, train_angles, train_glove, train_angles, k_nn)
mean_val_loss = find_loss(val_glove, val_angles, train_glove, train_angles, k_nn)
mean_test_loss = find_loss(test_glove, test_angles, train_glove, train_angles, k_nn)

print("Average Train Loss :\t{}".format(mean_train_loss))
print("Average Validation Loss :\t{}".format(mean_val_loss))
print("Average Test Loss :\t{}".format(mean_test_loss))



if use_plotter:
    plot_all_data(test_glove, test_angles, train_glove, train_angles, k_nn)


