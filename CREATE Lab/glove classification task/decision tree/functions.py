# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:15:32 2021

@author: Amirhossein
"""

import numpy as np
import os




def load_data(phase):

    sub_folders = os.listdir("../data/data {}".format(phase))
            
    data_list = []
    labels_list = []
    for folder_name in sub_folders:
        file_path = "../data/data {}/{}/{}.npy".format(phase, folder_name, folder_name)
        data_list.append(np.load(file_path))
        labels_list.append(np.ones((data_list[-1].shape[0],)) * int(folder_name))
    
    X = np.concatenate(data_list, axis = 0)
    Y = np.concatenate(labels_list, axis = 0).astype('int32')
    
    p = np.random.permutation(X.shape[0])
    
    X = X[p, :]
    Y = Y[p]
    
    return X, Y


def load_final_data():
    X_train, Y_train = load_data("train")
    X_val, Y_val = load_data("validation")
    
    X = np.concatenate((X_train, X_val), axis = 0)
    Y = np.concatenate((Y_train, Y_val), axis = 0)
    
    p = np.random.permutation(X.shape[0])
    
    X = X[p, :]
    Y = Y[p]
    
    return X, Y

def load_all_data():
    X_train, Y_train = load_data("train")
    X_val, Y_val = load_data("validation")
    X_test, Y_test = load_data("test")
    
    X = np.concatenate((X_train, X_val, X_test), axis = 0)
    Y = np.concatenate((Y_train, Y_val, Y_test), axis = 0)
    
    p = np.random.permutation(X.shape[0])
    
    X = X[p, :]
    Y = Y[p]
    
    return X, Y
    

def find_accuracy(model, x, y_true):
    y_pred = model.predict(x)
    accuracy = np.sum(y_pred == y_true) / y_pred.shape[0]
    return accuracy