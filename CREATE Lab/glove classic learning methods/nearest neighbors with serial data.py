# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 17:26:16 2021

@author: Amirhossein
"""

import numpy as np
from sklearn.decomposition import PCA
from data_plotter import AnglesDataPlotter
import time
import serial
import matplotlib.pyplot as plt



class NearestNeighborsPredictor:
    def __init__(self, pca_n, k_nn):
        
        self.k_nn = k_nn
        
        train_glove, self.train_angles = self.load_data()
        print("Train size :\t{}".format(train_glove.shape[0]))
        
        self.pca = PCA(n_components = pca_n)
        self.pca.fit(train_glove)
        
        self.train_glove = self.pca.transform(train_glove)
        
        
    
    @staticmethod
    def load_data():
        glove_data = np.load("data/glove data.npy")
        angles_data = np.load("data/angles data.npy")
        glove_data = glove_data / 99
        n = glove_data.shape[0]
        all_indices = list(range(n))
        np.random.shuffle(all_indices)
            
        return glove_data, angles_data


    

    def get_prediction(self, input_vector):
        input_data = self.pca.transform(np.expand_dims(input_vector, axis = 0))
        
        dist = (self.train_glove - input_data)**2
        dist = np.sum(dist, axis = 1)
        dist = np.sqrt(dist)
        best_idx = np.argpartition(dist, k_nn)[:k_nn]
        best_angles = self.train_angles[best_idx, :]
        pred = np.mean(best_angles, axis = 0)
        return pred
    
    
    @staticmethod
    def get_loss(pred_angles, true_angles):
        loss = np.linalg.norm(pred_angles - true_angles) / true_angles.shape[0]
        loss = np.sum(np.abs(pred_angles - true_angles)) / true_angles.shape[0]
        return loss


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





def plot_single_result(data_plotter, y_pred, y_true = None):
    
    y_pred_positions = NearestNeighborsPredictor.angle_to_position(y_pred)
    a = np.expand_dims(y_pred_positions, axis = 0)
    if y_true is not None:
        y_true_positions = NearestNeighborsPredictor.angle_to_position(y_true)
        b = np.expand_dims(y_true_positions, axis = 0)
        plot_data = np.concatenate((a, b), axis=0)
    else:
        plot_data = a
    
    data_plotter.plot_hands(plot_data)



def find_total_loss(glove_data, angles_data, model):
    all_loss = np.zeros((glove_data.shape[0],))
    loss_weight = np.zeros((glove_data.shape[0],))
    for idx in range(glove_data.shape[0]):
        input_glove = glove_data[idx, :]    
        pred_angles = model.get_prediction(input_glove)    
        true_angles = angles_data[idx, :]
        all_loss[idx] = NearestNeighborsPredictor.get_loss(pred_angles, true_angles)
        loss_weight[idx] = np.linalg.norm(true_angles)
    
    loss_weight = loss_weight / np.sum(loss_weight)
    mean_loss = np.dot(all_loss, loss_weight)
    return mean_loss






pca_n = 12
k_nn = 3

duration = 20




np.random.seed(6)

model = NearestNeighborsPredictor(pca_n = 12, k_nn = 3)

glove_data = np.load("data/glove data.npy")
angles_data = np.load("data/angles data.npy")
glove_data = glove_data / 99

mean_train_loss = find_total_loss(glove_data, angles_data, model)
print("Average Train Loss :\t{}".format(mean_train_loss))

data_plotter = AnglesDataPlotter()

glove_data = np.load("data/validation/glove data.npy")
angles_data = np.load("data/validation/angles data.npy")

mean_train_loss = find_total_loss(glove_data, angles_data, model)
print("Average Validation Loss :\t{}".format(mean_train_loss))

pred_angles_list = []
for idx in range(glove_data.shape[0]):
    pred_angles = model.get_prediction(glove_data[idx, :] / 99)
    pred_angles_list.append(pred_angles)
    #plot_single_result(data_plotter, pred_angles, y_true = angles_data[idx, :])
    #time.sleep(0.1)

pred_angles_list = np.array(pred_angles_list)

fig, axs = plt.subplots(2, 5)
for i in range(5):
    for j in range(2):
        axs[j, i].plot(pred_angles_list[:, 2 * i + j], color = 'r')
        axs[j, i].plot(angles_data[:, 2 * i + j], color = 'b')








# =============================================================================
# ser = serial.Serial(port="COM8",
#                     baudrate="460800",
#                     bytesize=8,
#                     timeout=2,
#                     stopbits=serial.STOPBITS_ONE)
# 
# captured_data = []
# start_time = time.time()
# while time.time() - start_time < duration:
#     if(ser.in_waiting > 0):
#         try:
#             ser.flushInput()
#             ser.flushOutput()
#             recived_data = ser.readline().decode('Ascii')
#             recived_data = recived_data.split(",")[:16]
#             recived_data = [int(x) for x in recived_data]
#             if len(recived_data) == 16 and max(recived_data) < 100:
#                 input_vector = np.array(recived_data) / 99
#                 
#                 captured_data.append(np.array(recived_data))
#                 
#                 pred_angles = model.get_prediction(input_vector)
#                 print(recived_data)
#                 #print(pred_angles)        
#                 plot_single_result(data_plotter, pred_angles, y_true = None)
#                 time.sleep(0.3)
#             
#             
#         except:
#             pass
#         
# ser.close()
# 
# 
# captured_data = np.array(captured_data)
# max_min_recived = np.max(captured_data, axis = 0) - np.min(captured_data, axis = 0)
# glove_data = np.load("data/glove data.npy")
# max_min_train = np.max(glove_data, axis = 0) - np.min(glove_data, axis = 0)
# 
# print(max_min_train)
# print(max_min_recived)
# =============================================================================
