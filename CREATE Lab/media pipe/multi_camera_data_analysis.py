# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:55:45 2021

@author: Amirhossein
"""

import numpy as np
import matplotlib.pyplot as plt
from multi_camera_data_plotter import DataPlotter


class DataAnalyzer:
    
    def __init__(self, number_of_cameras):
        self.number_of_cameras = number_of_cameras
        self.coordinates = np.zeros((self.number_of_cameras, 21, 3))
        self.image_shapes = np.zeros((self.number_of_cameras, 2))
        self.transformation_matrixes = None
        self.transformed_coordinates = None
        self.mean_coordinates = None
        self.data_plotter = DataPlotter()
        
        self.angles_log_max_length = 100
        
        self.angles_log = []
        
        self.first_angles_plot_done = False
        self.angles_error_plots_list = []
        
        

        
        
    def set_data(self, index, coordinates, image_shape):
        self.coordinates[index, :, :] = coordinates
        self.image_shapes[index, :] = np.array(image_shape)
        
    def analyse(self):
        
        self.translate_to_origin()
        self.transformation_matrixes = self.find_transformation_matrix(use_weights = True)
        
        self.transformed_coordinates = self.transform_coordinates(self.coordinates,
                                                                  self.transformation_matrixes)        
        self.mean_coordinates = np.mean(self.transformed_coordinates, axis=0)
   
    
    def plot_angles_log_error(self):
        
        if not self.first_angles_plot_done:
            
            self.data_plotter.figure = plt.figure(constrained_layout=True, figsize = (14, 7))
            self.data_plotter.gs = self.data_plotter.figure.add_gridspec(5, 6)
            
            figure = self.data_plotter.figure
            gs = self.data_plotter.gs
    
            cmap = plt.cm.get_cmap('hsv', 15)
            cmap_idx = list(range(15))
            np.random.seed(0)
            np.random.shuffle(cmap_idx)

            x = np.array(range(len(self.angles_log)))
            
            angles = np.array(self.angles_log)
            #angles_error = np.max(angles, axis = 1) - np.min(angles, axis = 1)
            angles_error = np.std(angles, axis = 1)
            for k in range(angles_error.shape[1]):
                ax = figure.add_subplot(gs[k//3, k%3])
                ax.set_xlim([0, self.angles_log_max_length])
                ax.set_ylim([0, 30])
                line, = ax.plot(x, angles_error[:, k], marker = '.', color = cmap(cmap_idx[k]))
                self.angles_error_plots_list.append(line)
            
            ax_2 = figure.add_subplot(gs[:, 3:6])
            ax_2.set_xlim([0, self.angles_log_max_length])
            ax_2.set_ylim([0, 30])
            line, = ax_2.plot(x, np.mean(angles_error, axis = 1), marker = '.')
            self.angles_error_plots_list.append(line)
            
            self.first_angles_plot_done = True
            
        else:
            angles = np.array(self.angles_log)
            #angles_error = np.max(angles, axis = 1) - np.min(angles, axis = 1)
            angles_error = np.std(angles, axis = 1)
            
            x = np.array(range(len(self.angles_log)))
            
            for k in range(angles_error.shape[1]):
                line = self.angles_error_plots_list[k]
                line.set_ydata(angles_error[:, k])
                line.set_xdata(x)
            line = self.angles_error_plots_list[-1]
            line.set_ydata(np.mean(angles_error, axis = 1))
            line.set_xdata(x)
            
            self.data_plotter.figure.canvas.draw()
            self.data_plotter.figure.canvas.flush_events()
            
        
        
    
    
    
    def add_angles_to_log_history(self):
        
        self.analyse()
        
        mean_angles = self.find_angles(self.mean_coordinates)        
        all_angles = np.zeros((self.transformed_coordinates.shape[0] + 1, mean_angles.shape[0]))
        all_angles[0, :] = mean_angles
        for k in range(1, all_angles.shape[0]):
            all_angles[k, :] = self.find_angles(self.transformed_coordinates[k - 1, :, :])
        
        self.angles_log.append(all_angles)
        
        if len(self.angles_log) > self.angles_log_max_length:
            self.angles_log.pop(0)
        
   
    def plot_all_hands(self):
        
        self.analyse()
        
        all_hands = np.concatenate((self.transformed_coordinates,
                                    np.expand_dims(self.mean_coordinates, axis=0)),
                                   axis=0)
        self.data_plotter.plot_hands(all_hands)
        

        

    @staticmethod
    def find_angles(coordinates):
        connected_nodes = [(0, 1, 2, 3, 4),
                           (0, 5, 6, 7, 8),
                           (0, 9, 10, 11, 12),
                           (0, 13, 14, 15, 16),
                           (0, 17, 18, 19, 20)
                           ]
        angles = np.zeros((15,))
        c = 0
        for nodes_list in connected_nodes:
            for k in range(1, len(nodes_list) - 1):
                v1 = coordinates[nodes_list[k + 1], :] - coordinates[nodes_list[k], :]
                v2 = coordinates[nodes_list[k - 1], :] - coordinates[nodes_list[k], :]
                
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                cos_angle = np.dot(v1, v2)
                
                angle = 180 * np.arccos(cos_angle) / np.pi
                angles[c] = angle
                c += 1
        return angles
                
    
    @staticmethod
    def transform_coordinates(coordinates, transformation_matrixes):
        new_coordinates = coordinates.copy()
        for k in range(1, new_coordinates.shape[0]):
            new_coordinates[k, :, :] = np.matmul(coordinates[k, :, :],
                                                 transformation_matrixes[k-1, :, :])
        return new_coordinates
        
    def translate_to_origin(self):
        
        for k in range(self.coordinates.shape[0]):
            origin = self.coordinates[k, 0, :].copy()
            for idx in range(self.coordinates.shape[1]):
                self.coordinates[k, idx, :] -= origin
    
    def find_transformation_matrix(self, use_weights = True):
        
        if use_weights:
            w = self.create_weight_matrix()
        else:
            w = np.identity(self.coordinates.shape[1])

        
        X_target = self.coordinates[0, :, :]
        X_target = np.matmul(w, X_target)
        Y = self.coordinates[1:, :, :]
        transformation_matrixes = np.zeros((self.coordinates.shape[0] - 1, 3, 3))
        for k in range(Y.shape[0]):
            X = Y[k, :, :]
            X = np.matmul(w, X)
            XtX = np.matmul(np.transpose(X), X)
            XtX_inv = np.linalg.inv(XtX)
            XtX_target = np.matmul(np.transpose(X), X_target)
            transformation_matrixes[k, :, :] = np.matmul(XtX_inv, XtX_target)
        
        return transformation_matrixes
            
            
    @staticmethod
    def create_weight_matrix():
        w_0 = 1
        w_1 = 1
        w_2 = 0.8
        w_3 = 0.6
        w_4 = 0.4
            
        v = np.zeros((21,))
        v[0] = w_0
        v[[1, 5, 9, 13, 17]] = w_1
        v[[2, 6, 10, 14, 18]] = w_2
        v[[3, 7, 11, 15, 19]] = w_3
        v[[4, 8, 12, 16, 20]] = w_4
        w = np.diag(v)
        return w
        

