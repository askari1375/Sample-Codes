# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:21:59 2021

@author: Amirhossein
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class AnglesDataPlotter:
    
    def __init__(self):
        self.coordinates = None
        self.figure = None
        self.ax = None
        self.hands = []
        self.connections = []
        self.colors_list = 'grbcmyk'
        
        self.first_plot_done = False
        
        self.last_hand_plot_time = time.time()
        self.last_error_plot_time = time.time()
        
        self.figure = None
        self.gs = None
        
    
    def plot_hands(self, coordinates):
        
        self.coordinates = coordinates
        
        if not self.first_plot_done:
            self.first_plot()
            self.first_plot_done = True
        else:
            self.update_plot()
    
    def update_plot(self):
        
        for k in range(self.coordinates.shape[0]):
            one_hand = self.coordinates[k, :, :]
            self.hands[k].set_xdata(one_hand[:, 0])
            self.hands[k].set_ydata(one_hand[:, 1])
            self.hands[k].set_3d_properties(one_hand[:, 2])
            self.plot_connections(one_hand, k, update_mode = True)
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        
    
    def first_plot(self):
       
        self.adjust_plot()
        for k in range(self.coordinates.shape[0]):
            one_hand = self.coordinates[k, :, :]
            hand, = self.ax.plot(one_hand[:, 0],
                                 one_hand[:, 1],
                                 one_hand[:, 2], marker='o', linestyle='None')
            self.plot_connections(one_hand, k, update_mode = False)
            self.hands.append(hand)
        
        plt.show()
        

    
    
    def adjust_plot(self, title = None):
        
        self.figure = plt.figure(figsize = (10, 10))    
        self.ax = Axes3D(self.figure, adjustable = 'datalim')
        
        
        axis_limits = {'min X' : -2,
                       'max X' : 2,
                       'min Y' : -2,
                       'max Y' : 2,
                       'min Z' : -2,
                       'max Z' : 2}
        
        
        self.ax.set_xlim3d(axis_limits['min X'], axis_limits['max X'])
        self.ax.set_ylim3d(axis_limits['min Y'], axis_limits['max Y'])
        self.ax.set_zlim3d(axis_limits['min Z'], axis_limits['max Z'])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        if title is not None:
            plt.title(title)
        
        
    
    def plot_connections(self, one_hand_points, hand_number, update_mode = False):
        
        if not update_mode:
            self.connections.append([])
        
        connections = [(0, 1, 2, 3),
                       (4, 5, 6, 7),
                       (8, 9, 10, 11),
                       (12, 13, 14, 15),
                       (16, 17, 18, 19)]
        
        for idx, connection in enumerate(connections):
            x = []
            y = []
            z = []
            for k in range(len(connection)):
                x.append(one_hand_points[connection[k], 0])
                y.append(one_hand_points[connection[k], 1])
                z.append(one_hand_points[connection[k], 2])
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)

            if not update_mode:
                connection, = self.ax.plot(x, y, z, color = self.colors_list[hand_number])
                self.connections[hand_number].append(connection)
            else:
                self.connections[hand_number][idx].set_xdata(x)
                self.connections[hand_number][idx].set_ydata(y)
                self.connections[hand_number][idx].set_3d_properties(z)
