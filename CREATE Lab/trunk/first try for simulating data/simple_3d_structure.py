# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:34:21 2021

@author: Amirhossein
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def show_points(points, elevation = 90, azimuth = 90):


    figure = plt.figure(figsize = (10, 10))    
    
    ax = Axes3D(figure, adjustable = 'datalim')
    
    color = 'brg'
    
    if type(points) is tuple:
        for k in range(len(points)):            
            ax.scatter(points[k][:, 0], points[k][:, 1], points[k][:, 2], marker='.', c = color[k])
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.', c =color[0])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]),max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    
    
    
    ax.view_init(elev = elevation, azim = azimuth)

    plt.show()


    

class Curve:
    
    def __init__(self,curve_function, length, split_number, length_step = 1e-6):  
        
        self.curve_function = curve_function
        self.curve_length = length
        self.max_t = self.find_t_at_length(length, length_step)
        self.curve_t_series, self.curve_points = self.find_curve_points(split_number)
    
    
    def find_curve_points(self, split_number):
        distance = self.curve_length / split_number
        t_series = np.zeros((split_number + 1,))
        points = np.zeros((split_number + 1, 3))
        
        for k in range(1, split_number + 1):
            t_series[k] = self.find_next_point(t_series[k-1], distance)
            
        for k in range(t_series.shape[0]):
            points[k, :] = self.curve_function(t_series[k])
        
        return t_series, points


    
    def find_t_at_length(self, length, step):
        final_t = 0
        total_dist = 0
        while total_dist < length:
            final_t += step
            previous_total_dist = total_dist
            total_dist += self.dist_on_curve(final_t, final_t - step)
        
        if total_dist - length > length - previous_total_dist:
            best_t = final_t - step
        else:
            best_t = final_t
        return best_t
    

    def dist_on_curve(self, t1, t2):
    
        p1 = self.curve_function(t1)
        p2 = self.curve_function(t2)
        d = np.linalg.norm(p1 - p2)
        return d


    def find_next_point(self, t, distance, epsilon = 1e-3):
        

        if self.dist_on_curve(t, self.max_t) < distance:
            return self.max_t
        
        max_t = self.max_t        
        min_t = t
        current_t = (min_t + max_t) / 2
        
        while np.abs(self.dist_on_curve(current_t, t) - distance) > epsilon:
            if self.dist_on_curve(current_t, t) > distance:
                max_t = current_t
            else:
                min_t = current_t
            current_t = (min_t + max_t) / 2
        
        return current_t
    
class Volume:
    
    def __init__(self, central_curve, radious, circle_points_number):
        
        self.central_curve = central_curve.curve_points
        self.radious = radious
        self.circle_points_number = circle_points_number
        
        self.rotation_matrixes = self.find_rotation_matrixes(self.central_curve)
        self.points = self.create_circles()
        
        
    def find_rotation_matrixes(self, central_curve):
        
        n = central_curve.shape[0]
        
        r_matrixes = np.zeros((n, 3, 3))
        axis = np.zeros((n, 3, 3))
        axis[0, :, :] = np.identity(3)
        r_matrixes[0, :, :] = np.identity(3)
        
        for k in range(1, n):
            v2 = central_curve[k, :] - central_curve[k - 1, :]
            if k > 1:
                v1 = central_curve[k - 1, :] - central_curve[k - 2, :]
            else:
                v1 = np.array([1., 0., 0.])
            
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            v3 = np.cross(v2, v1)
            v3 = v3 / np.linalg.norm(v3)
            
            v4 = np.cross(v2, v3)
            v4 = v4 / np.linalg.norm(v4)
            
            
            axis[k, 0, :] = v2
            axis[k, 2, :] = v3
            axis[k, 1, :] = v4
        
        for k in range(1, n):
            
            a1 = axis[k - 1, :, :]
            a2 = axis[k, :, :]
            
            r_matrixes[k, :, :] = np.matmul(np.linalg.inv(a1), a2)
        
        return r_matrixes
            
    
    
    @staticmethod
    def create_one_circle(r, n, center, rotation_matrix):
        
        points = np.zeros((n, 3))
        for k in range(n):
            theta = 2 * k * np.pi / n
            p = np.zeros((1, 3))
            p[0, 1] = r * np.cos(theta)
            p[0, 2] = r * np.sin(theta)
            
            points[k, :] = np.matmul(p, rotation_matrix)
            points[k, :] += center
        return points
        
    def create_circles(self):
        
        n_p = self.central_curve.shape[0]
        n_c = self.circle_points_number
        
        points = np.zeros((n_p, n_c, 3))
        accumulate_r_matrix = self.rotation_matrixes[0, :, :]
        
        for k in range(0, n_p):
            accumulate_r_matrix = np.matmul(accumulate_r_matrix, self.rotation_matrixes[k, :, :])
            points[k, :, :] = self.create_one_circle(self.radious, n_c, self.central_curve[k, :], accumulate_r_matrix)
        
        return np.reshape(points, (n_p * n_c, 3))
                


def main():
    
    
    def my_curve_function(t): 
         x = t

         
         r = t
         theta = t * 0.5 * np.pi
         
         y = np.cos(theta) * r
         z = np.sin(theta) * r
 
         return np.array([x, y, z])
    
    

    radious = 0.2
    central_curve_points = 50
    points_on_each_circle = 100
    
    
    curve = Curve(my_curve_function, length = 1 * np.pi, split_number = central_curve_points, length_step=1e-4)
    volume = Volume(curve, radious, points_on_each_circle)
    
    for azimuth in range(0, 100, 5):
        show_points((volume.points, volume.central_curve), 10, azimuth)

if __name__ == "__main__":
    main()




