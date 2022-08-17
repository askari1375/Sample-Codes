# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:34:57 2021

@author: Amirhossein
"""

from simple_3d_structure import Curve, Volume
import numpy as np
import matplotlib.pyplot as plt



class ImageRenderer:
    
    def __init__(self, cloud_points, camera_properties):
        
        self.cloud_points = cloud_points
        self.camera_matrix = self.create_camera_matrix(camera_properties)
        print(self.camera_matrix)
        
    def render(self, image_shape):
        image = np.zeros((image_shape[0], image_shape[1], 3))
        
        h_coordinates, points_in_image = self.find_homogeneous_coordinates()
        
        alphas = []
        rgba_colors = np.zeros((h_coordinates.shape[0],4))
        rgba_colors[:,0] = 1.0
        
        min_z = np.min(h_coordinates[:, 2])
        max_z = np.max(h_coordinates[:, 2])        
        for k in range(points_in_image.shape[0]):            
            alphas.append(1 - (h_coordinates[k, 2] - min_z) / (max_z - min_z))
        rgba_colors[:, 3] = np.array(alphas)

        print(rgba_colors.shape)
        print(points_in_image.shape)
        plt.figure()
        plt.scatter(points_in_image[:, 0],
                    points_in_image[:, 1],
                    marker = '.',
                    color = rgba_colors)
        plt.show()
        
    
    def find_homogeneous_coordinates(self):
        
        h_coordinates = np.zeros((self.cloud_points.shape[0], 3))
        points_in_image = np.zeros((self.cloud_points.shape[0], 2))
        for k in range(self.cloud_points.shape[0]):
            p = np.zeros((4, 1))
            p[0:3, 0] = self.cloud_points[k, :]
            p[3, 0] = 1
            h_coordinates[k, :] = np.reshape(np.matmul(self.camera_matrix, p), (3,))
            points_in_image[k, :] = h_coordinates[k, 0:2] /  h_coordinates[k, 2]
        
        return h_coordinates, points_in_image
        
        
        
        
  
    def create_camera_matrix(self, camera_properties):
        changing_coordinate_matrix = np.zeros((3, 4))
        
        R = self.find_rotation_matrix(camera_properties)
        changing_coordinate_matrix[:, 0:3] = R
        changing_coordinate_matrix[:, 3] = np.array(camera_properties['translation vector'])
        
        f = camera_properties['f']
        
        camera_matrix = np.array([[f, 0, 0],
                                  [0, f, 0],
                                  [0, 0, 1]])
        general_camera_matrix = np.matmul(camera_matrix, changing_coordinate_matrix)
        return general_camera_matrix
        
        
    def find_rotation_matrix(self, camera_properties):
        theta = np.pi * camera_properties['theta'] / 180
        phi = np.pi * camera_properties['phi'] / 180
        psi = np.pi * camera_properties['psi'] / 180
        
        s = np.sin(phi)
        c = np.cos(phi)
        
        R_x = np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
        
        s = np.sin(theta)
        c = np.cos(theta)
        
        R_y = np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
        
        s = np.sin(psi)
        c = np.cos(psi)
        
        R_z = np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
        
        R = np.matmul(R_z, np.matmul(R_y, R_x))
        
        return R
    


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

camera_properties = {'f'                    : 1,
                     'phi'                  : 10,
                     'theta'                : 10,
                     'psi'                  : 10,
                     'translation vector'   : [-2.3, -2.5, 10]}

n = 5
points = np.zeros((n**3, 3))
for k in range(n**3):
    points[k, 0] = (k % n**2) // n
    points[k, 1] = (k % n**2) % n
    points[k, 2] = k // (n**2)


image_renderer = ImageRenderer(points, camera_properties)
image_renderer.render((100, 100))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.')


