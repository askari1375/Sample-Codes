# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:07:27 2021

@author: Amirhossein
"""

import numpy as np
from numpy.polynomial import Polynomial


def boundary_curves_points(x, p, dp, r):
    
    b = r / np.sqrt(1 + np.power(dp(x), 2))
    a = -b * dp(x)
    
    y = p(x)
    
    x1, y1 = x + a, y + b
    x2, y2 = x - a, y - b
    
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])

    return p1, p2
    

def find_curves(x, p_coef, r):
    
    d_p_coef = []
    
    for i in range(1, len(p_coef)):
        d_p_coef.append(p_coef[i] * i)
    
    
    p = Polynomial(p_coef)
    d_p = Polynomial(d_p_coef)
    
    y = p(x)
    
    up = np.zeros((x.shape[0], 2))
    down = np.zeros((x.shape[0], 2))
    middle = np.zeros((x.shape[0], 2))
    
    for k in range(x.shape[0]):
        up[k, :], down[k, :] = boundary_curves_points(x[k], p, d_p, r[k])
        middle[k, 0], middle[k, 1] = x[k], y[k]
    
    return middle, up, down



def find_max_x(p_coef, length, epsilon = 1e-4):
    l = 0
    p = Polynomial(p_coef)
    
    x_0 = 0
    x_1 = epsilon
    y_0 = p(x_0)
    y_1 = p(x_1)
    
    while l < length:
        l += np.sqrt(np.power(x_1 - x_0, 2) + np.power(y_1 - y_0, 2))
        x_0 += epsilon
        y_0 = y_1
        x_1 += epsilon
        y_1 = p(x_1)
    return x_0

def find_x_list(p_coef, dist_list, epsilon = 1e-4):
    l = 0
    p = Polynomial(p_coef)
    
    x_0 = 0
    x_1 = epsilon
    y_0 = p(x_0)
    y_1 = p(x_1)
    
    x = np.zeros(dist_list.shape)
    c = 0
    
    
    while c < x.shape[0]:
        l += np.sqrt(np.power(x_1 - x_0, 2) + np.power(y_1 - y_0, 2))
        x_0 += epsilon
        y_0 = y_1
        x_1 += epsilon
        y_1 = p(x_1)
        
        if dist_list[c] < l:
            x[c] = x_0
            c += 1
                
    return x