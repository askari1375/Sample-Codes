# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:34:25 2021

@author: Amirhossein
"""

import numpy as np
import cv2
from functions import find_curves, find_x_list



def create_side_view_image(p_coef, length, r_0, n, image_shape, thickness):
    
    scale = image_shape[0] / 2
    vertical_shift = image_shape[0] / 2
    r = np.linspace(r_0, 0, n)
    dist_list = np.linspace(0, 1, n)
    
    x = find_x_list(p_coef, dist_list, epsilon = 1e-4)
    
    middle, up, down = find_curves(x, p_coef, r)
    
    image = np.zeros(image_shape, dtype = "uint8")
    for k in range(middle.shape[0]):
        y1 = int(up[k, 0] * scale)
        x1 = int(up[k, 1] * scale + image_shape[1] / 2)
        y2 = int(down[k, 0] * scale)
        x2 = int(down[k, 1] * scale + image_shape[1] / 2)
        
        image = cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    
    image = image.astype('uint8')
    shift_matrix = np.float32([[1, 0, 0], [0, 1, vertical_shift]])
    image = cv2.warpAffine(image, shift_matrix, (image.shape[1], image.shape[0]))
    return image


def main():
    #p_coef = [0, 0, 10, -20, 10]
    p_coef = [0, 0, 1]
    n = 50
    length = 1
    r_0 = 0.1
    thickness = 1
    image_shape = (1000, 1000, 3)
    
    image = create_side_view_image(p_coef, length, r_0, n, image_shape, thickness)
    
    cv2.imshow("side view", image)


if __name__ == "__main__":
    main()

# =============================================================================
# fig_lines = plt.figure(figsize = (10, 10))
# ax_lines = fig_lines.add_subplot(111)
# ax_lines.plot(middle[:, 0], middle[:, 1], marker = ".")
# ax_lines.plot(up[:, 0], up[:, 1], marker = '.', color = 'r')
# ax_lines.plot(down[:, 0], down[:, 1], marker = '.', color = 'g')
# 
# ax_lines.axis('equal')
# 
# fig = plt.figure(figsize = (10, 10))
# ax = fig.add_subplot(111)
# for k in range(middle.shape[0]):
#     x = [up[k, 0], middle[k, 0], down[k, 0]]
#     y = [up[k, 1], middle[k, 1], down[k, 1]]
#     ax.plot(x, y, color = 'g', linewidth = 2, alpha = 0.5)
# 
# ax.axis('equal')
# 
# plt.show()
# =============================================================================
