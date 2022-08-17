# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:16:05 2021

@author: Amirhossein
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from trunk_shape_simulator import create_trunk_image, create_image_from_side_points


class Trunk:
    def __init__(self, image_name):
        original_image = cv2.imread("targets/{}".format(image_name)) / 255
        predicted_image = cv2.imread("predictions/{}".format(image_name))
        predicted_image = cv2.resize(predicted_image, (original_image.shape[1], original_image.shape[0])) / 255
        self.image = np.mean(original_image, axis = 2)
        self.prediction = np.mean(predicted_image, axis = 2)
        self.angles_change = np.loadtxt("fit trunk/best angles/{}.txt".format(image_name[:-4]))
        self.trunk_image, self.trunk_points = create_trunk_image(self.angles_change)
        
    
    def show_images(self):
        cv2.imshow("original image", self.image)
        cv2.imshow("predicted image", self.prediction)
        cv2.imshow("trunk image", self.trunk_image)
    
    def find_IoU(self, length, threshold = 0.9):
        n = self.angles_change.shape[0]
        k = np.floor(length * n).astype("int32")
        alpha = length - k / n
        
        new_trunk = self.trunk_points[:k + 1, :]
        if alpha > 0:
            last_points = (1 - alpha) * self.trunk_points[k, :] + alpha * self.trunk_points[k + 1, :]
            last_points = np.reshape(last_points, (1, 4))
            new_trunk = np.concatenate((new_trunk, last_points), axis = 0)
        
        new_mask = create_image_from_side_points(new_trunk, self.image.shape)
        
        prediction_mask = np.multiply(self.prediction, new_mask)
        
        prediction_mask[prediction_mask < threshold] = 0
        prediction_mask[prediction_mask >= threshold] = 1
        
        IoU = np.sum(prediction_mask) / np.sum(new_mask)
        return IoU
        
        
        
    
image_names = os.listdir("fit trunk")[:-1]

n_l = 100
length = [(k + 1) / n_l for k in range(n_l)]
ious = np.zeros((n_l, len(image_names)))

for idx in tqdm(range(len(image_names))):
    image_name = image_names[idx]
    trunk = Trunk(image_name)
    for k in range(n_l):
        ious[k, idx] = trunk.find_IoU(length[k])

mean_ious = np.mean(ious, axis = 1)

fig, ax = plt.subplots()
ax.plot(mean_ious, marker = '.')
ax.set_ylim([0, 1])
ax.grid()
ax.set_title("Intersection over union over length")
ax.set_xlabel("Length")
ax.set_ylabel("IoU")
fig.savefig("Intersection over Union plot.png", dpi=200)
plt.show()



