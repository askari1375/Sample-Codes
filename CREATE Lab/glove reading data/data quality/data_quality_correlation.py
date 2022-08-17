# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:39:22 2021

@author: Amirhossein
"""

import numpy as np
import matplotlib.pyplot as plt



glove_data = np.load("processed data type 1/glove data.npy")
camera_data = np.load("processed data type 1/camera data.npy")


camera_data = camera_data.reshape((-1, 63))

data = np.concatenate((glove_data, camera_data), axis = 1)


corr = np.corrcoef(np.transpose(data))

corr = corr[0:16, 16:]

#plt.imshow(corr, cmap='hot', interpolation='nearest')
#plt.show()


data = glove_data[np.max(glove_data, axis = 1) < 100, :]
for k in range(data.shape[1]):
    plt.plot(data[:, k])

std = np.std(data, axis = 0)

plt.figure()
plt.plot(std)

