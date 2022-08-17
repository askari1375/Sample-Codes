# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:06:21 2021

@author: Amirhossein
"""

import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


phase = 'train'

sub_folders = os.listdir("data/data {}".format(phase))
n = len(sub_folders)

data_list = []
labels_list = []
for folder_name in sub_folders:
    file_path = "data/data {}/{}/{}.npy".format(phase, folder_name, folder_name)
    data_list.append(np.load(file_path))
    labels_list.append(np.ones((data_list[-1].shape[0],)) * int(folder_name))

X = np.concatenate(data_list, axis = 0)
Y = np.concatenate(labels_list, axis = 0)
print(X.shape)
print(Y.shape)

start_time = time.time()
X_embeded_tsne = TSNE(n_components = 2).fit_transform(X)
print("duration\t:\t{}".format(time.time() - start_time))
start_time = time.time()
X_embeded_pca = PCA(n_components = 2).fit_transform(X)
print("duration\t:\t{}".format(time.time() - start_time))


fig = plt.figure(figsize = (20, 20))
for i in range(n):
    indices = Y == i
    plt.scatter(X_embeded_tsne[indices, 0], X_embeded_tsne[indices, 1], color = "rgbcmy"[i], label = i)

plt.legend()
fig.savefig("{} data - TSNE.png".format(phase))

fig = plt.figure(figsize = (20, 20))
for i in range(n):
    indices = Y == i
    plt.scatter(X_embeded_pca[indices, 0], X_embeded_pca[indices, 1], color = "rgbcmy"[i], label = i)

plt.legend()
fig.savefig("{} data - PCA.png".format(phase))

