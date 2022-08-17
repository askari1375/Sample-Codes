# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:30:42 2021

@author: Amirhossein
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

for phase in np.linspace(0, 10*np.pi, 500):
    line1.set_ydata(np.sin(x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()
    



# =============================================================================
# x = np.linspace(0, 6*np.pi, 100)
# y = np.sin(x)
# z = np.cos(x)
# 
# figure = plt.figure(figsize = (6, 6))    
# ax = Axes3D(figure, adjustable = 'datalim')
# 
# curve, = ax.plot(x, y, z, marker = '.', linestyle='None')
# 
# 
# for phase in np.linspace(0, 10*np.pi, 500):
#     curve.set_xdata(x)
#     curve.set_ydata(np.sin(x + phase))
#     curve.set_3d_properties(np.cos(x + phase))
#     figure.canvas.draw()
#     figure.canvas.flush_events()
# =============================================================================



