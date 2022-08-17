# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:29:35 2021

@author: Amirhossein
"""

import serial
import time
import matplotlib.pyplot as plt
import numpy as np

duration = 15

ser = serial.Serial(port="COM6",
                    baudrate="9600",
                    bytesize=8,
                    timeout=2,
                    stopbits=serial.STOPBITS_ONE)

data = []
times = []

start_time = time.time()
while time.time() - start_time < duration:
    if(ser.in_waiting > 0):
        try:
            recived_data = int(ser.readline().decode('Ascii'))
            data.append(recived_data)
            times.append(time.time() - start_time)
        
            print(recived_data)
        except:
            pass

ser.close()

data = np.array([times, data])

plt.plot(data[0, :], data[1, :])



