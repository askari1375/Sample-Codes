# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:20:16 2021

@author: Amirhossein
"""

import serial
import time


duration = 30


ser = serial.Serial(port="COM8",
                    baudrate="460800",
                    bytesize=8,
                    timeout=2,
                    stopbits=serial.STOPBITS_ONE)

data = []
times = []

start_time = time.time()
while time.time() - start_time < duration:
    if(ser.in_waiting > 0):
        try:
            recived_data = ser.readline().decode('Ascii')
            recived_data = recived_data.split(",")[:16]
            recived_data = [int(x) for x in recived_data]
            data.append(recived_data)
            times.append(time.time() - start_time)
        
            print(recived_data)
            
        except:
            pass

ser.close()
