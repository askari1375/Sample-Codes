# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:55:34 2021

@author: Amirhossein
"""


import serial
import numpy as np
import cv2
import os


def get_new_file_number(file_path):
    file_names = os.listdir(file_path)
    file_names.sort()
    if len(file_names) == 0:
        file_number = 1
    else:
        file_number = int(file_names[-1].split(" ")[-1][:-4]) + 1
    
    return file_number


total_shapes = 6

ser = serial.Serial(port="COM8",
                    baudrate="460800",
                    bytesize=8,
                    timeout=2,
                    stopbits=serial.STOPBITS_ONE)



cv2.imshow("window", np.zeros((500, 500, 3)))

captured_data = []
for k in range(total_shapes):
    captured_data.append([])

print("Ready to capture data ...")
while True:
    

    if(ser.in_waiting > 0):
        try:
            recived_data = ser.readline().decode('Ascii')
            recived_data = recived_data.split(",")[:16]
            recived_data = [int(x) for x in recived_data]
            
            
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            
            for n in range(total_shapes):
                if pressedKey == ord('{}'.format(n)):
                    print(n)
                    if len(recived_data) == 16 and max(recived_data) <= 99:
                        captured_data[n].append(recived_data)
                    total_captured_data = sum([len(x) for x in captured_data])
                    if total_captured_data % 100 == 0:
                        print("Total captured data\t:\t{}".format(total_captured_data))
            
        except:
            pass
    
       

ser.close()
cv2.destroyAllWindows()

for n in range(total_shapes):
    if len(captured_data[n]) > 0:
        data = np.array(captured_data[n]) / 99
        if not os.path.exists('data/{}'.format(n)):
            os.mkdir('data/{}'.format(n))
        file_number = get_new_file_number('data/{}'.format(n))
        np.save('data/{}/{} - {}.npy'.format(n, n, file_number), data)
    

for n in range(total_shapes):
    print("captured data for {}\t:\t{}".format(n, len(captured_data[n])))

