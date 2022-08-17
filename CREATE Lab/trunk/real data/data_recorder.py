# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:25:13 2021

@author: Amirhossein
"""

import cv2
import numpy as np
import time
import os
from image_preprocessing import input_image_processor, output_image_processor



capture_delay = 1
start_delay = 5
record_allowed = False


cap_1 = cv2.VideoCapture(0)
cap_2 = cv2.VideoCapture(2)

start_time = time.time()
saving_started = False

c = len(list(os.walk('data/input/'))[0][2])


while True:
    
    success_1, image_1 = cap_1.read()
    success_2, image_2 = cap_2.read()    
    
    if not (success_1 and success_2):
        print("Ignoring empty camera frame.")
    
    
    image_1 = input_image_processor(image_1, 0.5)
    image_2 = output_image_processor(image_2, 0.53)
    img = np.concatenate((image_1, image_2), axis = 1)
    cv2.imshow("result", img)
    pressed_key = cv2.waitKey(10) & 0xFF
    if pressed_key == ord('q'):
        break
    
    if not saving_started:
        if time.time() - start_time > start_delay:
            saving_started = True
            print("Capturing started")
    else:
        if record_allowed:
            if time.time() - start_time > capture_delay:
                cv2.imwrite("data/input/{}.png".format(c), image_1)
                cv2.imwrite("data/output/{}.png".format(c), image_2)
                
                c += 1
                start_time = time.time()
    

