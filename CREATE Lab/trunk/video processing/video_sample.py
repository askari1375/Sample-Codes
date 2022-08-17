# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:45:50 2021

@author: Amirhossein
"""

import numpy as np
import cv2


cap = cv2.VideoCapture('data/1.mp4')

if not cap.isOpened():
    print("Error opening video file")


while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret:
        
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(5) and 0xFF == ord('q'):
            break
        
    else:
        break


