# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:58:36 2021

@author: Amirhossein
"""

import numpy as np
import cv2
import time


def show_video(frames, skip_rate):
    
    delay = 5 * skip_rate    
    for k in range(frames.shape[0]):
        frame = frames[k, ...]
        cv2.imshow("frame", frame)
        if cv2.waitKey(delay) and 0xFF == ord('q'):
            break


def load_video(path, skip_rate):
    start_time = time.time()
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    frames = []
    c = 0
    while cap.isOpened():
        
        ret, image = cap.read()    
        if ret:  
            if c % skip_rate == 0:
                w = int(image.shape[1] * scale)
                h = int(image.shape[0] * scale)
                image = cv2.resize(image, (w, h))
                frames.append(image / 255)
            c += 1
        
        else:
            break    
    
    frames = np.array(frames)    
    print("all frames be reed - duration\t{}".format(time.time() - start_time))    
    return frames



def create_croped_images(frames):
    
    start_time = time.time()
    
    croped_images = []
    left_border = int(frames.shape[2] * 0.3)
    right_border = frames.shape[2] - int(frames.shape[2] * 0.2)
    for k in range(frames.shape[0]):
        image = frames[k, ...]
        img = image[:, left_border:right_border, :]        
        croped_images.append(img)

    croped_images = np.array(croped_images)
    print("croped images prepered - duration\t{}".format(time.time() - start_time))
    return croped_images



video_path = 'data/1.mp4'
scale = 1
skip_rate = 10


frames = load_video(video_path, skip_rate)
frames = create_croped_images(frames)

gray_images = []
for k in range(frames.shape[0]):
    image = frames[k, ...]
    img_gray = np.mean(image, axis = 2)
    
    img_gray[img_gray < 0.8] = 0
    
    gray_images.append(img_gray)
    
gray_images = np.array(gray_images)






show_video(gray_images, skip_rate)

# =============================================================================
# start_time = time.time()
# std_map = np.std(frames, axis = 0)
# print("std map prepered - duration\t{}".format(time.time() - start_time))
# std_map = std_map / np.max(std_map)
# std_map = np.mean(std_map, axis = 2)
# cv2.imshow("std map", std_map)
# 
# mask = np.zeros(std_map.shape)
# mask[std_map < 0.3] = 1
# 
# background_image = np.median(frames, axis = 0)
# 
# for k in range(frames.shape[0]):
#     frames[k] = frames[k] - background_image
# 
# show_video(frames, skip_rate)
# =============================================================================
