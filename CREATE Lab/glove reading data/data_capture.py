# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:55:34 2021

@author: Amirhossein
"""

import cv2
import mediapipe as mp
import serial
import numpy as np
from data_plotter import DataPlotter
import time
import os


def translate_to_origin(points):    
    coordinates = np.copy(points)
    origin = coordinates[0, :].copy()
    for idx in range(coordinates.shape[0]):
        coordinates[idx, :] -= origin
    return coordinates

def find_rotation_matrix(points):
    coordinates = np.copy(points)
    p_1 = coordinates[0, :]
    p_2 = coordinates[5, :]
    p_3 = (coordinates[9, :] + coordinates[13, :] + coordinates[17, :]) / 3
    
    v_1 = np.cross(p_2 - p_1, p_3 - p_1)
    v_1 = v_1 / np.linalg.norm(v_1)
    v_2 = p_2 - p_1
    v_2 = v_2 / np.linalg.norm(v_2)
    v_3 = np.cross(v_2, v_1)
    
    R = np.linalg.inv(np.array([v_1, v_2, v_3]))
    
    return R

def rotate_points(points):
    rotation_matrix = find_rotation_matrix(points)
    coordinates = np.copy(points)
    
    for k in range(coordinates.shape[0]):
        coordinates[k, :] = np.matmul(coordinates[k, :], rotation_matrix)
    return coordinates



def plot_rotated_hand(coordinates, new_coordinates, data_plotter):
    a = np.expand_dims(coordinates, axis = 0)
    b = np.expand_dims(new_coordinates, axis = 0)
    c = np.concatenate((a, b), axis = 0)            
    data_plotter.plot_hands(c)    


def coordinates_processor(coordinates):
    coordinates = translate_to_origin(coordinates)
    new_coordinates = rotate_points(coordinates)
    return new_coordinates
    

def image_processing_process(image, hands, mp_drawing, mp_hands):
    coordinates = np.zeros((21, 3))
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)            
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_hight, image_width, _ = image.shape
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            for k in range(21):
                coordinates[k, 0] = hand_landmarks.landmark[k].x * image_width
                coordinates[k, 1] = hand_landmarks.landmark[k].y * image_hight
                coordinates[k, 2] = hand_landmarks.landmark[k].z * image_width
            
            
            
            mp_drawing.draw_landmarks(image, hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    
    return coordinates



use_plotter = False


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if use_plotter:
    data_plotter = DataPlotter()


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands = 1)

ser = serial.Serial(port="COM8",
                    baudrate="460800",
                    bytesize=8,
                    timeout=2,
                    stopbits=serial.STOPBITS_ONE)






start_time = time.time()
counter = 0

captured_coordinates = []
captured_glove_data = []
glove_times = []
camera_times = []



while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    coordinates = image_processing_process(image, hands, mp_drawing, mp_hands)
    new_coordinates = coordinates_processor(coordinates)
    camera_time = time.time()
    
    
    
    if(ser.in_waiting > 0):
        try:
            recived_data = ser.readline().decode('Ascii')
            recived_data = recived_data.split(",")[:16]
            recived_data = [int(x) for x in recived_data]
            glove_time = time.time()
            
        
            #print(recived_data)            
        except:
            pass
    
    
    
    
    if use_plotter:
        plot_rotated_hand(coordinates, new_coordinates, data_plotter)
    
    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord(' '):
        if len(recived_data) == 16:
            captured_glove_data.append(recived_data)            
            glove_times.append(glove_time)
            captured_coordinates.append(new_coordinates)    
            camera_times.append(camera_time)
        if len(glove_times) % 50 == 0:
            print(len(glove_times))
    
    counter += 1
    
    
            
duration = time.time() - start_time

hands.close()
ser.close()   
cap.release()
cv2.destroyAllWindows()


captured_glove_data = np.array(captured_glove_data)
captured_coordinates = np.array(captured_coordinates)
camera_times = np.array(camera_times)
glove_times = np.array(glove_times)

time.sleep(1)

file_names = os.listdir("captured data")
file_names = [name for name in file_names if name[:11] == "camera data"]
file_names.sort()
if len(file_names) == 0:
    new_file_number = 1
else:
    name = file_names[-1]
    new_file_number = int(name.split(" ")[-1].split(".")[0]) + 1



print("--------------------------------------------------")
print("capture frequency : {}".format(counter / duration))
print("counter : {}".format(counter))
print(captured_glove_data.shape)
print(captured_coordinates.shape)

np.save("captured data/glove data {}.npy".format(new_file_number), captured_glove_data)
np.save("captured data/camera data {}.npy".format(new_file_number), captured_coordinates)
np.save("captured data/glove times {}.npy".format(new_file_number), glove_times)
np.save("captured data/camera times {}.npy".format(new_file_number), camera_times)

# =============================================================================
# if __name__ == "__main__":
#     pass
# =============================================================================
