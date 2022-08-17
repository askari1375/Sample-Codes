# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:15:36 2021

@author: Amirhossein
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_hand(coordinates, image_number):
    image_path = 'image results/'
    fig_z = plot_hand_specific_view(coordinates, 90, 0, title="Z view")
    fig_x = plot_hand_specific_view(coordinates, 0, 0, title="X view")
    fig_y = plot_hand_specific_view(coordinates, 90, 90, title="Y view")
    
    
    for elev in range(0, 180, 5):
        fig = plot_hand_specific_view(coordinates, elev, 0, title="elevation = {}".format(elev))
        fig.savefig("{}{}_elevation {}.png".format(image_path, image_number, elev))
    
    fig_x.savefig("{}{}_X View.png".format(image_path, image_number))
    fig_y.savefig("{}{}_Y View.png".format(image_path, image_number))
    fig_z.savefig("{}{}_Z View.png".format(image_path, image_number))

def plot_hand_specific_view(coordinates, elevation = 80, azimuth = 0, title = None):
    
    connections = [(0, 1, 2, 3, 4),
                   (0, 5, 6, 7, 8),
                   (9, 10, 11, 12),
                   (13, 14, 15, 16),
                   (0, 17, 18, 19, 20),
                   (5, 9, 13, 17)
                   ]
    
    figure = plt.figure(figsize = (6, 6))    
    ax = Axes3D(figure, adjustable = 'datalim')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], marker='o')
    
    for connection in connections:
        for k in range(len(connection) - 1):
            x1 = coordinates[connection[k], 0]
            y1 = coordinates[connection[k], 1]
            z1 = coordinates[connection[k], 2]
            x2 = coordinates[connection[k + 1], 0]
            y2 = coordinates[connection[k + 1], 1]
            z2 = coordinates[connection[k + 1], 2]
            plt.plot([x1, x2], [y1, y2], [z1, z2], color = 'g')
    
    ax.set_xlim3d(0, 700)
    ax.set_ylim3d(0, 700)
    ax.set_zlim3d(-350, 350)
    ax.view_init(elev = elevation, azim = azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title is not None:
        plt.title(title)
    plt.show()
    
    return figure




mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

coordinates = np.zeros((21, 3))
captured_image_counter = 0

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
        
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
            
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            elif pressedKey == ord(' '):
                captured_image_counter += 1
                plot_hand(coordinates, captured_image_counter)
                
    
cap.release()
cv2.destroyAllWindows()