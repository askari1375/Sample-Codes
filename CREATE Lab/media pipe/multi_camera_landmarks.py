# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:58:03 2021

@author: Amirhossein
"""

import cv2
import mediapipe as mp
import numpy as np
import time

from multi_camera_data_analysis import DataAnalyzer



class HandProcessor:
    
    def __init__(self, camera_number):
        
        self.camera_number = camera_number
        self.cap = cv2.VideoCapture(camera_number)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
        self.hands = self.mp_hands.Hands(min_detection_confidence = 0.7,
                                         min_tracking_confidence = 0.7,
                                         max_num_hands = 1)
    
    def process_one_frame(self):
        if self.cap.isOpened():
        
            success, image = self.read_image_from_camera()
                
            hands_number, image, coordinates, image_shape = self.process_image(image)
                                    
            return image, coordinates, image_shape, hands_number
    
    
    def release_camera(self):
        self.cap.release()
        


    def read_image_from_camera(self):
        
        success, image = self.cap.read()
        if not (success):
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            return success, image
      
        image = cv2.flip(image, 1)
        cv2.rectangle(image, (10, 10), (190, 50), (255, 255, 100), -1)
        cv2.putText(image, 'Camera {}'.format(self.camera_number + 1),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
      
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        return success, image

    def process_image(self, image):
        
        coordinates = np.zeros((21, 3))
        
        results = self.hands.process(image)        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_hight, image_width, _ = image.shape
        
        if results.multi_hand_landmarks is None:
            hands_number = 0
        else:
            hands_number = len(results.multi_hand_landmarks)
    
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for k in range(21):
                    coordinates[k, 0] = hand_landmarks.landmark[k].x * image_width
                    coordinates[k, 1] = hand_landmarks.landmark[k].y * image_hight
                    coordinates[k, 2] = hand_landmarks.landmark[k].z * image_width
        
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        return hands_number, image, coordinates, (image_hight, image_width)


class MultiCameraHandler:
    
    def __init__(self, number_of_cameras):
        
        self.target_number_of_cameras = number_of_cameras
        self.available_cameras = self.find_available_cameras()
        
        self.hand_processors = []        
        self.data_analyzer = DataAnalyzer(self.available_cameras)

        
        for k in range(self.available_cameras):
            self.hand_processors.append(HandProcessor(camera_number = k))
    
    
    
    
    def find_available_cameras(self):
        c = 0
        for k in range(self.target_number_of_cameras):
            cap = cv2.VideoCapture(k)
            if cap is not None and cap.isOpened():
                c += 1
        if c < self.target_number_of_cameras:
            print("{} from {} cameras available".format(c, self.target_number_of_cameras))
        else:
            print("All cameras are available")        
        return c
    
    
    
    
    
    def start(self):
        
        while True:
            
            available_hands = 0
            
            images = []
            
            for k, hand_processor in enumerate(self.hand_processors):
                image, coordinates, image_shape, hands_number  = hand_processor.process_one_frame()
                self.data_analyzer.set_data(k, coordinates, image_shape)
                images.append(image)
                
                available_hands += 1 if hands_number > 0 else 0
            
            cv2.imshow('MediaPipe Hands', np.concatenate(images, axis = 0))
            
            
            if available_hands == self.available_cameras:
                self.data_analyzer.add_angles_to_log_history()

                if self.data_analyzer.data_plotter.check_hand_plot_time(1 / 30):                    
                    self.data_analyzer.plot_all_hands()
                    self.data_analyzer.data_plotter.last_hand_plot_time = time.time()
                    
# =============================================================================
#                 if self.data_analyzer.data_plotter.check_error_plot_time(1):
#                     self.data_analyzer.plot_angles_log_error()
#                     self.data_analyzer.data_plotter.last_error_plot_time = time.time()
# =============================================================================
                
                
            
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == ord('q'):
                break
            if pressed_key == ord(' '):
                self.data_analyzer.plot_angles_log_error()

            
        
        for hand_processor in self.hand_processors:            
            hand_processor.release_camera()

        cv2.destroyAllWindows()
    
      
            
     
        

multi_camera_handler = MultiCameraHandler(number_of_cameras = 2)
multi_camera_handler.start()
