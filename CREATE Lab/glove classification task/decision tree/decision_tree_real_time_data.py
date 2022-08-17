# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:30:12 2021

@author: Amirhossein
"""

from sklearn import tree
from functions import load_all_data, find_accuracy
import numpy as np
import serial
import time


X_train, Y_train = load_all_data()
decision_tree = tree.DecisionTreeClassifier(max_depth = 4, class_weight = 'balanced')
decision_tree.fit(X_train, Y_train)

train_acc = find_accuracy(decision_tree, X_train, Y_train)
print("train accuracy\t\t:\t{}".format(train_acc))


duration = 30


ser = serial.Serial(port="COM8",
                    baudrate="460800",
                    bytesize=8,
                    timeout=2,
                    stopbits=serial.STOPBITS_ONE)

captured_data = []
start_time = time.time()
while time.time() - start_time < duration:
    if(ser.in_waiting > 0):
        try:
            ser.flushInput()
            ser.flushOutput()
            recived_data = ser.readline().decode('Ascii')
            recived_data = recived_data.split(",")[:16]
            recived_data = [int(x) for x in recived_data]
            if len(recived_data) == 16 and max(recived_data) <= 99:
                print(recived_data)
                input_vector = np.array(recived_data) / 99
                
                captured_data.append(np.array(recived_data))
                
                y_pred = decision_tree.predict(np.expand_dims(input_vector, axis = 0))
                #print(recived_data)
                #print(y_pred)
            
            
        except:
            pass
        
ser.close()


# =============================================================================
# captured_data = np.array(captured_data)
# max_min_recived = np.max(captured_data, axis = 0) - np.min(captured_data, axis = 0)
# glove_data = np.load("data/glove data.npy")
# max_min_train = np.max(glove_data, axis = 0) - np.min(glove_data, axis = 0)
# 
# print(max_min_train)
# print(max_min_recived)
# =============================================================================
