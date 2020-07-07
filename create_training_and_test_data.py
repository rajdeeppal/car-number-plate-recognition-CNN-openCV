# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cv2
import os
import pickle


# List of images
data = []

# List of labels
labels = []

input_path = 'dataset'

for root, dirs, files in os.walk(input_path):

   
    for dir in dirs:

        print(" Class : \t \t " + dir)
        for filename in os.listdir(input_path + "/" + dir):

            
            if filename.endswith('.jpg'):

                img = cv2.imread(input_path + "/" + dir + "/" + filename)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                data.append(gray)
                labels.append(dir)


pickle.dump(data, open("data.pickle", "wb"))
pickle.dump(labels, open("labels.pickle", "wb"))

print('Length data : ' + str(len(data)))
print('Length labels : ' + str(len(labels)))
print('Processs finished !')
