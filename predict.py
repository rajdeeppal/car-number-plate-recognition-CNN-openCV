# -*- coding: utf-8 -*-


import  imutils
import sys
import pandas as pd
import time
import pandas as pd
import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from matplotlib import pyplot as plt
from digit_crop import plate_segmentation

# Load model
model = load_model('cnn_classifier.h5')

image = cv2.imread('car.jpeg')
print( image.shape )
image = imutils.resize(image, width=500)

#cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("1 - Grayscale Conversion", gray)

gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("2 - Bilateral Filter", gray)

edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("4 - Canny Edges", edged)

new, cnts , _ = cv2.findContours( edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
cnts =sorted(cnts,key=cv2.contourArea,reverse=True)[:10] 
NumberPlateCnt = None 

count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        x , y , w , h = cv2.boundingRect(c)
        NumberPlateCnt = approx 
        break
#print( NumberPlateCnt )
cropped = image [ y : y + h , x : x + w ]
cv2.imwrite('croping.jpeg' , cropped)
print(cropped.shape)

w = cropped.shape[ 1 ]
h = cropped.shape[ 0 ]
 
scale_percent = 260
if ( h < 50 ) and ( w < 300 ):
    h = int(h * scale_percent / 100) 
    w = int(w * scale_percent / 100)
    dim = ( w , h )
    resized = cv2.resize( cropped , dim , interpolation = cv2.INTER_AREA )
    print( resized.shape )
    cv2.imwrite('croping.jpeg' , resized )
# Detect chars
    
digits = plate_segmentation('croping.jpeg')
res = []
for d in digits:

    d = np.reshape(d, (1,28,28,1))
    d = d.astype('float32')
    d /= 255
    out = model.predict(d)
    res.append(out)
    # Get max pre arg
    p = []
    precision = 0
    for i in range(len(out)):
        z = np.zeros(36)
        z[np.argmax(out[i])] = 1.
        precision = max(out[i])
        p.append(z)
    prediction = np.array(p)

    # Inverse one hot encoding
    alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    classes = []
    for a in alphabets:
        classes.append([a])
    ohe = OneHotEncoder(handle_unknown='ignore', categorical_features=None)
    ohe.fit(classes)
    pred = ohe.inverse_transform(prediction)

    if precision > 0.4:
        print('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))