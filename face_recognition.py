# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 19:35:55 2018

@author: @adriantoto
"""

#Libraries
import cv2
import numpy as np

#Classifier
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#input
videoCam = cv2.VideoCapture(0)

while(True):
    cond, frame = videoCam.read()
   #gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   #face detection using classifier
    muka = face.detectMultiScale(gray, 1.3, 5)
   #rectangle
    for(x, y, w, h) in muka:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
   
   #output          
    cv2.imshow("Face", frame)
    if((cv2.waitKey(1) & 0xff) == ord('q')):
        break

#after break
videoCam.release()
cv2.destroyAllWindows()
