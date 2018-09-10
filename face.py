#from keras import backend as K
import time
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *

PADDING = 50

#from matplotlib.pyplot import imshow
import face_recognition

        
def face_recognizer(frame):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """

    #frame= allignFace(frame)

    boxes = face_recognition.face_locations(frame,model='cnn')
    if boxes ==[]:
        boxes=[(0,frame.shape[0],frame.shape[1],0)]
        #print('empty')    
    #print(boxes)
    encoding = face_recognition.face_encodings(frame, boxes)      

    return encoding  
        
def resize(img):
    height, width = img.shape[:2] # without channel
    #print(height, width)
    area= height* width
    #print(area)
    scale = 100000.0/area
    dim=(int(width*scale),int(height*scale)) 
    #print(dim)
    resized = cv2.resize(img, dim, interpolation =cv2.INTER_AREA)
    #imshow(resized) 
    return resized


#load image

img = cv2.imread('images\Ahmed_b.jpg')
print('images\Ahmed_b.jpg')
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img=resize(img)
encoding= face_recognizer(img)
print('encod',encoding)
