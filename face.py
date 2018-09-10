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


def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=resize(img)
        '''
        if 'ezz_rot' in identity:
            img= allignFace(img)
            print('ezz_rot')
        ''' 

        database[identity] = face_recognizer(img)
        print(identity)
        #imshow(img)

        
    return database


        
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


#database = prepare_database()
def match_face(identity_match,encoding, database):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    min_dist = 100

    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        #results = face_recognition.compare_faces([db_enc], encoding)
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        #print('db_enc',db_enc)
        #print('encoding',encoding)
        
        dist = np.linalg.norm(db_enc[0] - encoding[0])
        
        print('%s  %s  %s' %(identity_match,name,1.0/dist))

        #print('distance for %s is %s: %s' %(name,1.0/dist, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.52:
        return None
    else:
        return str(identity)

def welcome_users(identities):
    """ Outputs a welcome audio message to the users """
    welcome_message = 'Welcome '

    if len(identities) == 1:
        welcome_message += '%s, have a nice day.' % identities[0]
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += 'have a nice day!'

        
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
database = prepare_database()

#load image
for file in glob.glob("match/*"):
    identity = os.path.splitext(os.path.basename(file))[0]
            
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=resize(img)
    '''
       if 'ezz_rot' in identity:
        img= allignFace(img)
        print('ezz_rot')
    '''

    #database[identity] = face_recognizer(img)
 
    #print(' ',identity,'Head')#,img.shape)
    encoding= face_recognizer(img)
    #print('encod',encoding.shape)

    identity= match_face(identity,encoding,database)
