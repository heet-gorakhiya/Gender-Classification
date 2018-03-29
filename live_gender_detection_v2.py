import os
import cv2
import numpy as np
import keras
from keras.models import load_model
import warnings
import time
warnings.filterwarnings("ignore")

face_cascade = cv2.CascadeClassifier( os.getcwd() + '/haarcascade_frontalface_default.xml')

CNN_model = load_model('CNN_model_trained_64_v1.model')
face_size = [64,64]

def predict_output(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    faces = np.array(faces)
    if faces.size:
        [x,y,w,h] = faces[0]
        cropped_img = img[y-50:y+h+20, x:x+w]
        cropped_img = np.array(cropped_img).astype(np.float32)
        cropped_img = cropped_img/255.0
        display_img = cropped_img	
        if cropped_img.shape[0] >= face_size[0] and cropped_img.shape[1] >= face_size[1]:
            cropped_img = cv2.resize(cropped_img,(face_size[0],face_size[1]))
            x = CNN_model.predict(np.expand_dims(cropped_img,axis=0))
            if x < 0.5:
            	display_string = "Female"
            	cv2.putText(display_img,display_string,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,100))
            else:
            	display_string = "Male"
            	cv2.putText(display_img,display_string,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow('gender_detection',display_img)
            #

cap = cv2.VideoCapture(0)
fps = 0
t1 = time.time()
while(True):
    ret, frame = cap.read()
    t2 = time.time()
    if t2 - t1 >= 1:
    	print (fps,'fps')
    	fps = 0
    	t1 = time.time()
    fps = fps + 1
    predict_output(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
