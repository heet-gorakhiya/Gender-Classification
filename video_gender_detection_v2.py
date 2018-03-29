import os
import cv2
import numpy as np
import keras
from keras.models import load_model



face_cascade = cv2.CascadeClassifier(
    os.getcwd() + '/haarcascade_frontalface_default.xml')


CNN_model = load_model('CNN_model_trained_64_v1.model')
face_size = [64, 64]


def predict_output(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 7)
    faces = np.array(faces)
    for (x, y, w, h) in faces:
        cropped_img = img[y-50:y + h + 50, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if cropped_img.shape[0] >= face_size[0] and cropped_img.shape[1] >= face_size[1]:
            cropped_img = cv2.resize(cropped_img, (face_size[0], face_size[1]))
            cropped_img_norm = cropped_img / 255.0
            x_pred = CNN_model.predict(np.expand_dims(cropped_img_norm, axis=0))
            if x_pred < 0.5:
                display_string = "Female"
                # print("Female")
                cv2.putText(img, display_string, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            else:
                display_string = "Male"
                # print("Male")
                cv2.putText(img, display_string, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    #cv2.imshow('gender_detection', img)
    out.write(img)


cap = cv2.VideoCapture('test_2.mp4')
skipFrame = 500
i = 0
endFrame = 1200
resize_Shape = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_1.avi', fourcc, 20.0, resize_Shape)
while(cap.isOpened()):
    ret, frame = cap.read()
    if i > skipFrame:
        #frame = cv2.resize(frame, resize_Shape)
        predict_output(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if i > endFrame:
        break
    i = i + 1

cap.release()
out.release()









