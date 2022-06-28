#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import glob
import cv2 as cv

#%%
model = keras.models.load_model('0001.h5')
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
xml = 'haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(xml)
#%%

while(True):
    ret, frame = cap.read() # 한장 씩 읽는다.
    frame = cv.flip(frame, 1) #  플립!
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.03, 5, minSize=(200,200)) #  튜플을 반환한다
    
    if len(faces) : 
        for face in faces :
            x,y,w,h, = face            
            cv.rectangle(frame, (x,y), (x+w,y+h), (255,125,125), 3)
            cropped_img = frame[y:y+h, x:x+w]
            cropped_img = cv.resize(cropped_img, (64,64))
            test_set = np.array([cropped_img])
            # print(test_set.shape)
            test_set = test_set.reshape(1, 64, 64, 3)
            test_set = np.asarray(test_set)
            results = model.predict(test_set)
            print(results)
            if np.argmax(results) == 0 :
                cv.putText(frame, 'half-masked', (x,y), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            if np.argmax(results) == 1 :
                cv.putText(frame, 'masked', (x,y), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            if np.argmax(results) == 2 :
                cv.putText(frame, 'unmasked', (x,y), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)



    cv.imshow('cam1', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cv.destroyAllWindows()
# %%
