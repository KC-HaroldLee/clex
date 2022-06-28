import glob
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2 as cv
import time
import datetime

base_path = ''
dataset_path = os.path.join(base_path, 'results-2011')

paths = glob.glob(dataset_path+'/*/*.png') # 될까?
paths = np.random.permutation(paths)

print(paths[0])
    # tensorFlow/notMNIST_small\E\Q2FsdmVydCBNVCBMaWdodC50dGY=.png
ind_val = np.array([plt.imread(paths[i]) for i in range(len(paths))])
dep_val = np.array([paths[i].split('\\')[1] for i in range(len(paths))])

print(ind_val.shape, dep_val.shape)
    # (990, 64, 64, 3) (990,) <늘수록 바뀜
pass
# 변수 전처리
ind_val = ind_val.reshape(848, 64, 64, 3)
# 종속 = pd.get_dummies(종속)
dep_val = pd.get_dummies(dep_val) # 다행히 1차원
# dep_val = dep_val.reshape(990, 7)
print(ind_val.shape, dep_val.shape)
    # (990, 64, 64, 3) (990, 7) <늘수록 바뀜


X = tf.keras.layers.Input(shape=[64, 64, 3])

Y = tf.keras.layers.Conv2D(filters=8, kernel_size=3, dilation_rate=1, activation='swish')(X)
# HS2 = tf.keras.layers.MaxPool2D()(Y)

Y = tf.keras.layers.Conv2D(filters=20, kernel_size=3, activation='swish')(Y)
# HS4 = tf.keras.layers.MaxPool2D()(Y)

Y = tf.keras.layers.Flatten()(Y)

Y = tf.keras.layers.Dense(units=360, activation='swish')(Y)
# Y = tf.keras.layers.Dense(units=450, activation='swish')(Y)
# Y = tf.keras.layers.Dense(units=100, activation='swish')(Y)
Y = tf.keras.layers.Dense(units=49, activation='swish')(Y)

Y = tf.keras.layers.Dense(units=7, activation='softmax')(Y)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 학습
model.fit(ind_val, dep_val, epochs=10)

# # 예측 
# pred = model.predict(ind_val[0:5])
# print(pd.DataFrame(pred).round(2))
 
# # 정답 확인
# print(dep_val[0:5])

val_count = 200
pred = model.predict(ind_val[0:val_count])

correct = 0
incorrect = 0
trust_list = []
current_trust = 0
for idx in range(len(pred)) :
    pred_text = pd.DataFrame(pred[idx:idx+1]).round(2)
    answer_text = dep_val[idx:idx+1]  

    guess = pred_text.values[0].argmax()
    trust = int(max(pred_text.values[0])*100)
    answer_no = answer_text.values[0].argmax()
    # print('예측 : {} ({}%) | 정답 : {}'.format(guess, trust, answer_no))
    if guess == answer_no :
        correct += 1
    else :
        pass
    trust_list.append(trust)
    current_trust = (current_trust*idx + trust)/(idx+1)
    
    # print(pred_text)
    # print(answer_text)
    # a = np.zeros((100,100,1), dtype=np.uint8)
    # cv.imshow('a', a)
    # cv.waitKey(0)

    # print('-------------------------')

print('총 정답 : {}/{}'.format(correct, val_count))
# sa1 = time.time()
a1 = int(sum(trust_list, 0.0)/len(trust_list))
print('확신 {}%'.format(a1))
# ea1 = time.time()

model.summary()

## 모델저장
# now = datetime.datetime.now()
# model.save(base_path+'\\my_model\\model{}{}{}{}{}{}({},{})'
#                 .format(now.year, now.month, now.day, now.hour, now.minute, now.second,
#                 correct, val_count))


# sa2 = time.time()
# a2np = np.array(trust_list)
# a2 = np.mean(a2np)
# ea2 = time.time()

# print('확신2 {}'.format(a1, ea2-sa2))
# print('확신3 {}'.format(int(current_trust)))


# pred = model.predict(ind_val[50:55])
# print(dep_val[5:55])
 

# pred = model.predict(ind_val[100:105])
# print(dep_val[100:105])




# cascade_filename = '.\\haar\\haarcascade_frontalface_alt.xml'

# cascade = cv.CascadeClassifier(cascade_filename)

# chk_file_path = os.path.join(base_path, 'chk')
# chk_file_list = os.listdir(chk_file_path)

# print('검증 파일 개수 : ', len(chk_file_list))

# for chk_file in chk_file_list :
#     file_path = os.path.join(chk_file_path, chk_file)
#     if os.path.isdir(file_path) :
#         continue
#     img_origin = cv.imread(file_path)

#     gray = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY) 

#     results = cascade.detectMultiScale(gray,            # 입력 이미지
#                                     scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
#                                     minNeighbors=5,  # 인접 객체 최소 거리 픽셀
#                                     minSize=(32,32)  # 탐지 객체 최소 크기
#                                     )

#     resize_path = os.path.join(chk_file_path, chk_file.split('.')[0])
#     if not os.path.isdir(resize_path) :
#         os.mkdir(resize_path)
        
#     for box_idx, box in enumerate(results):
            
#         x, y, w, h = box
#         img_drew = img_origin.copy()
#         cv.rectangle(img_drew, (x,y), (x+w, y+h), (255,255,255), thickness=0)
#         imgcrop = img_origin [y:y+h, x:x+w] # (149,149, 3)
#         imgcrop = cv.resize(imgcrop, dsize=(64,64)) # (64, 64, 3)

#         hash = str(uuid.uuid1())[:8]
#         cv.imwrite(resize_path+'\\'+hash+'.png', imgcrop)

#     chk_resized_file_list = os.listdir(resize_path)
#     for chk_resized_file in chk_resized_file_list :
#         file_path = os.path.join(resize_path, chk_resized_file)

#         image_to_model = plt.imread(file_path)

#         image_show = cv.cvtColor(image_to_model, cv.COLOR_RGB2BGR)
#         image_show = cv.resize(image_show, (256,256))
#         cv.imshow('aaa', image_show)
#         chk_val = np.array([image_to_model])
#         pred = model.predict(chk_val[0:1])
#         print(pd.DataFrame(pred).round(2))    
#         cv.waitKey(0)