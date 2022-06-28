#%%
from tensorflow import keras
#%%
model = keras.models.load_model('0001.h5')
# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import glob
import cv2 as cv
# %%
test_input = plt.imread('val/half-mask/cam_721.png')
print(test_input.shape)
# %%
model.layers[0].input_shape
# %%
base_path = ''
dataset_path = os.path.join(base_path, 'val')

paths = glob.glob(dataset_path+'/*/*.png') # 될까? 
test_set = np.array([plt.imread(paths[i]) for i in range(len(paths))])
print(test_set.shape)
test_set = test_set.reshape(68, 64, 64, 3)
test_set = np.asarray(test_set)
# %%
test_set
# %%

results = model.predict(test_set)
# %%

import random

random.shuffle(paths)
for i, file in enumerate(paths) :
    src = cv.imread(paths[i])
    src = cv.resize(src, (256,256))
    result = np.argmax(results[i])
    if result == 0 : 
        print('\rHALF-MASK                ', end='')
    if result == 1 : 
        print('\rMASKED                ', end='')
    if result == 2 : 
        print('\rUNMASK                ', end='')
    cv.imshow('src', src)
    cv.waitKey(0)

    

# %%
