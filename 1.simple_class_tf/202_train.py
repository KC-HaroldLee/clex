#%%
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
#%%
base_path = ''
dataset_path = os.path.join(base_path, 'train')

paths = glob.glob(dataset_path+'/*/*.png') # 될까?
paths = np.random.permutation(paths)

print(paths[0])
    # tensorFlow/notMNIST_small\E\Q2FsdmVydCBNVCBMaWdodC50dGY=.png
ind_val = np.array([plt.imread(paths[i]) for i in range(len(paths))])
dep_val = np.array([paths[i].split('\\')[1] for i in range(len(paths))])

print(ind_val.shape, dep_val.shape)
    # (990, 64, 64, 3) (990,) <늘수록 바뀜

#%%
# 변수 전처리
ind_val = ind_val.reshape(413, 64, 64, 3)
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
Y = tf.keras.layers.Dense(units=9, activation='swish')(Y)

Y = tf.keras.layers.Dense(units=3, activation='softmax')(Y)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 학습
model.fit(ind_val, dep_val, epochs=40)
#%%
model.save('0001.h5')
# %%
