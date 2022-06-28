import os
import cv2 as cv
from cv2 import resize

path = 'train/unmask/'
target_list = os.listdir(path)

for target in target_list :
    src = cv.imread(path+target)
    resize_src = cv.resize(src, (64,64))
    cv.imwrite('here/'+target, resize_src)