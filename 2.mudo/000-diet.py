import cv2 as cv
import os
import uuid

base_path = 'D:\workspace\python_study\OpenCV\97-haar\images-2006'
file_list = os.listdir(base_path)

for idx, file in enumerate(file_list) : 
    if not (idx % 4) == 0 :
        pass
        # print('패스')
    else :
        os.remove(os.path.join(base_path, file))
        # print('삭제')

print('끄끝')