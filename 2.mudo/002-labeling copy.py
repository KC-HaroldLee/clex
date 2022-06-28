import cv2 as cv
import os
import uuid
import shutil

cascade_filename = '.\\97-haar\\haarcascade_frontalface_alt.xml'
cascade = cv.CascadeClassifier(cascade_filename)


base_path = '97-haar'
images_path = os.path.join(base_path, 'images')
ends_path = os.path.join(images_path, 'ends')

file_list = os.listdir(images_path)
print('len(file_list) : ', len(file_list))

for idx, file in enumerate(file_list) : 
    file_path = os.path.join(images_path, file)
    img_origin = cv.imread(file_path, cv.IMREAD_REDUCED_COLOR_2)

    cv.imshow('a', img_origin)
    key = cv.waitKey(0)

    if key == ord('q') :
        pass

    if key == ord('w') :
        ends_file_path = os.path.join(ends_path, file)
        shutil.move(file_path, ends_file_path)

    if key == ord('d') :
        os.remove(file_path)

