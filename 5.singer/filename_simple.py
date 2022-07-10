import shutil
import cv2 as cv
import os
import uuid

cascade_filename = './haarcascade_frontalface_alt.xml'
cascade = cv.CascadeClassifier(cascade_filename)

base_path = './dataset/full_img/'
dst_path = './dataset/uuid/'
singer_list = os.listdir(base_path)

for idx, singer_folder in enumerate(singer_list) : 
    print(singer_folder)
    os.mkdir(dst_path+singer_folder)
    full_img_list = os.listdir(base_path+singer_folder)
    crop_idx = 0
    for full_img in full_img_list :
        ext = full_img.split('.')[-1]
        file_name = str(uuid.uuid4())[:6]

        shutil.copy(os.path.join(base_path, singer_folder, full_img),
                    os.path.join(dst_path, singer_folder, file_name+'.'+ext))