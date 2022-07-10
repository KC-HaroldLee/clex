import cv2 as cv
import os
import shutil

cascade_filename = './haarcascade_frontalface_alt.xml'
cascade = cv.CascadeClassifier(cascade_filename)

base_path = './dataset/full_img/'
singer_list = os.listdir(base_path)

for idx, singer_folder in enumerate(singer_list) : 
    print(singer_folder)
    full_img_list = os.listdir(base_path+singer_folder)
    crop_idx = 0
    for full_img in full_img_list :
        save_dir = './dataset/crop_img/'+singer_folder
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        file_path = os.path.join(base_path, singer_folder, full_img)
        img_origin = cv.imread(file_path)
        try :
            img_gray = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY) 
        except :
            print('wrongfile!')
            print(file_path)
            continue

        results = cascade.detectMultiScale(img_gray,            # 입력 이미지
                                        scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                        minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                        minSize=(64,64)  # 탐지 객체 최소 크기
                                        ) 
        for rect in results :
            sy, sx, h, w = rect
            crop = img_gray[sx:sx+w, sy:sy+h]
            # cv.imshow('crop', crop)
            # cv.waitKey(0)
            crop = cv.resize(crop, (64,64))
            cv.imwrite(os.path.join(save_dir, f'{singer_folder}_'+str(crop_idx)+'.png'), crop)
            
            crop_idx += 1
