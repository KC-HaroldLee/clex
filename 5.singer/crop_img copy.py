import cv2 as cv
import os
import shutil

cascade_filename = './haarcascade_frontalface_alt.xml'
cascade = cv.CascadeClassifier(cascade_filename)

base_path = './dataset/crop_img/'
save_path = './dataset/f_img/'
singer_list = os.listdir(base_path)

for idx, singer_folder in enumerate(singer_list) : 
    print(singer_folder)
    # os.mkdir(save_path+singer_folder)
    full_img_list = os.listdir(base_path+singer_folder)
    crop_idx = 0
    save_dir = './dataset/crop_img/'+singer_folder
    for full_img in full_img_list :
        src = cv.imread(base_path + singer_folder + '/' + full_img)
        src = cv.resize(src, (256,256))
        cv.imwrite(os.path.join(save_path+singer_folder, f'{singer_folder}_'+str(crop_idx)+'.png'), src)
        crop_idx += 1