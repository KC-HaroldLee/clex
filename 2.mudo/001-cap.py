import cv2 as cv
import os
import uuid

base_path = 'E:\\1. MH'
# base_path = 'D:\\Downloads\\2016-2018'

videos_path = os.path.join(base_path, '2011')
file_list = os.listdir(videos_path)

captures_path = 'D:\\workspace\\python_study\\playGround\\haar\\chk_2011'


print('len(file_list) : ', len(file_list))

for file in file_list : 
    file_path = os.path.join(videos_path, file)
    cap = cv.VideoCapture(file_path)

    frame_location = 54716
    while(True) :
        try : 
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_location)
            ret, frame_img = cap.read()
            hash = str(uuid.uuid1())[:8]
            cv.imwrite(os.path.join(captures_path, hash+'.png'), frame_img)

            frame_location += 75521
        except :
            break
    print('처리끝' , file)
