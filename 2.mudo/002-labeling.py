import cv2 as cv
import os
import uuid
import shutil

cascade_filename = '.\\97-haar\\haarcascade_frontalface_alt.xml'
cascade = cv.CascadeClassifier(cascade_filename)


base_path = '97-haar'
images_path = os.path.join(base_path, 'images-2017')
results_path = os.path.join(base_path, 'results-2017')
ends_path = os.path.join(images_path, 'ends')

label_list = [
            '0PMS', # q 박명수
            '1JJH', # w 정준하
            '2YJS', # e 유재석
            # '3GIL', # a 길성준
            # '4JHD', # s 정형돈
            # '5NHC', # z 노홍철
            '3HDH', # x 하동훈
            '4HKH', # c 황광희
            '5YSH', # v 양세형
            ]

# for label in label_list :
#     os.mkdir(os.path.join(results_path, label))

file_list = os.listdir(images_path)
print('len(file_list) : ', len(file_list))

for file_idx, file in enumerate(file_list) : 
    if file == 'ends' : continue
    file_path = os.path.join(images_path, file)
    img_origin = cv.imread(file_path)
    # print(type(img))
    gray = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY) 

    results = cascade.detectMultiScale(gray,            # 입력 이미지
                                    scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                    minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                    minSize=(120,120)  # 탐지 객체 최소 크기
                                    )  
    if len(results) == 0 :
        print('{} 삭제됨 - 인식얼굴없음'.format(file))
        os.remove(file_path)
        continue
    for box_idx, box in enumerate(results):
            
        x, y, w, h = box
        img_drew = img_origin.copy()
        cv.rectangle(img_drew, (x,y), (x+w, y+h), (255,255,255), thickness=0)
        hash = str(uuid.uuid1())[:8]
        # 사진 출력        
        cv.imshow('facenet',img_drew) 

        key = cv.waitKey(0)

        if key == ord('q') : label = label_list[0]
        if key == ord('w') : label = label_list[1]
        if key == ord('e') : label = label_list[2]
        # if key == ord('a') : label = label_list[3]
        # if key == ord('s') : label = label_list[4]
        # if key == ord('z') : label = label_list[5]        
        if key == ord('x') : label = label_list[3]
        if key == ord('c') : label = label_list[4]
        if key == ord('v') : label = label_list[5]
        if key == ord('1') : 
            if box_idx+1 == len(results) :
                os.remove(file_path)
                print('{} 삭제됨 - 멤버없음'.format(file))
                break
            else :
                continue

        imgcrop = img_origin [y:y+h, x:x+w] # (149,149, 3)
        imgcrop = cv.resize(imgcrop, dsize=(128,128)) # (64, 64, 3)

        results_label_path = os.path.join(results_path, label)
        cv.imwrite(os.path.join(results_label_path, hash+'.png'), imgcrop)
        # print('crop image 저장완료')
        # os.remove(file_path)

    ends_file_path = os.path.join(ends_path, file)
    try :
        shutil.move(file_path, ends_file_path)
    except :
        print('뭔가 에러 : {}'.format(file))
    print('{} 작업완료 | idx = {}/{}'.format(file, file_idx, len(file_list)))