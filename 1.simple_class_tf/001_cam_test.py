import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

print(type(cap))

if cap.isOpened() == False :
    print('카메라 연결 실패')
    exit(1)