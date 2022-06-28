import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
print(cv.__version__)
print(type(cap))

if cap.isOpened() == False :
    print('카메라 연결 실패')
    exit(1)

print('카메라 연결 완료')

cv.namedWindow('cam1')
while(True):
    ret, frame = cap.read() # 한장 씩 읽는다.
    cv.imshow('cam1', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cv.destroyAllWindows()