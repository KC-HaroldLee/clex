import cv2 as cv
import os 

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

xml = 'haarcascades/haarcascade_frontalface_default.xml'
save_dir = 'save/'

if not os.path.isfile(xml) :
    print('xml 파일이 없어요')
    exit(1)
face_cascade = cv.CascadeClassifier(xml)

print(type(cap))

if cap.isOpened() == False :
    print('카메라 연결 실패')
    exit(1)

print('카메라 연결 완료')

# cv.namedWindow('cam1')

max_timer = 20
timer = 0
save_no = 814
while(True):
    ret, frame = cap.read() # 한장 씩 읽는다.
    frame = cv.flip(frame, 1) #  플립!
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.05, 1) #  튜플을 반환한다
    
    if len(faces) : 
        for face in faces :
            x,y,w,h, = face
            if timer == 0 :
                cropped_img = frame[y:y+h, x:x+w]
                cv.imwrite(save_dir+'cam_{}.png'.format(save_no), cropped_img)
                save_no += 1
            
            cv.rectangle(frame, (x,y), (x+w,y+h), (255,125,125), 3)

    # cv.imshow('cam1', frame)
    # cv.imshow('gray', gray)


    k = cv.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break
    
    if timer < max_timer :
        timer += 1
    else :
        timer = 0
    

cv.destroyAllWindows()