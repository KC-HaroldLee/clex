import numpy as np
import cv2


xml = 'haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml)


cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용
cap.set(3,640) # 너비
cap.set(4,480) # 높이

max_timer = 20
timer = 0
save_no = 0
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.05, 5)
    # print("Number of faces detected: " + str(len(faces)))
    print(faces)
    # if len(faces):
    #     for (x,y,w,h) in faces:
    #         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    if timer == max_timer :
        x,y,w,h = faces[0]
        cropped_img = frame[y: y + h, x: x + w]
        cv2.imwrite('save/no/cam_{}.png'.format(save_no), cropped_img)
        print('save_no is ', save_no)
        save_no += 1
        timer = 0

    cv2.imshow('result', frame)
    timer +=1
    # print(timer)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()