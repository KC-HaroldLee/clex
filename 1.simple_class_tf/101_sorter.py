import os


import os
import cv2 as cv

wait_dir = 'save/'
sort_dir = 'sort/'

class_list = ['mask', 'half-mask', 'unmask']

for class_name in class_list :
    print(class_name)
    if os.path.isdir(sort_dir+class_name) : 
        pass
    else :
        os.makedirs(sort_dir+class_name)

file_list = os.listdir(wait_dir)

save_shape = (256,256)

def save_image(file_name, src, no) :
    save_path = sort_dir+class_list[no]+'/'+file_name
    print(save_path)
    cv.imwrite(save_path, src)
    print('sort to {}'.format(class_list[no]))


for file in file_list :
    src = cv.imread(wait_dir+file)
    src = cv.resize(src, (256,256))

    key_input = True
    while(key_input):
        cv.imshow('scr', src)
        key = cv.waitKey()
        
        if key == ord('q') :
            save_image(file, src, 0)
            break

        elif key == ord('w') :
            save_image(file, src, 1)
            break

        elif key == ord('e') :
            save_image(file, src, 2)
            break

        elif key == ord('r') :
            print('pass!')
            break
        pass
    os.remove(wait_dir+file)