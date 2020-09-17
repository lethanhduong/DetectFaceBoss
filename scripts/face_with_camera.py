import os
import cv2
import dlib
import yaml
import time
from random import randint


def get_face_camera(folder_out):
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    cap = cv2.VideoCapture(0)
    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()
    
    i = 0
    extention = '.png'
    while True:
        ret, frame = cap.read()
        if ret:
            # Resize window
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 1200, 800)
            cv2.moveWindow('frame', 100, 100)
            
            # Resize frame for decrease predict time
            scale = 1 # 0.5
            
            # start = time.time()
            # resize_frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
            # print(time.time() - start)

            faces_hog = hog_face_detector(frame)
            frame_h, frame_w, _ = frame.shape
            
            reindex_x = lambda x: max(min(x, frame_w), 1)
            reindex_y = lambda x: max(min(x, frame_h), 1)
            
            # loop over detected faces
            for face in faces_hog:
                x = reindex_x(int(face.left() / scale))
                y = reindex_y(int(face.top() / scale))
                r = reindex_x(int(face.right() / scale))
                b = reindex_y(int(face.bottom() / scale))
                
                # draw box over face
                padding = 0
                sub_img = frame[y - padding:b + padding, x - padding:r + padding]
                
                if i % 20 == 0:
                    image_name_out = os.path.join(folder_out, str(randint(0, 10000000000)) + extention)
                    cv2.imwrite(image_name_out, sub_img)
                    
                cv2.rectangle(frame, (x, y), (r, b), (0, 255, 0), 2)
                cv2.imshow("frame", frame)
                
                key = cv2.waitKey(1)
                if key == 27:
                    break
                i += 1

    cap.release()
    cv2.destroyAllWindows()

folder_out = os.path.join('/Users/138210/Desktop/Research/FaceBoss/data/tests/get_face_from_camera')
get_face_camera(folder_out)
