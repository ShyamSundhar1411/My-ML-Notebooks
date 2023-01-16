import cv2
import numpy as np
import pyfirmata
import Hand_Motion_Detector as hmd
import math
import time
capture = cv2.VideoCapture(0)
capture.set(3,1280)
capture.set(4,720)
hand_detector = hmd.HandDetector(detection_confidence=0.7)
port = "COM8"
board = pyfirmata.Arduino(port)
servoPin = board.get_pin('d:9:s')
previous_time = 0
angle = 0
angle_bar = 0

while True:
    success,image = capture.read()
    image = hand_detector.find_hands(image)
    landmark_list = hand_detector.find_position(image,draw = False)
    if len(landmark_list)!=0:
        print(landmark_list)
        x1,y1 = landmark_list[4][1],landmark_list[4][2]
        x2,y2 = landmark_list[8][1],landmark_list[8][2]
        x3,y3 = landmark_list[9][1],landmark_list[9][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        cv2.circle(image,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(image,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.circle(image,(x3,y3),15,(255,0,255),cv2.FILLED)
        cv2.circle(image,(cx,cy),15,(255,0,255),cv2.FILLED)
        cv2.line(image,(x1,y1),(x2,y2),(255,255,255),2)
        length = math.hypot(x2-x1,y2-y1)
        if length<50:
            cv2.circle(image,(cx,cy),15,(0,255,0),cv2.FILLED)
        angle = np.interp(length,[50,300],[0,180])
        
        angle_bar = np.interp(length,[50,300],[400,150])
        print("Angle: ",int(angle))
        
        
        cv2.rectangle(image,(50,150),(85,400),(255,0,255),3)
        cv2.rectangle(image,(50,int(angle_bar)),(85,400),(255,0,255),cv2.FILLED)
        cv2.putText(image,
        "Angle: {} deg".format(int(angle)),(40,450),
        cv2.FONT_HERSHEY_DUPLEX,
        1,(0,255,0),3
        )
        servoPin.write(angle);
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv2.putText(image,
    "FPS: {}".format(int(fps)),(40,70),
    cv2.FONT_HERSHEY_DUPLEX,
    1,(0,255,0),3
    )
    cv2.imshow("Img",image)

    cv2.waitKey(1)