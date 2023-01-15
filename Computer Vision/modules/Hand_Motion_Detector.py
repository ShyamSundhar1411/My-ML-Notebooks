
import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
capture = cv2.VideoCapture(0)
previous_time = 0
while True:
    success,image = capture.read()
    imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id,landmark in enumerate(hand_landmark.landmark):
                
                height,width,center = image.shape
                center_x,center_y = int(landmark.x*width),int(landmark.y*height)
                print(id,center_x,center_y)
                if id==0:
                    cv2.circle(
                        image,(center_x,center_y),
                        6,(255,0,255),cv2.FILLED
                    )
            mp_draw.draw_landmarks(image,hand_landmark,mp_hands.HAND_CONNECTIONS)
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    cv2.putText(
    image,"FPS: {}".format(int(fps)),
    (10,70),cv2.FONT_HERSHEY_TRIPLEX,2,(0,255,0),3
    )
    previous_time = current_time
    cv2.imshow("Image",image)
    cv2.waitKey(1)
    