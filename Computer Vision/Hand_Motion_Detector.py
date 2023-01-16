
import cv2
import mediapipe as mp
import time
import serial

class HandDetector():
    def __init__(self,mode = False,max_hands = 2,detection_confidence = 0.5,track_confidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.hands = self.mp_hands.Hands(self.mode,self.max_hands,min_detection_confidence = self.detection_confidence,min_tracking_confidence = self.track_confidence,)
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self,image,draw = True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image,hand_landmark,self.mp_hands.HAND_CONNECTIONS)
        return image
    def find_position(self,image,hand_id=0,draw = True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_id]
            for id,landmark in enumerate(my_hand.landmark):
                height,width,center = image.shape
                center_x,center_y = int(landmark.x*width),int(landmark.y*height)
                landmark_list.append([id,center_x,center_y])
                if draw:
                    cv2.circle(
                        image,(center_x,center_y),
                        6,(255,0,255),cv2.FILLED 
                    )
        return landmark_list
            
def main():
    capture = cv2.VideoCapture(0)
    previous_time = 0
    detector = HandDetector()
    while True:
        success,image = capture.read()
        image = detector.find_hands(image)
        landmark_list = detector.find_position(image)
        if len(landmark_list)!=0:
            print(landmark_list[4])
        current_time = time.time()
        fps = 1/(current_time-previous_time)
        cv2.putText(
        image,"FPS: {}".format(int(fps)),
        (10,70),cv2.FONT_HERSHEY_TRIPLEX,2,(0,255,0),3
        )
        previous_time = current_time
        cv2.imshow("Image",image)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()