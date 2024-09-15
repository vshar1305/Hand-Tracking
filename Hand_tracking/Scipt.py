import cv2
import mediapipe as mp
import time

# explain this code line by line i am using mediapipe for the very first time i am very confused by this code as i am a new beginner to python even so can you please help me to understand this code an i am building the hand tracking project using palm detection and hall landmarks
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
while True:
    r,frame = cap.read()
    if r == True:
        re_frame = cv2.resize(frame,(400,300))
        imgRGB = cv2.cvtColor(re_frame,cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h,w,c  =re_frame.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    print(id,cx,cy)
                    if id == 4:
                        cv2.circle(re_frame,(cx,cy),15,(255,0,255),cv2.FILLED)
                mpDraw.draw_landmarks(re_frame,handLms,mpHands.HAND_CONNECTIONS)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(re_frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("org_vid",re_frame)
        if cv2.waitKey(25) & 0xff == ord("p"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()