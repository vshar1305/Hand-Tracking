import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialize hand detection with parameters
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Setup mediapipe hands and drawing utilities
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands, 
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, re_frame, draw=True):
        # Convert image to RGB (required by mediapipe)
        imgRGB = cv2.cvtColor(re_frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        self.result = self.hands.process(imgRGB)  # Store the result as a class attribute
        
        # If hand landmarks are detected, draw them on the frame
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(re_frame, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return re_frame
    
    def findPosition(self, re_frame, handNo=0, draw=True):
        lmList = []

        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = re_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(re_frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    cTime = 0  # Current time for FPS calculation
    pTime = 0  # Previous time for FPS calculation
    
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    # Initialize hand detector
    detector = handDetector()

    while True:
        r, frame = cap.read()  # Capture frame from the camera
        if r:
            re_frame = cv2.resize(frame, (400, 300))  # Resize the frame
            
            # Detect and draw hand landmarks
            re_frame = detector.findHands(re_frame)
            lmList = detector.findPosition(re_frame)
            if len(lmList) != 0:
                print(lmList[4])

            # Calculate Frames Per Second (FPS)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            # Display FPS on the frame
            cv2.putText(re_frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            
            # Display the frame
            cv2.imshow("org_vid", re_frame)
        
        # Exit when 'p' is pressed
        if cv2.waitKey(25) & 0xFF == ord("p"):
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Corrected function entry point
if __name__ == "__main__":
    main()
