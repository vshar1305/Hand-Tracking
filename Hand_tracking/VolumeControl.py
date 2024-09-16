import cv2
import time 
import numpy as np
import ScriptModule as scm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
pTime = 0

detector = scm.handDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
minVol = volumeRange[0]
maxVol = volumeRange[1]

while True:
    r,frame = cap.read()
    if r == True:
        re_frame = cv2.resize(frame,(400,300))

        re_frame = detector.findHands(re_frame)
        lmList = detector.findPosition(re_frame,draw=False)
        # if len(lmList) != 0:
            # print(lmList[4],lmList[8])

        if len(lmList) >= 9:
            # Get the coordinates of landmarks 4 and 8 (thumb and index finger tips)
            x1, y1 = lmList[4][1], lmList[4][2] 
            x2, y2 = lmList[8][1], lmList[8][2]
            cx,cy = (x1+x2)//2 , (y1+y2)//2

            cv2.circle(re_frame,(x1,y1),5,(255,0,255),cv2.FILLED)
            cv2.circle(re_frame,(x2,y2),5,(255,0,255),cv2.FILLED)
            cv2.line(re_frame,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.circle(re_frame,(cx,cy),5,(255,0,255),cv2.FILLED)

            length = math.hypot(x2-x1,y2-y1)
            # print(length)

            # Hand Range 10-240
            # Volume Range -65 -0

            vol = np.interp(length,[10,240],[minVol,maxVol])
            print(int(length),vol)
            volume.SetMasterVolumeLevel(vol, None)

            if length < 20:
                cv2.circle(re_frame,(cx,cy),5,(0,255,0),cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(re_frame,f'FPS:{int(fps)}',(30,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
        cv2.imshow("org_vid",re_frame)
        if cv2.waitKey(25) & 0xff == ord("p"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
