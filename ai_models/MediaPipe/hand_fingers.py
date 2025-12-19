import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)

mpHand=mp.solutions.hands
hands=mpHand.Hands()

while True:
    success, img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result=hands.process(imgRGB)

    if result.multi_hand_landmarks:
        
        hand=result.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(img,hand,mpHand.HAND_CONNECTIONS)

        lmList=[]
        fingers=0
        for id, lm in enumerate(hand.landmark):
            h,w,c=img.shape
            cx , cy=int(lm.x*w) , int(lm.y*h)
            lmList.append([id,cx,cy])

        if len(lmList)!=0:
            
            #Pinky
            if lmList[20][2] < lmList[19][2] : 
                fingers+=1

            #Ring
            if lmList[16][2] < lmList[15][2] : 
                fingers+=1

            #Middle
            if lmList[12][2] < lmList[11][2] : 
                fingers+=1

            #Index
            if lmList[8][2] < lmList[7][2] : 
                fingers+=1

            #Thumb
            if lmList[4][1] < lmList[3][1]  : 
                fingers+=1

        cv2.putText(img,f'{fingers}', (100,300),cv2.FONT_HERSHEY_COMPLEX,5,(0,255,255),3)
        # print(fingers)




    cv2.imshow("image",img)
    cv2.waitKey(1)