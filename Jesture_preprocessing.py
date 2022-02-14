from CustomHandTrackingModule import HandDetector
import pandas as pd
import numpy as np
import cv2, time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height
detector = HandDetector(detectionCon=0.8, maxHands=2)
hand_data = pd.DataFrame()

while cap.isOpened():
    sucess, img = cap.read()
    if not sucess:
        print('video is over')
        break
    img = cv2.flip(img, 1)  # 좌우반전
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        if hands[0]['type'] == 'Right':
            R_lmList = hands[0]['lmList']
            R_fingers = detector.fingersUp(hands[0])
        elif hands[0]['type'] == 'Left':
            L_lmList = hands[0]['lmList']
            L_fingers = detector.fingersUp(hands[0])

        if len(hands) == 2:
            if hands[1]['type'] == 'Right':
                R_lmList = hands[1]['lmList']
                R_fingers = detector.fingersUp(hands[1])
            elif hands[1]['type'] == 'Left':
                L_lmList = hands[1]['lmList']
                L_fingers = detector.fingersUp(hands[1])


    cv2.imshow('Image', img)
    cv2.waitKey(1)
