from cvzone.HandTrackingModule import HandDetector
import pandas as pd
import cv2, time, glob

res = glob.glob('./resources/*.mp4')
cat = ['rock', 'scissor']

detector = HandDetector(detectionCon=0.8, maxHands=1)
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = [], [], [], [], [], [], [], [], [], []
x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20 = [], [], [], [], [], [], [], [], [], [], []
y0, y1, y2, y3, y4, y5, y6, y7, y8, y9 = [], [], [], [], [], [], [], [], [], []
y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20 = [], [], [], [], [], [], [], [], [], [], []

total = [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9,
         x10, y10, x11, y11, x12, y12, x13, y13, x14, y14, x15, y15, x16, y16, x17, y17, x18, y18, x19, y19, x20, y20]

x, y = [], [],
# z = []
category = []

for i in range(len(cat)):  # 영상 파일 불러오기
    cap = cv2.VideoCapture(res[i])
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 영상의 프레임 폭을 획득한다.
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 영상의 프레임 높이를 획득한다.

    while cap.isOpened():
        sucess, img = cap.read()
        if not sucess:  # 비디오가 끝났을때 while문을 빠져나가도록 설계
            print('video is over')
            break

        img = cv2.flip(img, 1)  # 좌우반전
        hands, img = detector.findHands(img, flipType=False)  # handtrckingmodule 사용

        if hands:  # 손이 감지될 때만 코드 활성화
            lmList = hands[0]['lmList']  # 손의 각 점좌표 할당
            if len(lmList) == 21:  # 손좌표가 21개일때만 좌표를 받아들일 수 있도록 설정
                for j in range(0, len(total), 2):
                    nom_x = lmList[j//2][0] / width  # x좌표 표준화
                    nom_y = lmList[j//2][1] / height # y좌표 표준화
                    # total[j].append(lmList[j//2][0])  # x좌표 입력
                    # total[j+1].append(lmList[j//2][1])  # y좌표 입력
                    total[j].append(nom_x)  # 표준화 x좌표 입력
                    total[j + 1].append(nom_y)  # 표준화 y좌표 입력
                category.append(cat[i])  # 카테고리명 지정

        cv2.imshow('input video', img)  # 이미지 출력
        cv2.waitKey(1)

print(len(x0), len(y20), len(category))
# 데이터를 판다스 데이터프레임으로 통합한다.
hand_data = pd.DataFrame({'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'x3':x3, 'y3':y3, 'x4':x4, 'y4':y4, 'x5':x5, 'y5':y5,
                          'x6':x6, 'y6':y6, 'x7':x7, 'y7':y7, 'x8':x8, 'y8':y8, 'x9':x9, 'y9':y9,
                          'x10':x10, 'y10':x10, 'x11':x11, 'y11':y11, 'x12':x12, 'y12':y12, 'x13':x13, 'y13':y13, 'x14':x14, 'y14':y14,
                          'x15':x15, 'y15':y15, 'x16':x16, 'y16':y16, 'x17':x17, 'y17':y17, 'x18':x18, 'y18':y18, 'x19':x19,
                          'y19':y19, 'x20':x20, 'y20':y20, 'category':category})
print(hand_data.head())
hand_data.info()
# 데이터 프레임을 저장한다.
hand_data.to_csv('./resources/rock_scissor_nomallize_data.csv', index=False)
