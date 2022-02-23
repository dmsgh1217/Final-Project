# 녹화된 영상을 불러와서 손을 감지하고 각 랜드마크의 좌표를 계산하여 데이터프레임으로 생성하는 프로세스입니다.
# Release 1.0 by Min-chul

from cvzone.HandTrackingModule import HandDetector
import cv2
import glob
import os
import pandas as pd


if __name__ == '__main__':
    # "cvzone"라이브러리의 "HandDetector"모듈을 사용하기 위한 객체를 생성합니다.
    detector = HandDetector(detectionCon=0.9, maxHands=1)

    # 카테고리에 추가할 라벨의 명칭을 정의합니다.
    label_list = ['move', 'click']

    # 각 랜드마크(최대 21개)의 좌표값 및 카테고리 정보를 저장할 딕셔너리 타입의 변수를 생성합니다.
    landmark_data = {}
    for idx in range(21):
        landmark_data[f'x{idx}'] = []
        landmark_data[f'y{idx}'] = []
    landmark_data['category'] = []

    # 지정한 디렉토리에서 .mp4 확장자를 가진 파일만 가져옵니다.
    directory = './videosource'
    video_list = glob.glob(f'{directory}/*.mp4')

    for filename in video_list:
        cap = cv2.VideoCapture(filename=filename, apiPreference=None)
        category = None
        # 영상 파일의 파일명에서 라벨 이름을 찾기위한 작업을 수행합니다.
        for label_name in label_list:
            if label_name in filename:
                category = label_name

        # 영상의 해상도(width, height) 정보를 획득합니다.
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width, height = int(width), int(height)

        while cap.isOpened():
            # 영상 객체에서 영상 데이터(동작 여부, 프레임)을 획득합니다.
            ret, frame = cap.read()
            # 카메라가 정상 동작하지 않는 경우, 즉시 종료합니다.
            if not ret:
                break

            # 영상으로부터 획득한 프레임을 "detector"객체에 정의되어 있는 "findHands"함수의 입력으로 주어 리턴값을 획득합니다.
            hands, img = detector.findHands(frame, flipType=False)

            # 영상에서 손이 감지되었을때만 실행합니다.
            if hands:
                # 영상에서 감지된 손의 각 랜드마크 데이터를 획득합니다.
                lmList = hands[0]['lmList']
                # 각 랜드마크(최대 21개)의 좌표 데이터 중에 None 값이 존재할 경우 해당 프레임은 데이터프레임에 추가하지 않도록 합니다.
                none_flag = False
                # 각 랜드마크(최대 21개)의 좌표 값을 검사합니다.
                for idx in range(len(lmList)):
                    # 랜드마크의 좌표가 해상도의 최대, 최소값을 벗어나는 경우, 데이터프레임에 추가하지 않도록 "none_flag"값을 변경합니다.
                    if lmList[idx][0] < 0 or lmList[idx][0] > width:
                        none_flag = True
                    if lmList[idx][1] < 0 or lmList[idx][1] > height:
                        none_flag = True

                    # None 값이 검출되지 않았을 경우(none_flag = False)에만 정규화 작업 및 데이터 구성을 진행합니다.
                    if not none_flag:
                        # 랜드마크의 좌표값을 0에서 1사이의 값으로 정규화(Normalize) 합니다.
                        normalize_x = lmList[idx][0] / width
                        normalize_y = lmList[idx][1] / height
                        # 정규화된 좌표를 딕셔너리에 추가합니다.
                        landmark_data[f'x{idx}'].append(normalize_x)
                        landmark_data[f'y{idx}'].append(normalize_y)
                    # None 값이 검출되었을 경우, 아무런 작업도 진행하지 않습니다.
                    else:
                        print('A landmark out of resolution has been detected.'
                              'This data is not appended to the dataframe.')

                # None 값이 검출되지 않았을 경우에만 카테고리 정보를 딕셔너리에 추가합니다.
                if not none_flag:
                    landmark_data['category'].append(category)
                # None 값이 검출되었을 경우, 아무런 작업도 진행하지 않습니다.
                else:
                    print('A landmark out of resolution has been detected. This data does not add labels.')
            # 영상을 출력합니다.
            cv2.imshow('Video', frame)
            # 'Q' 버튼을 누르면 영상을 종료합니다.
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                break
    # 데이터 프레임 생성
    df = pd.DataFrame()
    for idx in range(21):
        df[f'x{idx}'] = landmark_data[f'x{idx}']
        df[f'y{idx}'] = landmark_data[f'y{idx}']
    df['category'] = landmark_data['category']
    # 생성된 데이터프레임의 정보를 확인합니다.
    print(df.info())

    # 현재 작업중인 디렉토리에 "rawdata" 디렉토리가 없는 경우, 디렉토리를 생성합니다.
    os.makedirs('rawdata', exist_ok=True)

    # 데이터프레임(df)을 "rawdata" 디렉토리에 저장합니다. 인덱스는 부여하지 않습니다.
    df.to_csv(f'./rawdata/landmark_position_normalize_w{int(width)}_h{int(height)}.csv', index=False)
