# 녹화된 영상을 불러와서 손을 감지하고 각 랜드마크의 좌표를 계산하여 데이터프레임으로 생성하는 프로세스입니다.

# Release 1.2 by Min-chul
# 카테고리 라벨 변경("click" -> "leftclick")
# 카테고리 라벨 추가("leftclick", "rightclick", "scroll", "pause", "screenshot")
# normalization_flag 변수 기본 값(default) 변경

# Release 1.1 by Min-chul
# 비 정상적인 랜드마크 값을 획득하였을 때, 의도치 않게 데이터프레임을 구성 시도를 하면서 발생하는 오류 해결
# 각 영상마다 데이터프레임(.csv)을 저장하도록 저장 구조를 변경 (나중에 통합할 때 유리함)
# cvzone 라이브러리를 개조한 "CustomHandTrackingModule"을 사용하는 방향으로 변경
# 변경된 라이브러리에서 정규화(Normalization) 작업을 설정할 수 있는 플래그 변수 추가
# 플래그 변수 추가에 따른 딕셔너리 및 리스트 생성 과정 변경
# 플래그 변수 추가에 따른 파일 저장 형식 변경
# 데이터프레임(df)을 저장할 때 파일명 형식 변경

# Release 1.0 by Min-chul
# 최초 버전 공유

from CustomHandTrackingModule import HandDetector
import cv2
import glob
import os
import pandas as pd
import time


if __name__ == '__main__':
    # "cvzone"라이브러리를 커스터마이즈한 "HandDetector"모듈을 사용하기 위한 객체를 생성합니다.
    detector = HandDetector(detectionCon=0.9, maxHands=1)

    # 카테고리에 추가할 라벨의 명칭을 정의합니다.
    label_list = ['move', 'leftclick', 'rightclick', 'scroll', 'pause', 'screenshot']

    # 랜드마크의 좌표를 정규화(Normalize) 과정을 진행할 지 결정합니다. True: 진행, False: 진행 안함
    normalization_flag = True

    # 현재 작업중인 디렉토리에 "rawdata" 디렉토리가 없는 경우, 디렉토리를 생성합니다.
    os.makedirs('rawdata', exist_ok=True)

    # 지정한 디렉토리에서 .mp4 확장자를 가진 파일만 가져옵니다.
    video_list = glob.glob('./videosource/*.mp4')
    for filename in video_list:
        s_time = time.time()
        cap = cv2.VideoCapture(filename=filename, apiPreference=None)
        print(f'"{filename}" is opened.')
        category = None
        # 영상 파일의 파일명에서 라벨 이름을 찾기위한 작업을 수행합니다.
        for label_name in label_list:
            if label_name in filename:
                category = label_name

        # 영상의 해상도(width, height) 정보를 획득합니다.
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width, height = int(width), int(height)

        # 각 랜드마크(최대 21개)의 좌표값 및 카테고리 정보를 저장할 딕셔너리 타입의 변수를 생성합니다.
        landmark_data = {}
        for idx in range(21):
            landmark_data[f'x{idx}'] = []
            landmark_data[f'y{idx}'] = []
        landmark_data['category'] = []

        while cap.isOpened():
            # 영상 객체에서 영상 데이터(동작 여부, 프레임)을 획득합니다.
            ret, frame = cap.read()
            # 카메라가 정상 동작하지 않는 경우, 즉시 종료합니다.
            if not ret:
                break

            # 영상으로부터 획득한 프레임을 "detector"객체에 정의되어 있는 "findHands"함수의 입력으로 주어 리턴값을 획득합니다.
            hands, img = detector.findHands(frame, flipType=False, normalization=normalization_flag)
            # 각 랜드마크(최대 21개)의 좌표 데이터 중에 None 값이 존재할 경우 해당 프레임은 데이터프레임에 추가하지 않도록 합니다.
            none_flag = False
            # 영상에서 손이 감지되었을때만 실행합니다.
            if hands:
                # 영상에서 감지된 손의 각 랜드마크 데이터를 획득합니다.
                lmList = hands[0]['lmList']
                # 랜드마크의 좌표값을 저장하는 리스트를 생성합니다.
                loc_x, loc_y = [], []
                normalize_x, normalize_y = [], []
                # 각 랜드마크(최대 21개)의 좌표 값을 검사합니다.
                for idx in range(len(lmList)):
                    # 랜드마크의 좌표가 해상도의 최대, 최소값을 벗어나는 경우, 데이터프레임에 추가하지 않도록 "none_flag"값을 변경합니다.
                    if lmList[idx][0] < 0 or lmList[idx][0] > width:
                        none_flag = True
                        print('A landmark out of resolution has been detected. '
                              'This data is not appended to the dataframe. (x coordinate)')
                    if lmList[idx][1] < 0 or lmList[idx][1] > height:
                        none_flag = True
                        print('A landmark out of resolution has been detected. '
                              'This data is not appended to the dataframe. (y coordinate)')
                    # 만약, 정규화된 데이터를 획득하였을 때 1보다 큰 값을 획득한 경우 데이터프레임에 추가하지 않도록 합니다.
                    if normalization_flag and (lmList[idx][0] > 1 or lmList[idx][1] > 1):
                        none_flag = True
                        print('A landmark out of resolution has been detected. '
                              'This data is not appended to the dataframe. (from normalized module.)')

                    # None 값이 검출되지 않았을 경우(none_flag = False)에만 정규화 작업 및 데이터 구성을 진행합니다.
                    if not none_flag:
                        # 랜드마크의 좌표값을 0에서 1사이의 값으로 정규화(Normalize)한 값을 리스트에 추가합니다.
                        loc_x.append(lmList[idx][0])
                        loc_y.append(lmList[idx][1])

                # None 값이 검출되지 않았을 경우에만 정규화 된 좌표 및 카테고리 정보를 딕셔너리에 추가합니다.
                if not none_flag:
                    # 정규화된 좌표를 데이터프레임에 추가합니다.
                    for idx in range(len(loc_x)):
                        landmark_data[f'x{idx}'].append(loc_x[idx])
                        landmark_data[f'y{idx}'].append(loc_y[idx])
                    # 카테고리 정보를 데이터프레임에 추가합니다.
                    landmark_data['category'].append(category)
            if not none_flag:
                # 데이터 프레임 생성
                df = pd.DataFrame()
                for idx in range(21):
                    df[f'x{idx}'] = landmark_data[f'x{idx}']
                    df[f'y{idx}'] = landmark_data[f'y{idx}']
                df['category'] = landmark_data['category']

            # 영상을 출력합니다.
            cv2.imshow('Video', frame)
            # 'Q' 버튼을 누르면 영상을 종료합니다.
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                break
        # 영상 분석에 걸린 시간을 출력합니다.
        print(f'runtime is {(time.time() - s_time):.3f} seconds.')
        # 데이터프레임(df)의 파일 이름을 결정하기 위해 불러온 영상 파일의 이름을 수정합니다.
        split_filename = filename.split('\\')[-1].replace('.mp4', '')
        save_filename = f'{split_filename}_normalize.csv' if normalization_flag else f'{split_filename}.csv'
        # 데이터프레임(df)을 "rawdata" 디렉토리에 저장합니다. 인덱스는 부여하지 않습니다.
        df.to_csv(f'./rawdata/{save_filename}', index=False)
        print(f'"{save_filename}" is saved.')
