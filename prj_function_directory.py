import glob
import numpy as np
import pandas as pd


def reference_loc(x, y):
    x3 = (x[5] + x[17] + x[0]) / 3
    y3 = (y[5] + y[17] + y[0]) / 3
    return [x3, y3]

def sol_length(x1, y1, x2, y2):  # solution_length, 두 점사이의 길이
  return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def sol_ratio(x1, y1, x2, y2, x3, y3):  # solution_ratio
  len_a = sol_length(x1, y1, x3, y3)
  len_b = sol_length(x2, y2, x3, y3)
  return len_a / len_b

def __angle_between(p1, p2):  # 두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    res = np.rad2deg((ang1 - ang2) % (2 * np.pi))  # radius 값 -> degree 로 바꾼 값에 2파이로 나눈 나머지를 구함 // 360도 넘어가지 않도록 설정
    return res

def getAngle3P(p1, p2, p3, direction="CW"):  # 세점 사이의 각도 1->2->3, CW = 시계 방향
    pt1 = (p1[0] - p2[0], p1[1] - p2[1])
    pt2 = (p3[0] - p2[0], p3[1] - p2[1])  # p2 좌표를 0,0으로 재설정
    res = __angle_between(pt1, pt2)
    res = (res + 360) % 360
    if direction == "CCW":  # 반시계 방향
        res = (360 - res) % 360
    return res

def convert_ratio(x, y, normalization=2.5):  # 비율에 대한 DataFrame 생성 함수, 손바닥 끝~손가락 가장 끝마디/손바닥 끝~손가락 가장 안쪽마디 = 2.5 배
    fingers = [[4, 5, 17], [8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
    result = []

    for i in fingers:
        result.append(sol_ratio(x[i[0]], y[i[0]], x[i[1]], y[i[1]], x[i[2]], y[i[2]]) / normalization)

    return result


def convert_angle(x, y, normalization=360):  # 각도에 대한 DataFrame 생성 함수
    fingers = [[4, 3, 2, 1], [8, 7, 6, 5, 9], [12, 11, 10, 9, 13], [16, 15, 14, 13, 17], [20, 19, 18, 17, 0]]  # 엄지~새끼손가락 순
    result = []

    for i in range(5):  # 손가락 개수 (엄지~새끼손가락 순)
        count = len(fingers[i]) - 2 # 한 리스트에서 세 점을 순차적으로 추출해서 계산 ex) 엄지 = [4,3,2], [3,2,1]

        for j in range(count):
            result.append(
                getAngle3P([x[fingers[i][j]], y[fingers[i][j]]],
                           [x[fingers[i][j + 1]], y[fingers[i][j + 1]]],
                           [x[fingers[i][j + 2]], y[fingers[i][j + 2]]]) / normalization)

    return result


# 여러 개의 데이터프레임(.csv)을 한 개의 통합 데이터프레임으로 생성하는 함수입니다.
def df_concat(**kwargs):
    path = kwargs['path'] if 'path' in kwargs else './rawdata'
    name = kwargs['name'] if 'name' in kwargs else 'integration'

    # 지정한 디렉토리에서 .csv 확장자를 가진 파일만 가져옵니다.
    df_list = glob.glob(f'{path}/*.csv')

    # 통합본(integration.csv)을 생성하기 위한 데이터프레임을 생성합니다.
    df = pd.DataFrame()

    # "df_list"등록된 파일의 갯수만큼 반복합니다.
    for filename in df_list:
        # 통합본(integration.csv)이 이미 생성되어 있는 경우, 데이터프레임 통합에서 제외합니다.
        if 'integration' not in filename:
            # 데이터프레임을 불러옵니다.
            read_df = pd.read_csv(filename, index_col=False)
            # 기존에 구성된 데이터프레임(df)과 새로 불러온 데이터프레임(read_df)을 통합합니다.
            df = pd.concat([df, read_df], ignore_index=True)
    # 통합본(integration.csv)을 "rawdata"디렉토리에 저장합니다.
    name = ''.join([name, f'_col{len(df)}.csv'])
    df.to_csv(f'{path}/{name}', index=False)
