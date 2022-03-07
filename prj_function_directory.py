import glob
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import pyautogui
# from gui_app import GUI


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
    # 다른 함수에서 데이터 프레임 자체를 리턴받고 싶은 경우, "True"를 인자값으로 전달하면 됩니다.
    ref = kwargs['ref'] if 'ref' in kwargs else False

    # 지정한 디렉토리에서 .csv 확장자를 가진 파일만 가져옵니다.
    df_list = glob.glob(f'{path}/*.csv')

    # 통합본(integration.csv)을 생성하기 위한 데이터프레임을 생성합니다.
    df = pd.DataFrame()

    # "df_list"등록된 파일의 갯수만큼 반복합니다.
    for filename in df_list:
        # 통합본(integration.csv)이 이미 생성되어 있는 경우, 데이터프레임 통합에서 제외합니다.
        if name not in filename:
            # 데이터프레임을 불러옵니다.
            read_df = pd.read_csv(filename, index_col=False)
            # 기존에 구성된 데이터프레임(df)과 새로 불러온 데이터프레임(read_df)을 통합합니다.
            df = pd.concat([df, read_df], ignore_index=True)
    # 통합본(integration.csv)을 "rawdata"디렉토리에 저장합니다.
    name = ''.join([name, f'_col{len(df)}.csv'])
    df.to_csv(f'{path}/{name}', index=False)

    if ref:
        return df

# 데이터 분포를 균등하게 만들어 주는 함수 입니다.
def split_evenly_df(path='./rawdata', name='integration', ref=False):
    df = df_concat(path=path, name=name, ref=ref)
    cat_uniq = (df['category'].unique()).tolist()

    # print(cat_uniq)
    # print(type(cat_uniq))

    # 데이터 분량의 최소값 산출
    min_volume = 987654321
    for i in cat_uniq:
        if min_volume > len(df[df.category == i]):
            min_volume = len(df[df.category == i])

    # 최종 데이터 프레임 생성
    final_df = pd.DataFrame()
    for i in cat_uniq:
        temp_df = df[df.category == i]  # 임시 데이터 프레임에 카테고리 하나만 입력
        temp_df = (shuffle(temp_df)).head(min_volume)  # 임시 데이터프레임을 섞고, 최소치만큼 출력
        temp_df.reset_index(drop=True, inplace=True)  # 리셋 인덱스
        final_df = pd.concat([final_df, temp_df])  # 최종 데이터프레임에 통합

    final_df.reset_index(drop=True, inplace=True)

    return final_df

# 캠 좌표 및 캠 내 마우스 조작 영역 좌표를 윈도우 좌표에 맞게 재설정한다.
def convert_loc(win_h, win_w, x, y, cam_h=1, cam_w=1):
    result_x = win_h * x / cam_h
    result_y = win_w * y / cam_w

    return result_x, result_y


#이하 마우스 이벤트에 대한 function 함수

def move_event(x, y):
    pyautogui.moveTo(x, y)

def leftclick_event(x, y):
    pyautogui.leftClick(x, y)

def rightclick_event(x, y):
    pyautogui.rightClick(x, y)

def doubleclick_event(x, y):
    pyautogui.doubleClick(x, y)

drag_flag, no_dup_drag  = False, True

def drag_event(drag_flag):
    global no_dup_drag #no duplicate drag - 마우스 업/다운이 중복 방지를 위한 flag
    if drag_flag and no_dup_drag: #드래그 할 때 마우스 다운(버튼 누르기)
        pyautogui.mouseDown()
        no_dup_drag = False
    elif not drag_flag and not no_dup_drag: #드래그 할 때 마우스 업(버튼 떼기)
        pyautogui.mouseUp()
        no_dup_drag = True

def screenshot_event():
    img_print = pyautogui.press('printscreen') #printscreen key 누름
    img_print.show() #이미지 화면으로 띄움
    return True

def scroll_event(vector):
    if vector <= -10: # 이동된 y 좌표가 -10 이하일때
        pyautogui.scroll(-80)  # 스크롤 다운
    elif vector >= 10:  # 이동된 y좌표가 10 이상일때
        pyautogui.scroll(80)  # 스크롤 업

# 캠 좌표 및 캠 내 마우스 조작 영역 좌표를 윈도우 좌표에 맞게 재설정한다.
def convert_loc(xy, win_xy, cam_xy, margin):
    result_x = np.interp(xy[0], (margin, cam_xy[0] - margin), (0, win_xy[0]))
    result_y = np.interp(xy[1], (margin, cam_xy[1] - margin), (0, win_xy[1]))

    return (result_x, result_y)

# 아이콘 크기 512 * 512  0.15
# x나 y 둘중 하나라도 아이콘 영역을 벗어나면 False를 준다.
def region_in(xy, start_xy, end_xy):
    result = True
    if start_xy[0] > xy[0] or start_xy[0] + end_xy[0] < xy[0]: result = False
    if start_xy[1] > xy[1] or start_xy[1] + end_xy[1] < xy[1]: result = False

    return result