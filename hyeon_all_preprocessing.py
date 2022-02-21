import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle, math

# 사용자 지정 리소스
__loc = True
__ratio = False
__angle = False
ratio_norm = 2.5
angle_norm = 360
encoder_folder_location = 'resources/'
npy_folder_location = 'resources/'
df = pd.read_csv('./파일 위치')

# 본 코드
result_ang_df = pd.DataFrame()
result_ratio_df = pd.DataFrame()

# print(df.head())
# df.info()

Y = df['category']  # 카테고리 y값으로 추출
X = df.loc[:, df.columns != 'category']  # 카테고리를 제외한 X값 추출

# 함수
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

def start_create_ratio_df(normalization=2.5):  # 비율에 대한 DataFrame 생성 함수, 손바닥 끝~손가락 가장 끝마디/손바닥 끝~손가락 가장 안쪽마디 = 2.5 배
    global result_ratio_df

    print('start_ratio_calc')
    ratio_0, ratio_1, ratio_2, ratio_3, ratio_4 = [], [], [], [], []

    total = [ratio_0, ratio_1, ratio_2, ratio_3, ratio_4]
    total_name = ['ratio_0', 'ratio_1', 'ratio_2', 'ratio_3', 'ratio_4']

    for i in range(len(X)):  # 엄지 / 검지~새끼손가락 비율 산출을 위한 포인트는 각각 다름
        ratio_0.append(sol_ratio(X['x4'][i], X['y4'][i], X['x5'][i], X['y5'][i], X['x17'][i], X['y17'][i])) #엄지
        ratio_1.append(sol_ratio(X['x8'][i], X['y8'][i], X['x5'][i], X['y5'][i], X['x0'][i], X['y0'][i])) #검지
        ratio_2.append(sol_ratio(X['x12'][i], X['y12'][i], X['x9'][i], X['y9'][i], X['x0'][i], X['y0'][i])) #중지
        ratio_3.append(sol_ratio(X['x16'][i], X['y16'][i], X['x13'][i], X['y13'][i], X['x0'][i], X['y0'][i])) #약지
        ratio_4.append(sol_ratio(X['x20'][i], X['y20'][i], X['x17'][i], X['y17'][i], X['x0'][i], X['y0'][i])) #새끼손가락

        # 중간 진행상황 체크를 위한 코드 "...."
        if i % 100: print('.', end='')
        if i % 1000: print('')

    # DataFrame 생성
    for i in range(len(total)):
        result_ratio_df[total_name[i]] = total[i]

    # 만든 DataFrame에 정규화를 위해 설정값으로 나눠줌
    result_ratio_df = result_ratio_df / normalization

    # print(result_ratio_df.head())
    result_ratio_df.info()

def start_create_angle_df(normalization=360):  # 각도에 대한 DataFrame 생성 함수
    global result_ang_df

    print('start_angle_calc')
    fingers = [[4, 3, 2, 1], [8, 7, 6, 5, 9], [12, 11, 10, 9, 13], [16, 15, 14, 13, 17], [20, 19, 18, 17, 0]]  # 엄지~새끼손가락 순

    # 엄지 = 각도 2개, 엄지 외 각도 3개씩
    ang_0_0, ang_0_1, ang_1_0, ang_1_1, ang_1_2, ang_2_0, ang_2_1, ang_2_2, ang_3_0, ang_3_1, ang_3_2, ang_4_0, ang_4_1, ang_4_2 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    total = [ang_0_0, ang_0_1, ang_1_0, ang_1_1, ang_1_2, ang_2_0, ang_2_1, ang_2_2, ang_3_0, ang_3_1, ang_3_2, ang_4_0,
             ang_4_1, ang_4_2]

    for k in range(len(X)):
        for i in range(5):  # 손가락 개수 (엄지~새끼손가락 순)
            count = len(fingers[i]) - 2 # 한 리스트에서 세 점을 순차적으로 추출해서 계산 ex) 엄지 = [4,3,2], [3,2,1]
            if i == 0:  # 엄지
                # ang_0_0
                total[0].append(getAngle3P([X[f'x{fingers[0][0]}'][k], X[f'y{fingers[0][0]}'][k]],
                                           [X[f'x{fingers[0][1]}'][k], X[f'y{fingers[0][1]}'][k]],
                                           [X[f'x{fingers[0][2]}'][k], X[f'y{fingers[0][2]}'][k]]))
                # ang_0_1
                total[1].append(getAngle3P([X[f'x{fingers[0][1]}'][k], X[f'y{fingers[0][1]}'][k]],
                                           [X[f'x{fingers[0][2]}'][k], X[f'y{fingers[0][2]}'][k]],
                                           [X[f'x{fingers[0][3]}'][k], X[f'y{fingers[0][3]}'][k]]))
            else:  # 그 외
                for j in range(count):
                    # total list의 0, 1번의 엄지 index 제외한 손가락 각도 계산 필요하므로 1 + 필요
                    total[1 + i * count + j].append(
                        getAngle3P([X[f'x{fingers[i][j]}'][k], X[f'y{fingers[i][j]}'][k]],
                                   [X[f'x{fingers[i][j + 1]}'][k], X[f'y{fingers[i][j + 1]}'][k]],
                                   [X[f'x{fingers[i][j + 2]}'][k], X[f'y{fingers[i][j + 2]}'][k]]))

        # 중간 진행 상황 체크 "..." 프린트
        if i % 100: print('.', end='')
        if i % 1000: print('')

    total_name = ['ang_0_0', 'ang_0_1', 'ang_1_0', 'ang_1_1', 'ang_1_2', 'ang_2_0', 'ang_2_1', 'ang_2_2', 'ang_3_0',
                  'ang_3_1', 'ang_3_2', 'ang_4_0', 'ang_4_1', 'ang_4_2']

    # DataFrame 생성
    for i in range(len(total)):
        result_ang_df[total_name[i]] = total[i]

    # 만든 DataFrame 정규화
    result_ang_df = result_ang_df / normalization
    result_ang_df.info()

# 최종 사용할 DataFrame
final_use_df = pd.DataFrame()
name = ''

if __loc == True:
    name = name + '_loc'
    final_use_df = pd.concat([final_use_df, X], axis=1)  # 좌표데이터 통합

if __ratio == True:
    name = name + '_ratio'
    start_create_ratio_df(normalization=ratio_norm)
    final_use_df = pd.concat([final_use_df, result_ratio_df], axis=1)  # 비율데이터 통합

if __angle == True:
    name = name + '_angle'
    start_create_angle_df(normalization=angle_norm)
    final_use_df = pd.concat([final_use_df, result_ang_df], axis=1)  # 각도데이터 통합

final_use_df.info()

# DataFrame to Numpy 변환
final_use_df = final_use_df.to_numpy()

# Label Enconding
encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)  # 라벨을 숫자에 대응
label = encoder.classes_  # 라벨 - 숫자 목록

# One-hot Encoding
onehot_Y = to_categorical(labeled_y)
print(final_use_df.shape)
print(onehot_Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(final_use_df, onehot_Y, test_size=0.1)  # 테스트 데이터 분리, test_size = 0.1으로 고정
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test

#데이터 저장

if [__loc, __ratio, __angle] == [True, True, True]:
    np.save('./{}encoder_complex_data'.format(npy_folder_location), xy)

    with open('./{}encoder_complex_data_lbl.pickle'.format(encoder_folder_location), 'wb') as f:
        pickle.dump(encoder, f)
else:
    np.save('./{}encoder{}_data'.format(npy_folder_location, name), xy)

    with open('./{}encoder{}_data_lbl.pickle'.format(encoder_folder_location, name), 'wb') as f:
        pickle.dump(encoder, f)