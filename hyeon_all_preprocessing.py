import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle, math

__loc = True
__ratio = False
__angle = False
ratio_norm = 2.5
angle_norm = 360

result_ang_df = pd.DataFrame()
result_ratio_df = pd.DataFrame()

df = pd.read_csv('./파일 위치')  # 추후 파일명에 맞게 수정

# print(df.head())
# df.info()

Y = df['category']  # 카테고리 y값으로 추출
X = df.loc[:, df.columns != 'category']  # 카테고리를 제외한 X값 추출

def sol_length(x1, y1, x2, y2):
  return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def sol_ratio(x1, y1, x2, y2, x3, y3):
  len_a = sol_length(x1, y1, x3, y3)
  len_b = sol_length(x2, y2, x3, y3)
  return len_a / len_b

def __angle_between(p1, p2):  #두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return res

def getAngle3P(p1, p2, p3, direction="CW"): #세점 사이의 각도 1->2->3
    pt1 = (p1[0] - p2[0], p1[1] - p2[1])
    pt2 = (p3[0] - p2[0], p3[1] - p2[1])
    res = __angle_between(pt1, pt2)
    res = (res + 360) % 360
    if direction == "CCW":  #반시계방향
        res = (360 - res) % 360
    return res

def start_create_ratio_df(normalization=2.5):
    global result_ratio_df

    ratio_0, ratio_1, ratio_2, ratio_3, ratio_4 = [], [], [], [], []

    for i in range(len(X)):
        ratio_0.append(sol_ratio(X['x4'][i], X['y4'][i], X['x5'][i], X['y5'][i], X['x17'][i], X['y17'][i]))
        ratio_1.append(sol_ratio(X['x8'][i], X['y8'][i], X['x5'][i], X['y5'][i], X['x0'][i], X['y0'][i]))
        ratio_2.append(sol_ratio(X['x12'][i], X['y12'][i], X['x9'][i], X['y9'][i], X['x0'][i], X['y0'][i]))
        ratio_3.append(sol_ratio(X['x16'][i], X['y16'][i], X['x13'][i], X['y13'][i], X['x0'][i], X['y0'][i]))
        ratio_4.append(sol_ratio(X['x20'][i], X['y20'][i], X['x17'][i], X['y17'][i], X['x0'][i], X['y0'][i]))

    # 진행상황 보여주는 코드 필요(?)

    result_ratio_df['ratio_0'] = ratio_0
    result_ratio_df['ratio_1'] = ratio_1
    result_ratio_df['ratio_2'] = ratio_2
    result_ratio_df['ratio_3'] = ratio_3
    result_ratio_df['ratio_4'] = ratio_4

    result_ratio_df = result_ratio_df / normalization

    # print(result_ratio_df.head())
    result_ratio_df.info()

def start_create_angle_df(normalization=360):
    global result_ang_df
    fingers = [[4, 3, 2, 1], [8, 7, 6, 5, 9], [12, 11, 10, 9, 13], [16, 15, 14, 13, 17], [20, 19, 18, 17, 0]]

    ang_0_0, ang_0_1, ang_1_0, ang_1_1, ang_1_2, ang_2_0, ang_2_1, ang_2_2, ang_3_0, ang_3_1, ang_3_2, ang_4_0, ang_4_1, ang_4_2 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    total = [ang_0_0, ang_0_1, ang_1_0, ang_1_1, ang_1_2, ang_2_0, ang_2_1, ang_2_2, ang_3_0, ang_3_1, ang_3_2, ang_4_0,
             ang_4_1, ang_4_2]

    for k in range(len(X)):
        for i in range(5):  # 손가락 개수
            count = len(fingers[i]) - 2
            if i == 0:  # 엄지
                total[0].append(getAngle3P([X[f'x{fingers[0][0]}'][k], X[f'y{fingers[0][0]}'][k]],
                                           [X[f'x{fingers[0][1]}'][k], X[f'y{fingers[0][1]}'][k]],
                                           [X[f'x{fingers[0][2]}'][k], X[f'y{fingers[0][0]}'][k]]))

                total[1].append(getAngle3P([X[f'x{fingers[0][1]}'][k], X[f'y{fingers[0][1]}'][k]],
                                           [X[f'x{fingers[0][2]}'][k], X[f'y{fingers[0][2]}'][k]],
                                           [X[f'x{fingers[0][3]}'][k], X[f'y{fingers[0][3]}'][k]]))
            else:  # 그 외
                for j in range(count):
                    total[2 + (i - 1) * count + j].append(
                        getAngle3P([X[f'x{fingers[i][j]}'][k], X[f'y{fingers[i][j]}'][k]],
                                   [X[f'x{fingers[i][j + 1]}'][k], X[f'y{fingers[i][j + 1]}'][k]],
                                   [X[f'x{fingers[i][j + 2]}'][k], X[f'y{fingers[i][j + 2]}'][k]]))

    total_name = ['ang_0_0', 'ang_0_1', 'ang_1_0', 'ang_1_1', 'ang_1_2', 'ang_2_0', 'ang_2_1', 'ang_2_2', 'ang_3_0',
                  'ang_3_1', 'ang_3_2', 'ang_4_0', 'ang_4_1', 'ang_4_2']

    for i in range(len(total)):
        result_ang_df[total_name[i]] = total[i]

    result_ang_df = result_ang_df / normalization
    result_ang_df.info()

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

final_use_df = final_use_df.to_numpy()

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)  # 라벨을 숫자에 대응
label = encoder.classes_  # 라벨 - 숫자 목록

with open('./resources/test_encoder_ratio_normalization_lbl.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_y)  # onehotencoding
print(final_use_df.shape)
print(onehot_Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(final_use_df, onehot_Y, test_size=0.1)  # 테스트 데이터 분리
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# exit()
xy = X_train, X_test, Y_train, Y_test

if [__loc, __ratio, __angle] == [True, True, True]:
    np.save('./resources/encoder_complex_data', xy)
else:
    np.save('./resources/encoder{}_data'.format(name), xy)