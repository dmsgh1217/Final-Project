import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from prj_function_directory import sol_ratio, getAngle3P
import pickle, math

# 사용자 지정 리소스
__loc = False
__ratio = False
__angle = True
ratio_norm = 2.5
angle_norm = 360
encoder_folder_location = 'resources/'
npy_folder_location = 'resources/'
df = pd.read_csv('./resources/landmark_position_normalize_w1280_h720.csv')

# 본 코드
result_ang_df = pd.DataFrame()  # 각도 데이터 프레임
result_ratio_df = pd.DataFrame()  # 비율 데이터 프레임

# print(df.head())
# df.info()

# 결측치 처리
df.dropna(inplace=True)  # Nan 값 제거
df.reset_index(drop=True, inplace=True)  # 인덱스 초기화

# 카데고리 분리
Y = df['category']  # 카테고리 y값으로 추출
X = df.loc[:, df.columns != 'category']  # 카테고리를 제외한 X값 추출

# 비율에 대한 DataFrame 생성 함수, 손바닥 끝~손가락 가장 끝마디/손바닥 끝~손가락 가장 안쪽마디 = 2.5 배
def start_create_ratio_df(normalization=2.5):
    global result_ratio_df

    print('start_ratio_calc')
    ratio_0, ratio_1, ratio_2, ratio_3, ratio_4 = [], [], [], [], []
    total = [ratio_0, ratio_1, ratio_2, ratio_3, ratio_4]
    total_name = ['ratio_0', 'ratio_1', 'ratio_2', 'ratio_3', 'ratio_4']
    fingers = [[4, 5, 17], [8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]  # 엄지~새끼손가락 순

    for i in range(len(X)):  # 엄지 / 검지~새끼손가락 비율 산출을 위한 포인트는 각각 다름
        for j in range(len(total)):
            total[j].append(sol_ratio(X[f'x{fingers[j][0]}'][i], X[f'y{fingers[j][0]}'][i],
                                      X[f'x{fingers[j][1]}'][i], X[f'y{fingers[j][1]}'][i],
                                      X[f'x{fingers[j][2]}'][i], X[f'y{fingers[j][2]}'][i]))

        # 중간 진행상황 체크를 위한 코드 "...."
        if i % 100 == 0: print('.', end='')
        if i % 2000 == 0: print('')

    # DataFrame 생성
    for i in range(len(total)):
        result_ratio_df[total_name[i]] = total[i]

    # 만든 DataFrame에 정규화를 위해 설정값으로 나눠줌
    result_ratio_df = result_ratio_df / normalization

    # print(result_ratio_df.head())
    result_ratio_df.info()

# 각도에 대한 DataFrame 생성 함수
def start_create_angle_df(normalization=360):
    global result_ang_df

    print('start_angle_calc')
    fingers = [[4, 3, 2, 1], [8, 7, 6, 5, 9], [12, 11, 10, 9, 13], [16, 15, 14, 13, 17], [20, 19, 18, 17, 0]]  # 엄지~새끼손가락 순

    # 엄지 = 각도 2개, 엄지 외 각도 3개씩
    ang_0_0, ang_0_1, ang_1_0, ang_1_1, ang_1_2, ang_2_0, ang_2_1, ang_2_2, ang_3_0, ang_3_1, ang_3_2, ang_4_0, ang_4_1, ang_4_2 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    total = [ang_0_0, ang_0_1, ang_1_0, ang_1_1, ang_1_2, ang_2_0, ang_2_1, ang_2_2, ang_3_0, ang_3_1, ang_3_2, ang_4_0,
             ang_4_1, ang_4_2]
    total_name = ['ang_0_0', 'ang_0_1', 'ang_1_0', 'ang_1_1', 'ang_1_2', 'ang_2_0', 'ang_2_1', 'ang_2_2', 'ang_3_0',
                  'ang_3_1', 'ang_3_2', 'ang_4_0', 'ang_4_1', 'ang_4_2']

    for k in range(len(X)):
        for i in range(5):  # 손가락 개수 (엄지~새끼손가락 순)
            count = len(fingers[i]) - 2  # 한 리스트에서 세 점을 순차적으로 추출해서 계산 ex) 엄지 = [4,3,2], [3,2,1]
            _step = 2 + (i - 1) * count
            if i == 0: _step = 0  # 엄지손가락이라면 _step을 0으로 한다.
            for j in range(count):
                # total list의 0, 1번의 엄지 index 제외한 손가락 각도 계산 필요하므로 1 + 필요
                total[_step + j].append(
                    getAngle3P([X[f'x{fingers[i][j]}'][k], X[f'y{fingers[i][j]}'][k]],
                               [X[f'x{fingers[i][j + 1]}'][k], X[f'y{fingers[i][j + 1]}'][k]],
                               [X[f'x{fingers[i][j + 2]}'][k], X[f'y{fingers[i][j + 2]}'][k]]))

        # 중간 진행 상황 체크 "..." 프린트
        if k % 100 == 0: print('.', end='')
        if k % 2000 == 0: print('')

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

# 데이터 저장
if [__loc, __ratio, __angle] == [True, True, True]:
    np.save('./{}encoder_complex_data'.format(npy_folder_location), xy)

    with open('./{}encoder_complex_data_lbl.pickle'.format(encoder_folder_location), 'wb') as f:
        pickle.dump(encoder, f)
else:
    np.save('./{}encoder{}_data'.format(npy_folder_location, name), xy)

    with open('./{}encoder{}_data_lbl.pickle'.format(encoder_folder_location, name), 'wb') as f:
        pickle.dump(encoder, f)