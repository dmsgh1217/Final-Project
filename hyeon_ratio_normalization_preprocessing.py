import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

def sol_length(x1, y1, x2, y2):  # 길이 함수
  return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def sol_ratio(x1, y1, x2, y2, x3, y3):  # 비율 함수
  len_a = sol_length(x1, y1, x3, y3)
  len_b = sol_length(x2, y2, x3, y3)
  return len_a / len_b

df = pd.read_csv('./resources/파일명.csv')  # 추후 파일명에 맞게 수정

print(df.head())
df.info()

Y = df['category']  # 카테고리 y값으로 추출
X = df.loc[:, df.columns != 'category']  # 카테고리를 제외한 X값 추출

result_df = pd.DataFrame()  # 비율 처리를 위한 빈 데이터 프레임 생성

ratio_0, ratio_1, ratio_2, ratio_3, ratio_4 = [], [], [], [], []  # 손가락 리스트(업지 ~ 새끼)

for i in range(len(X)):  #비율 함수를 통해 생성된 값을 리스트에 저장
    ratio_0.append(sol_ratio(X['x4'][i], X['y4'][i], X['x5'][i], X['y5'][i], X['x17'][i], X['y17'][i]))
    ratio_1.append(sol_ratio(X['x8'][i], X['y8'][i], X['x5'][i], X['y5'][i], X['x0'][i], X['y0'][i]))
    ratio_2.append(sol_ratio(X['x12'][i], X['y12'][i], X['x9'][i], X['y9'][i], X['x0'][i], X['y0'][i]))
    ratio_3.append(sol_ratio(X['x16'][i], X['y16'][i], X['x13'][i], X['y13'][i], X['x0'][i], X['y0'][i]))
    ratio_4.append(sol_ratio(X['x20'][i], X['y20'][i], X['x17'][i], X['y17'][i], X['x0'][i], X['y0'][i]))

# 위에서 받은 리스트를 result_df에 저장
result_df['ratio_0'] = ratio_0
result_df['ratio_1'] = ratio_1
result_df['ratio_2'] = ratio_2
result_df['ratio_3'] = ratio_3
result_df['ratio_4'] = ratio_4

result_df.info()

X = result_df / 2.5  # 비율 정규화

X.info()
X = X.to_numpy()  # np.array로 변환

# y = pd.get_dummies(y)
# print(y[:5])

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)  # 라벨을 숫자에 대응
label = encoder.classes_  # 라벨 - 숫자 목록

with open('./resources/test_encoder_ratio_normalization_lbl.pickle', 'wb') as f:
    pickle.dump(encoder, f)  # 정보 저장

onehot_Y = to_categorical(labeled_y)  # onehotencoding
print(X.shape)
print(onehot_Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, onehot_Y, test_size=0.1)  # 테스트 데이터 분리
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# exit()
xy = X_train, X_test, Y_train, Y_test

np.save('./resources/test_encoder_ratio_normalization_data', xy)