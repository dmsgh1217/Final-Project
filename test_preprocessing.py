import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

df = pd.read_csv('./resources/rock_scissor_paper_nomallize_data_w_640_h_480.csv')

print(df.head())
df.info()

Y = df['category']  # 카테고리 y값으로 추출
X = df.loc[:, df.columns != 'category']  # 카테고리를 제외한 X값 추출

X.info()
X = X.to_numpy()

# y = pd.get_dummies(y)
# print(y[:5])

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)  # 라벨을 숫자에 대응
label = encoder.classes_  # 라벨 - 숫자 목록

with open('./resources/encoder_ro_sci_pa.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_y)  # onehotencoding
print(X.shape)
print(onehot_Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, onehot_Y, test_size=0.05)  # 테스트 데이터 분리
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# exit()
xy = X_train, X_test, Y_train, Y_test

np.save('./resources/test_ro_sci_pa_preprocessing_data', xy)