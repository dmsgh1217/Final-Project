import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = np.load('./resources/test_ro_sci_pa_preprocessing_data.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Dense(256, input_dim = 42, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.05))
model.add(Dense(3, activation = 'softmax'))  # 다진 분류기에 사용, 가장 확률이 높은 분류에 배분한다.
print(model.summary())

opt = Adam(lr = 0.001)  # 러닝 레이트 값 설정
model.compile(opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size = 5, epochs = 20, verbose = 1)

score = model.evaluate(X_test, Y_test, verbose = 0)
print('Final test set accuracy : ', score[1])


plt.plot(fit_hist.history['accuracy'])
plt.show()


model.save('./resources/test_classfication_locate.h5')