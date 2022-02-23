from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt


def train_model(**kwargs):
    try:
        model_sequence = int(kwargs['model_sequence']) if 'model_sequence' in kwargs else 1
    except Exception as Err:
        model_sequence = 1
        print(f'Malformed "model_sequence" parameter.. Set to default (1).\nError: {Err}')

    filename = kwargs['filename'] if 'filename' in kwargs else 'model'
    print(type(model_name), filename)
    if model_sequence == 1:
        print(f'input_shape: {x_train[0].shape}')
        network = Sequential([Dense(256, input_shape=x_train[0].shape, activation='relu'),
                              Dropout(0.3),
                              Dense(128, activation='relu'),
                              Dropout(0.2),
                              Dense(64, activation='relu'),
                              Dropout(0.2),
                              Dense(32, activation='relu'),
                              Dropout(0.1),
                              Dense(16, activation='relu'),
                              Dense(2, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}.h5', monitor='val_accuracy',
                                                                 verbose=1, save_best_only=True, mode='auto'),
                                                 ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                                   verbose=1, mode='auto')])
        return network, network_history
    else:
        print(f'There is no model corresponding to "model_sequence({model_sequence})".'
              f'Training proceeds with the basic model.')
        print(f'input_shape: {x_train[0].shape}')
        network = Sequential([Dense(256, input_shape=x_train[0].shape, activation='relu'),
                              Dropout(0.3),
                              Dense(128, activation='relu'),
                              Dropout(0.2),
                              Dense(64, activation='relu'),
                              Dropout(0.2),
                              Dense(32, activation='relu'),
                              Dropout(0.1),
                              Dense(16, activation='relu'),
                              Dense(2, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}.h5', monitor='val_accuracy',
                                                                 verbose=1, save_best_only=True, mode='auto'),
                                                 ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                                   verbose=1, mode='auto')])
        return network, network_history


if __name__ == '__main__':
    # 학습 하려는 액션(category)을 정의합니다.
    action = ['move', 'click']
    # 모델 학습량(epochs)을 정의합니다.
    epochs = 150

    # 전처리된 데이터(.npy)를 불러옵니다.
    x_train, x_test, y_train, y_test = np.load(file='./dataset/encoder_loc_data.npy', allow_pickle=True)
    print(f'x_train.shape: {x_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'x_test.shape: {x_test.shape}')
    print(f'y_test.shape: {y_test.shape}')

    # 인공지능 모델 함수를 호출하여 전처리된 데이터를 학습하고 결과값을 획득합니다.
    # model_sequence: 구성한 인공지능 네트워크에 번호를 부여하여 전달한 번호의 네트워크에 학습을 진행합니다.
    # filename: 저장되는 모델의 이름을 지정합니다.
    model_name = 'test'
    model, history = train_model(model_sequence=1, filename=model_name)

    # 학습 결과를 그래프로 시각화하여 표현합니다.
    plt.figure(figsize=(10, 12))
    plt.suptitle(t=f'Learning result of "{model_name}"', fontsize=20)

    plt.subplot(2, 1, 1)
    plt.title(label=f'Accuracy', loc='left', color='blue', fontsize=15)
    plt.plot(history.history['accuracy'], color='black', label='Train', linewidth=1)
    plt.plot(history.history['val_accuracy'], color='lime', label='Validation', linewidth=2.5)
    plt.xlabel(xlabel='Epochs')
    plt.ylabel(ylabel='Ratio(%)')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.title(label='Loss', loc='left', color='blue', fontsize=15)
    plt.plot(history.history['loss'], color='black', label='Train', linewidth=1)
    plt.plot(history.history['val_loss'], color='lime', label='Validation', linewidth=2.5)
    plt.xlabel(xlabel='Epochs')
    plt.ylabel(ylabel='Value')
    plt.legend(loc='upper right')
    plt.show()
