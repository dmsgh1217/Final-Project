# 전처리된 데이터를 활용하여 인공지능 모델을 학습하는 스크립트 입니다.

# Release 1.1 by Min-chul
# Dense Layer 출력 노드의 갯수를 전처리 된 카테고리의 갯수(output_units)만큼 자동으로 반영되도록 수정
# 일부 코드 재구성(Refactoring)

# Release 1.0 by Min-chul

from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os


def train_model(**kwargs):
    try:
        model_sequence = int(kwargs['model_sequence']) if 'model_sequence' in kwargs else 1
    except Exception as Err:
        model_sequence = 1
        print(f'Malformed "model_sequence" parameter.. Set to default (1).\nError: {Err}')

    filename = kwargs['filename'] if 'filename' in kwargs else 'model'
    # 현재 작업중인 디렉토리에 "models" 디렉토리가 없는 경우, 디렉토리를 생성합니다.
    os.makedirs('models', exist_ok=True)

    # 모델의 입력 차원을 정의합니다.
    input_shape = x_train[0].shape
    # Dense Layer 출력 노드의 갯수를 전처리된 카테고리의 갯수만큼 정의합니다.
    output_units = len(y_train[0])
    print(f'input_shape: {input_shape}')
    if model_sequence == 1:
        epoch = 150
        network = Sequential([Dense(256, input_shape=input_shape, activation='relu'),
                              Dropout(0.3),
                              Dense(128, activation='relu'),
                              Dropout(0.2),
                              Dense(64, activation='relu'),
                              Dropout(0.2),
                              Dense(32, activation='relu'),
                              Dropout(0.1),
                              Dense(16, activation='relu'),
                              Dense(output_units, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}_seq{model_sequence}.h5',
                                                                 monitor='val_accuracy', verbose=1, save_best_only=True,
                                                                 mode='auto'),
                                                 ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                                   verbose=1, mode='auto')])
        return network, network_history
    elif model_sequence == 2:
        epoch = 300
        network = Sequential([Dense(256, input_shape=input_shape, activation='relu'),
                              Dropout(0.3),
                              Dense(128, activation='relu'),
                              Dropout(0.2),
                              Dense(64, activation='relu'),
                              Dropout(0.2),
                              Dense(32, activation='relu'),
                              Dropout(0.1),
                              Dense(16, activation='relu'),
                              Dense(output_units, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}_seq{model_sequence}.h5',
                                                                 monitor='val_accuracy', verbose=1, save_best_only=True,
                                                                 mode='auto'),
                                                 ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                                   verbose=1, mode='auto')])
        return network, network_history
    elif model_sequence == 3:
        epoch = 300
        network = Sequential([Dense(512, input_shape=input_shape, activation='relu'),
                              Dropout(0.25),
                              Dense(256, activation='relu'),
                              Dropout(0.2),
                              Dense(128, activation='relu'),
                              Dropout(0.2),
                              Dense(64, activation='relu'),
                              Dropout(0.2),
                              Dense(32, activation='relu'),
                              Dropout(0.1),
                              Dense(16, activation='relu'),
                              Dense(output_units, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}_seq{model_sequence}.h5',
                                                                 monitor='val_accuracy', verbose=1, save_best_only=True,
                                                                 mode='auto'),
                                                 ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                                   verbose=1, mode='auto')])
        return network, network_history
    # 3번 모델에서 학습률(Learning Rate) 통제 정책을 사용하지 않도록 변경
    # 3번 모델 대비 성능 악화(0.8764)
    elif model_sequence == 4:
        epoch = 200
        network = Sequential([Dense(512, input_shape=input_shape, activation='relu'),
                              Dropout(0.25),
                              Dense(256, activation='relu'),
                              Dropout(0.2),
                              Dense(128, activation='relu'),
                              Dropout(0.2),
                              Dense(64, activation='relu'),
                              Dropout(0.2),
                              Dense(32, activation='relu'),
                              Dropout(0.1),
                              Dense(16, activation='relu'),
                              Dense(output_units, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}_seq{model_sequence}.h5',
                                                                 monitor='val_accuracy', verbose=1, save_best_only=True,
                                                                 mode='auto')])
        return network, network_history
    # 3번 모델에서 활성화 함수를 "ReLU"에서 "ELU"로 변경 (0.9976)
    elif model_sequence == 5:
        epoch = 200
        network = Sequential([Dense(512, input_shape=input_shape, activation='elu'),
                              Dropout(0.25),
                              Dense(256, activation='elu'),
                              Dropout(0.2),
                              Dense(128, activation='elu'),
                              Dropout(0.2),
                              Dense(64, activation='elu'),
                              Dropout(0.2),
                              Dense(32, activation='elu'),
                              Dropout(0.1),
                              Dense(16, activation='elu'),
                              Dense(output_units, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}_seq{model_sequence}.h5',
                                                                 monitor='val_accuracy', verbose=1, save_best_only=True,
                                                                 mode='auto'),
                                                 ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                                   verbose=1, mode='auto')])
        return network, network_history
    # 5번 모델에서 활성화 함수를 "ELU"에서 "Swish"로 변경 (0.9989)
    elif model_sequence == 6:
        epoch = 200
        network = Sequential([Dense(512, input_shape=input_shape, activation='swish'),
                              Dropout(0.25),
                              Dense(256, activation='swish'),
                              Dropout(0.2),
                              Dense(128, activation='swish'),
                              Dropout(0.2),
                              Dense(64, activation='swish'),
                              Dropout(0.2),
                              Dense(32, activation='swish'),
                              Dropout(0.1),
                              Dense(16, activation='swish'),
                              Dense(output_units, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}_seq{model_sequence}.h5',
                                                                 monitor='val_accuracy', verbose=1, save_best_only=True,
                                                                 mode='auto'),
                                                 ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                                   verbose=1, mode='auto')])
        return network, network_history
    else:
        epoch = 100
        print(f'There is no model corresponding to "model_sequence({model_sequence})".'
              f'Training proceeds with the basic model.')
        network = Sequential([Dense(256, input_shape=input_shape, activation='relu'),
                              Dropout(0.3),
                              Dense(128, activation='relu'),
                              Dropout(0.2),
                              Dense(64, activation='relu'),
                              Dropout(0.2),
                              Dense(32, activation='relu'),
                              Dropout(0.1),
                              Dense(16, activation='relu'),
                              Dense(output_units, activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        network.summary()
        network_history = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch,
                                      callbacks=[ModelCheckpoint(f'./models/{filename}_based.h5', monitor='val_accuracy',
                                                                 verbose=1, save_best_only=True, mode='auto'),
                                                 ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10,
                                                                   verbose=1, mode='auto')])
        return network, network_history


if __name__ == '__main__':

    # 전처리된 데이터(.npy)를 불러옵니다.
    x_train, x_test, y_train, y_test = np.load(file='./resources/encoder_loc_data_d42.npy', allow_pickle=True)

    # 인공지능 모델 함수를 호출하여 전처리된 데이터를 학습하고 결과값을 획득합니다.
    # model_sequence: 구성한 인공지능 네트워크에 번호를 부여하여 전달한 번호의 네트워크에 학습을 진행합니다.
    # filename: 저장되는 모델의 이름을 지정합니다.
    model_name = 'model'
    sequence_number = 5
    model, history = train_model(model_sequence=sequence_number, filename=model_name)

    # 학습 결과를 그래프로 시각화하여 표현합니다.
    plt.figure(figsize=(10, 12))
    plt.suptitle(t=f'Learning result of "{model_name}_seq{sequence_number}"'
                   f'({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})', fontsize=20)

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
