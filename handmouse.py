# Release 1.5 by Min-chul
# 일시정지(Pause) 이벤트 리턴 후 연속적으로 일정시간(1초) 동안 Pause 동작을 리턴하지 못하도록 Block 기능 추가

# Release 1.4 by Min-chul
# 스크린샷(Screenshot) 이벤트 처리시 중간에 다른 동작이 간섭하였을 때 "screenshot"으로 결과가 나오는 문제점 해결
# 상기 문제 해결은 "Pause" 동작 해결과 동일 방법(call_count 계산)으로 해결하였음.

# Release 1.3 by Min-chul
# 리팩토링(Refactoring) 진행
# 외부입력 카메라 관리(management) 및 opencv 라이브러리는 "gui_app.py"로 이관
# 스크린샷(screenshot), 일시정지(pause), 스크롤(scroll) 처리 로직 추가

# Release 1.2 by Min-chul
# cv2 객체를 관리하는 함수(draw_frame) 추가 - 더 이상 메인(main)에서 관리하지 않음.
# 마우스 이벤트를 관리하는 함수(execute_mouse_event) 추가

# Release 1.1 by Min-chul
# 멀티스레드 실행하는 코드를 함수화
# 42차원의 좌표(Location) 데이터와 좌표 데이터를 기반으로 산출한 14차원의 각도(Angle) 데이터를 병합하는 알고리즘 추가
# pyautogui 라이브러리 제거 -> GUI 스크립트에서 사용 예정

from collections import deque
from tensorflow.keras.models import load_model
from threading import Thread
import numpy as np
import os
import pickle
import time
import prj_function_directory as pfd

model = None
label = None
result_asset = {}
event_asset = {}
flag_asset = {}
call_count = 0
check_duplicate_queue = deque()


def initialize(**kwargs):
    """
    model_path
    학습된 인공지능 모델 파일(.h5)의 디렉토리를 지정합니다.
    encoder_path
    학습에 사용된 데이터의 인코더 파일(.pickle)의 디렉토리를 지정합니다.
    """
    global model, label, result_asset, event_asset, flag_asset
    model_path = kwargs['model_path'] if 'model_path' in kwargs else './models/model_seq5_loc_angle(default).h5'
    encoder_path = kwargs['encoder_path'] if 'encoder_path' in kwargs else './resources/encoder_loc_angle_data_lbl_d56.pickle'

    # 현재 작업중인 디렉토리에 "screenshot" 디렉토리가 없는 경우, 디렉토리를 생성합니다.
    os.makedirs('screenshot', exist_ok=True)
    # 학습된 인공지능 모델을 불러옵니다.
    model = load_model(filepath=model_path)
    # 카테고리와 관련된 인코더 정보를 불러옵니다.
    with open(file=encoder_path, mode='rb') as f:
        label = pickle.load(f).classes_

    """
    학습된 인공지능 모델이 분류한 카테고리와 분류 확률 값을 관리하는 변수(predict_result)를 초기화합니다.
    category:   한 개 프레임(frame)을 분류한 카테고리 값을 저장합니다.
    ratio:      한 개 프레임(frame)을 분류한 확률 값을 저장합니다.   
    """
    result_asset = {'category': 'default', 'ratio': np.float(0.0)}

    """
    이벤트 처리를 종합적으로 관리하는 변수(event_asset)를 초기화합니다.
    sensitivity:        인공지능 모델이 분류한 카테고리의 확률값이 일정 수치 이상에서만 이벤트로 처리하도록 민감도(Sensitivity)를 정의합니다.
    switch:             일시정지(Pause) 동작에 의한 이벤트 처리 유무를 관리하는 값을 정의합니다. (False: 실행, True: 실행하지 않음)
    event:              최종적으로 실행하고자 하는 이벤트 값을 정의합니다.
    lclick_interval:    드래그, 클릭, 더블클릭을 구분하기 위한 상태 변화 최대 감지 시간을 정의합니다. (default 0.5초)
    rclick_interval:    우클릭(rightclick) 상태를 확인하기 위한 상태 변화 최대 감지 시간을 정의합니다. (default 0.4초)
    screen_interval:    스크린샷을 처리를 하기위한 상태 변화 최대 감지 시간을 정의합니다. (default 1초)
    locked_interval:    일시정지(Pause) 상태로 접근하기 위해 지속적으로 Pause 상태를 유지해야하는 시간을 정의합니다.
    unlocked_interval:  일시정지(Pause) 상태를 해제하기 위해 지속적으로 Pause 상태를 유지해야하는 시간을 정의합니다.
    """
    event_asset = {'sensitivity': 0.99, 'switch': False, 'event': 'default', 'lclick_interval': 0.5,
                   'rclick_interval': 0.4, 'screen_interval': 1.0, 'locked_interval': 2.0, 'unlocked_interval': 1.0}

    """
    이벤트 처리에 필요한 플래그 변수(flag_asset)를 초기화합니다.
    lclick_trigger: 카테고리가 "leftclick"으로 분류 되었을 때, 좌클릭과 관련된 처리를 위한 멀티스레드 상태를 명시하는 논리값을 정의합니다.
    lclick_run:     드래그, 클릭, 더블클릭 이벤트 처리를 위한 스레드 동작 상태를 명시하는 논리값을 정의합니다.
    rclick_trigger: 카테고리가 "rightclick"으로 분류 되었을 때, 좌클릭과 관련된 처리를 위한 멀티스레드 상태를 명시하는 논리값을 정의합니다.
    rclick_run:     연속적인 "rightclick" 카테고리를 확인 및 처리하기 위한 스레드 동작 상태를 명시하는 논리값을 정의합니다.
    screen_trigger: 카테고리가 "screenshot"으로 분류 되었을 때, 스크린샷 처리를 위한 멀티스레드 상태를 명시하는 논리값을 정의합니다.
    screen_run:     연속적인 "screenshot" 카테고리를 확인 및 처리하기 위한 스레드 동작 상태를 명시하는 논리값을 정의합니다.
    pause_trigger:  카테고리가 "pause"로 분류되었을 때, 동작 일시정지 처리를 위한 멀티스레드 상태를 명시하는 논리값을 정의합니다. 
    pause_run:      연속적인 "pause" 카테고리를 확인 및 처리하기 위한 스레드 동작 상태를 명시하는 논리값을 정의합니다.
    """
    flag_asset = {'lclick_trigger': False, 'lclick_run': False, 'rclick_trigger': False, 'rclick_run': False,
                  'screen_trigger': False, 'screen_run': False, 'pause_trigger': False, 'pause_run': False}

    # 멀티스레드를 실행합니다.
    initialize_thread()


def initialize_thread():
    # 좌클릭 이벤트 처리를 위한 멀티스레드를 실행합니다.
    try:
        thread_left_click = Thread(target=_thread_left_click, name='thread_left_click')
        thread_left_click.daemon = True
        thread_left_click.start()
    except Exception as E:
        print(f'Cannot launched "{thread_left_click.name}" thread..\nError: {E}')
        exit()
    else:
        print(f'"{thread_left_click.name}" thread start.')

    # 우클릭 이벤트 처리를 위한 멀티스레드를 실행합니다.
    try:
        thread_right_click = Thread(target=_thread_right_click, name='thread_right_click')
        thread_right_click.daemon = True
        thread_right_click.start()
    except Exception as E:
        print(f'Cannot launched "{thread_right_click.name}" thread..\nError: {E}')
        exit()
    else:
        print(f'"{thread_right_click.name}" thread start.')

    # 스크린샷 이벤트 처리를 위한 멀티스레드를 실행합니다.
    try:
        thread_screenshot = Thread(target=_thread_screenshot, name='thread_screenshot')
        thread_screenshot.daemon = True
        thread_screenshot.start()
    except Exception as E:
        print(f'Cannot launched "{thread_screenshot.name}" thread..\nError: {E}')
        exit()
    else:
        print(f'"{thread_screenshot.name}" thread start.')

    # 일시정지 이벤트 처리를 위한 멀티스레드를 실행합니다.
    try:
        thread_pause = Thread(target=_thread_pause, name='thread_pause')
        thread_pause.daemon = True
        thread_pause.start()
    except Exception as E:
        print(f'Cannot launched "{thread_pause.name}" thread..\nError: {E}')
        exit()
    else:
        print(f'"{thread_pause.name}" thread start.')


def _thread_left_click():
    global flag_asset, event_asset
    while True:
        if flag_asset['lclick_trigger'] and not flag_asset['lclick_run']:
            # 좌클릭(left-click) 이벤트 처리를 위한 멀티스레드를 동작중(True)으로 명시합니다.
            flag_asset['lclick_run'] = True
            # 상태 변화를 감지하기 위한 상태 기록(record_status)을 초기화합니다.
            record_status = 'leftclick'
            # 함수가 실행되는 동안 상태 변화가 몇 번 발생하였는지 카운트하는 변수를 초기화합니다.
            status_count = 0
            # 동작 인식 제한시간을 측정하기 위한 시간 측정을 시작합니다.
            rec_time = time.time()
            # 추가 동작 감지 대기 시간동안 반복하여 상태 변화가 얼마나 발생했는지 계산하는 반복문을 시행합니다.
            while time.time() - rec_time < event_asset['lclick_interval']:
                # 실시간으로 분류하고 있는 모델의 결과가 스레드 내에서 기록한 상태와 다를 경우, 상태의 변화가 발생하였다고 처리합니다.
                if result_asset['category'] != record_status:
                    # 상태가 변화해도 "Move" 또는 "leftclick"상태로 변화하였을 때만 상태변화로 처리합니다.
                    if result_asset['category'] == 'leftclick' or result_asset['category'] == 'move':
                        # 상태가 변화하였으므로, 현재 상태값을 모델이 분류한 결과값으로 대체합니다.
                        record_status = result_asset['category']
                        # 상태 변화가 발생하였으므로 카운트 변수를 추가합니다.
                        status_count += 1
                # 제한 시간 내에 3회 이상 과도한 변화가 발생한 경우, 반복문을 종료합니다.
                if status_count >= 3:
                    break
                time.sleep(0.01)
            # 상태 변화 횟수가 0이면 제한 시간내에 계속해서 "Click"상태를 유지하고 있으므로 "Drag"이벤트로 연결합니다.
            if status_count == 0:
                event_asset['event'] = 'drag'
            # 상태 변화 횟수가 1이면 제한 시간내에 "Move"로 변화하였으므로 "Click"이벤트로 연결합니다.
            elif status_count == 1:
                event_asset['event'] = 'leftclick'
            # 상태 변화 횟수가 3회 이상이면, 제한 시간내에 "Move" -> "Click" -> "Move"로 변화하였으므로 "Double-Click"이벤트로 연결합니다.
            elif status_count >= 3:
                event_asset['event'] = 'doubleclick'

            # 스레드 내부 실행이 종료되었으므로, 부분 동작 처리를 활성화 할 수 있도록 flag_asset['click_trigger']를 "False"로 변경합니다.
            flag_asset['lclick_trigger'] = False
            # 스레드 내부 실행이 종료되었으므로, flag_asset['click_run'] 변수를 "False"로 변경하여 비활성화 상태임을 명시합니다.
            flag_asset['lclick_run'] = False
        time.sleep(0.01)


def _thread_right_click():
    global flag_asset, event_asset
    while True:
        if flag_asset['rclick_trigger'] and not flag_asset['rclick_run']:
            # 우클릭(right-click) 이벤트 처리를 위한 멀티스레드를 동작중(True)으로 명시합니다.
            flag_asset['rclick_run'] = True
            # 상태 변화를 감지하기 위한 상태 기록(record_status)을 초기화합니다.
            record_status = 'rightclick'
            # 함수가 실행되는 동안 상태 변화가 몇 번 발생하였는지 카운트하는 변수를 초기화합니다.
            status_count = 0
            # 동작 인식 제한시간을 측정하기 위한 시간 측정을 시작합니다.
            rec_time = time.time()
            # 추가 동작 감지 대기 시간동안 반복하여 상태 변화가 얼마나 발생했는지 계산하는 반복문을 시행합니다.
            while time.time() - rec_time < event_asset['rclick_interval']:
                # 실시간으로 분류하고 있는 모델의 결과가 스레드 내에서 기록한 상태와 다를 경우, 상태의 변화가 발생하였다고 처리합니다.
                if result_asset['category'] != record_status:
                    # 상태가 변화해도 "move" 또는 "rightclick"상태로 변화하였을 때만 상태변화로 처리합니다.
                    if result_asset['category'] == 'rightclick' or result_asset['category'] == 'move':
                        # 상태 변화가 발생하였으므로 카운트 변수를 추가하고, 상태 변화가 발생하였으므로 감지 대기를 중지합니다.
                        status_count += 1
                        break
                time.sleep(0.01)
            # 상태 변화 횟수가 1 이상이면 제한 시간내에 계속해서 "rightclick"상태를 유지하고 있는것으로 간주합니다.
            if status_count >= 1:
                event_asset['event'] = 'rightclick'

            # 스레드 내부 실행이 종료되었으므로, 부분 동작 처리를 활성화 할 수 있도록 flag_asset['click_trigger']를 "False"로 변경합니다.
            flag_asset['rclick_trigger'] = False
            # 스레드 내부 실행이 종료되었으므로, flag_asset['click_run'] 변수를 "False"로 변경하여 비활성화 상태임을 명시합니다.
            flag_asset['rclick_run'] = False
        time.sleep(0.01)


def _thread_screenshot():
    global flag_asset, event_asset, call_count
    while True:
        if flag_asset['screen_trigger'] and not flag_asset['screen_run']:
            # 스크린샷(screenshot) 이벤트 처리를 위한 멀티스레드를 동작중(True)으로 명시합니다.
            flag_asset['screen_run'] = True
            # 상태 변화를 감지하기 위한 상태 기록(record_status)을 초기화합니다.
            record_status = 'screenshot'
            # 함수가 실행되는 동안 상태 변화가 몇 번 발생하였는지 카운트하는 변수를 초기화합니다.
            status_count = 0
            # 동작 인식 제한시간을 측정하기 위한 시간 측정을 시작합니다.
            rec_time = time.time()
            # 추가 동작 감지 대기 시간동안 반복하여 상태 변화가 얼마나 발생했는지 계산하는 반복문을 시행합니다.
            while time.time() - rec_time < event_asset['screen_interval']:
                # 실시간으로 분류하고 있는 모델의 결과가 스레드 내에서 기록한 상태와 다를 경우, 상태의 변화가 발생하였다고 처리합니다.
                if result_asset['category'] != record_status:
                    # 상태 변화가 발생하였으므로 카운트 변수를 추가하고, 상태 변화가 발생하였으므로 감지 대기를 중지합니다.
                    status_count += 1
                    break
                time.sleep(0.01)
            # 상태 변화 횟수가 0이면 제한 시간내에 계속해서 "Screenshot"상태를 유지하고 있는것으로 간주합니다.
            print(f'call_count(in screenshot): {call_count}')
            if status_count == 0 and call_count >= 6:
                event_asset['event'] = 'screenshot'
            else:
                event_asset['event'] = 'default'
            call_count = 0
            # 스레드 내부 실행이 종료되었으므로, 부분 동작 처리를 활성화 할 수 있도록 flag_asset['screen_trigger']를 "False"로 변경합니다.
            flag_asset['screen_trigger'] = False
            # 스레드 내부 실행이 종료되었으므로, flag_asset['screen_run'] 변수를 "False"로 변경하여 비활성화 상태임을 명시합니다.
            flag_asset['screen_run'] = False
        time.sleep(0.01)


def _thread_pause():
    global flag_asset, event_asset, call_count
    while True:
        if flag_asset['pause_trigger'] and not flag_asset['pause_run']:
            # 일시정지(Pause) 이벤트 처리를 위한 멀티스레드를 동작중(True)으로 명시합니다.
            flag_asset['pause_run'] = True
            # 상태 변화를 감지하기 위한 상태 기록(record_status)을 초기화합니다.
            record_status = 'pause'
            # 함수가 실행되는 동안 상태 변화가 몇 번 발생하였는지 카운트하는 변수를 초기화합니다.
            status_count = 0
            # 동작 인식 제한시간을 측정하기 위한 시간 측정을 시작합니다.
            rec_time = time.time()
            # 일시정지 상태가 아닌 상태에서 "Pause" 동작이 발생한 경우 이하 스크립트를 실행합니다.
            if not event_asset['switch']:
                while time.time() - rec_time < event_asset['locked_interval']:
                    # 실시간으로 분류하고 있는 모델의 결과가 스레드 내에서 기록한 상태와 다를 경우, 상태의 변화가 발생하였다고 처리합니다.
                    if result_asset['category'] != record_status:
                        # 상태 변화가 발생하였으므로 카운트 변수를 추가하고, 상태 변화가 발생하였으므로 감지 대기를 중지합니다.
                        status_count += 1
                        break
                    time.sleep(0.01)
            else:
                while time.time() - rec_time < event_asset['unlocked_interval']:
                    # 실시간으로 분류하고 있는 모델의 결과가 스레드 내에서 기록한 상태와 다를 경우, 상태의 변화가 발생하였다고 처리합니다.
                    if result_asset['category'] != record_status:
                        # 상태 변화가 발생하였으므로 카운트 변수를 추가하고, 상태 변화가 발생하였으므로 감지 대기를 중지합니다.
                        status_count += 1
                        break
                    time.sleep(0.01)
            # 상태 변화 횟수가 0이면 제한 시간내에 계속해서 "Pause"상태를 유지하고 있는것으로 간주합니다.
            if status_count == 0:
                if event_asset['switch']:
                    if call_count >= 5:
                        event_asset['switch'] = False
                else:
                    if call_count >= 10:
                        event_asset['switch'] = True
            call_count = 0
            # 동작이 완료된 후 일정 시간(1초) 동안 연속적인 Pause 동작을 할 수 없도록 Block 합니다.
            time.sleep(1)
            # 스레드 내부 실행이 종료되었으므로, 부분 동작 처리를 활성화 할 수 있도록 flag_asset['pause_trigger']를 "False"로 변경합니다.
            flag_asset['pause_trigger'] = False
            # 스레드 내부 실행이 종료되었으므로, flag_asset['pause_run'] 변수를 "False"로 변경하여 비활성화 상태임을 명시합니다.
            flag_asset['pause_run'] = False
            call_count = 0
        time.sleep(0.01)


def calculate_loc_info(landmarks):
    global check_duplicate_queue, flag_asset, call_count
    call_count += 1
    # 결과 값(return)을 전달할 때 중복되어 전달되면 안되는 카테고리를 정의합니다.
    duplicate_list = ['leftclick', 'doubleclick', 'screenshot', 'rightclick']

    # 42차원의 좌표 데이터를 각도 변환 함수(convert_angle())의 입력에 사용할 수 있도록 딕셔너리(location)로 재구성합니다.
    location = {'x': [], 'y': []}
    for idx in range(len(landmarks)):
        location['x'].append(landmarks[idx][0])
        location['y'].append(landmarks[idx][1])

    # 딕셔너리(location)를 이용하여 14차원의 각도(Angle)로 변환해주는 함수(convert_angle())를 호출합니다.
    angle = np.array(pfd.convert_angle(x=location['x'], y=location['y']))
    # 기존에 구성된 42차원의 좌표데이터를 갖고있는 변수(coordination)를 1차원 데이터로 변형합니다.
    coordination = landmarks.flatten()
    # 모델의 입력에 사용할 변수(input_data)를 42차원의 좌표 데이터와 14차원의 각도 데이터를 병합한 형태로 구성합니다.
    input_data = np.concatenate([coordination, angle])
    # 인공지능 모델의 입력 형태에 맞추기 위해 변수(segment)의 차원을 (1, n)으로 변형합니다.
    input_data = input_data.reshape(-1, input_data.shape[0])
    # 랜드마크의 좌표(location)를 이용한 중심좌표(momentum_position)를 산출합니다.
    momentum_position = pfd.reference_loc(x=location['x'], y=location['y'])

    try:
        model_predict = model.predict(input_data)
    except ValueError:
        print(f'An invalid input shape{input_data.shape} was detected..')
    except Exception as Err:
        print(f'Unknown error occurred.. {Err}')
    else:
        # 학습된 인공지능 모델이 분류한 카테고리 결과 값을 변수에 저장합니다.
        result_asset['category'] = label[np.argmax(model_predict)]
        # 분류된 카테고리의 확률 값을 변수에 저장합니다.
        result_asset['ratio'] = np.max(model_predict)
        # 확률 값이 sensitivity(0.99) 이상인 경우에만 최종 이벤트로서 해당 카테고리를 사용합니다.
        if result_asset['ratio'] >= event_asset['sensitivity']:
            # 분류 카테고리(result_asset['category']가 "leftclick"인 경우 클릭 이벤트 처리를 위한 멀티스레드를 활성화합니다.
            if result_asset['category'] == 'leftclick':
                if not flag_asset['lclick_run']:
                    flag_asset['lclick_trigger'] = True
            # 분류 카테고리(result_asset['category']가 "rightclick"인 경우 우클릭 이벤트 처리를 위한 멀티스레드를 활성화합니다.
            elif result_asset['category'] == 'rightclick':
                if not flag_asset['rclick_run']:
                    flag_asset['rclick_trigger'] = True
            # 분류 카테고리(result_asset['category']가 "screenshot"인 경우 스크린샷 이벤트 처리를 위한 멀티스레드를 활성화합니다.
            elif result_asset['category'] == 'screenshot':
                if not flag_asset['screen_run']:
                    flag_asset['screen_trigger'] = True
            # 분류 카테고리(result_asset['category']가 "pause"인 경우 스크린샷 이벤트 처리를 위한 멀티스레드를 활성화합니다.
            elif result_asset['category'] == 'pause':
                if not flag_asset['pause_run']:
                    flag_asset['pause_trigger'] = True
            elif result_asset['category'] == 'scroll':
                event_asset['event'] = result_asset['category']
        else:
            event_asset['event'] = 'default'

        # 최종 이벤트 전달값이 중복되면 안되는 카테고리(duplicate_list)에 대해 중복되지 않도록 처리합니다.
        if event_asset['event'] in duplicate_list:
            # 큐(check_duplicate_queue)에 최종 이벤트 전달값을 추가합니다.
            check_duplicate_queue.append(event_asset['event'])
            # 큐(check_duplicate_queue)의 길이가 2보다 크거나 같은 경우 이하 처리를 진행합니다.
            if len(check_duplicate_queue) >= 2:
                # 직전 상태의 결과(previous_result)에 큐(check_duplicate_queue)의 첫 번째 등록된 요소를 할당합니다.
                previous_result = check_duplicate_queue.popleft()
                # 직전 상태의 결과와 마지막으로 등록된 최종 이벤트 전달값이 같은 경우, 중복하여 처리하지 못하도록 'default' 처리합니다.
                if previous_result == check_duplicate_queue[0]:
                    event_asset['event'] = 'default'
                    # 큐(check_duplicate_queue)를 초기화합니다.
                    check_duplicate_queue = deque()

        # "pause"동작에 의한 시스템 일시정지가 되었는지 검사합니다. 일시 정지 상태인 경우, 모든 전달값은 "default"로 전달합니다.
        if not event_asset['switch']:
            return momentum_position, event_asset['event'], event_asset['switch']
        else:
            event_asset['event'] = 'default'
            return momentum_position, event_asset['event'], event_asset['switch']


def main():
    initialize()


if __name__ == '__main__':
    main()
