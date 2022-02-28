from tensorflow.keras.models import load_model
from threading import Thread
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import time


# 멀티스레드 - 1
# 영상으로부터 획득한 랜드마크의 좌표 또는 각도 및 비율 데이터를 이용하여 인공지능 모델로부터 출력값을 획득하는 스레드입니다.
def predict_motion():
    global PREDICT_FLAG, CLICK_FLAG, predict_result
    while True:
        if PREDICT_FLAG:
            PREDICT_FLAG = False
            # 인공지능 모델에 "segment"변수를 입력으로 주어 분류 결과를 획득합니다.
            # 영상에서 비 정상적인 랜드마크가 수집되는 경향이 있어 예외처리를 하여 시스템 다운을 방지합니다.
            try:
                predict_value = model.predict(segment)
            except Exception:
                print(f'Error occurred.. {segment.shape}')
            else:
                # "softmax"함수를 사용하여 획득한 결과값 중 가장 큰 확률값을 가진 카테고리를 라벨에 매핑(mapping)합니다.
                predict_result['category'] = label[np.argmax(predict_value)]
                # 가장 큰 확률값을 cv2.imshow()에서 확인하기 위해 자릿수 표현 및 문자열 변환 과정을 수행합니다.
                predict_result['ratio'] = ''.join([str(np.around(np.max(predict_value), 4)), '%'])
                # predict_result['ratio'] = np.max(predict_value)
                # print(predict_result['ratio'], type(predict_result['ratio']))
                # 모델의 분류 결과가 "click"이면 클릭 이벤트 처리를 위한 스레드 모듈 활성화를 진행합니다.
                if predict_result['category'] == 'click' and not CLICK_FLAG['run']:
                    CLICK_FLAG['detect'] = True
        time.sleep(0.01)


def click_trigger():
    global CLICK_FLAG, click_count
    while True:
        # 클릭(click) 이벤트가 감지되었으며 CLICK_FLAG['run']가 False 일 때만 동작합니다.
        # CLICK_FLAG['run'] 변수는 이하 스레드 내부 함수가 동작하고 있으면 True 상태가 되며, 작업이 완료되면 False 상태로 변경됩니다.
        # 이는 제한 시간 이내에 연속적으로 함수를 호출하는 것을 방지하기 위한 정책입니다.
        if CLICK_FLAG['detect'] and not CLICK_FLAG['run']:
            # 클릭(click) 이벤트 처리를 위한 작업을 시작하므로 CLICK_FLAG['run']를 True 상태로 변경합니다.
            CLICK_FLAG['run'] = True
            # 함수가 실행되는 동안 상태(Status)의 변화가 몇 번 발생했는지 카운트하는 변수를 초기화합니다.
            status_count = 0
            # 상태 변화를 감지하기 위해 현재 상태(current_status)를 모델이 분류한 값('click')으로 설정합니다.
            current_status = 'click'
            # 동작 인식 제한시간을 측정하기 위한 시간 측정을 시작합니다.
            s_time = time.time()
            # 동작 인식 제한시간(0.5초)동안 반복하여 상태 변화가 얼마나 발생했는지 계산하는 반복문을 시행합니다.
            while time.time() - s_time < click_interval:
                # 실시간으로 분류하고 있는 모델의 결과가 스레드 내에서 선언한 현재 상태와 다를 경우, 상태의 변화가 발생하였다고 처리합니다.
                if predict_result['category'] != current_status:
                    # 상태가 변화해도 "Move" 또는 "Click"상태로 변화하였을 때만 상태변화로 처리합니다.
                    if predict_result['category'] == 'click' or predict_result['category'] == 'move':
                        # 상태가 변화하였으므로, 현재 상태값을 모델이 분류한 결과값으로 대체합니다.
                        current_status = predict_result['category']
                        # 상태가 1회 변화하였으므로 카운트를 추가합니다.
                        status_count += 1
                # 제한 시간 내에 3회 이상 변화하였을 경우 더블 클릭 이상의 이벤트 처리가 없으므로 반복문을 종료합니다.
                if status_count >= 3:
                    break
                time.sleep(0.01)
            # 상태가 몇 번 변화하였는지 디버그합니다.
            # print(f'status_count: {status_count}')
            # 상태 변화 횟수가 0이면 제한 시간내에 계속해서 "Click"상태를 유지하고 있으므로 "Drag"이벤트로 연결합니다.
            if status_count == 0:
                print(f'Drag')
            # 상태 변화 횟수가 1이면 제한 시간내에 "Move"로 변화하였으므로 "Click"이벤트로 연결합니다.
            elif status_count == 1:
                print(f'Click')
            # 상태 변화 횟수가 3회 이상이면, 제한 시간내에 "Move" -> "Click" -> "Move"로 변화하였으므로 "Double-Click"이벤트로 연결합니다.
            elif status_count >= 3:
                print(f'Double-Click')

            # 스레드 내부 함수 실행이 종료되었으므로, 부분 동작 처리를 활성화 할 수 있도록 CLICK_FLAG['detect']를 "False"로 변경합니다.
            CLICK_FLAG['detect'] = False
            # 스레드 내부 함수 실행이 종료되었으므로, CLICK_FLAG['run'] 변수를 "False"로 변경하여 비활성화 상태임을 명시합니다.
            CLICK_FLAG['run'] = False
            print(f'end thread.')
        time.sleep(0.01)


if __name__ == '__main__':
    # 미디어파이프 라이브러리에서 제공하는 함수를 사용하기 위한 객체를 생성합니다.
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 설정하고자 하는 카메라의 해상도 값을 가져옵니다.
    cam_width, cam_height = 1280, 720
    # 인공지능 모델을 가져옵니다.
    model = load_model('./models/test2_seq5_test1.h5')
    # 라벨과 관련된 인코더 정보를 가져옵니다.
    with open('./resources/encoder_loc_data_lbl.pickle', 'rb') as f:
        encoder = pickle.load(f)
        label = encoder.classes_
    # 모델이 예측한 결과 값을 저장하는 객체를 생성합니다.
    predict_result = {'category': '', 'ratio': ''}
    # 사용자의 실제 디스플레이 해상도 값을 획득합니다.
    screen_width, screen_height = pyautogui.size()
    # 클릭 이벤트, 더블 클릭 이벤트, 드래그 이벤트에 사용할 변수를 초기화합니다.
    click_interval = 0.5
    click_count = 0

    # 멀티 스레드에서 사용하는 변수를 선언합니다.
    PREDICT_FLAG = False
    CLICK_FLAG = {'detect': False, 'run': False}

    # 멀티스레드 모듈을 초기화하고, 설정이 완료되면 실행합니다.
    while True:
        try:
            thread_predict_motion = Thread(target=predict_motion, name='predict_motion')
            thread_predict_motion.daemon = True
            thread_predict_motion.start()
        except Exception as E:
            print(f'Cannot launched "{thread_predict_motion.name}" thread.. retry after 3 seconds.\nError: {E}')
            time.sleep(3)
        else:
            print(f'"{thread_predict_motion.name}" thread start.')
            break

    while True:
        try:
            thread_click_trigger = Thread(target=click_trigger, name='click_trigger')
            thread_click_trigger.daemon = True
            thread_click_trigger.start()
        except Exception as E:
            print(f'Cannot launched "{thread_click_trigger.name}" thread.. retry after 3 seconds.\nError: {E}')
            time.sleep(3)
        else:
            print(f'"{thread_click_trigger.name}" thread start.')
            break

    print(f'Video resolution: {cam_width, cam_height}')
    print(f'Monitor resolution: {screen_width, screen_height}')

    # OpenCV 라이브러리를 이용하여 비디오 객체를 생성합니다.
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # 초당 프레임(FPS) 계산을 위해 현재 시간을 획득합니다.
        start_time = time.time()
        # 비디오 객체로부터 리턴 값과 영상을 가져옵니다.
        ret, frame = cap.read()
        # 영상을 좌우 반전하여 출력합니다.
        frame = cv2.flip(src=frame, flipCode=1)
        # 영상의 출력되는 사이즈를 조절합니다.
        frame = cv2.resize(src=frame, dsize=(cam_width, cam_height), interpolation=cv2.INTER_AREA)
        # 영상의 채널을 BGR 채널에서 RGB 채널로 변환합니다.
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        # 미디어파이프(mediapipe)의 함수에 현재 영상 정보를 입력으로 한 결과(return)를 받습니다.
        result = hands.process(frame)
        # 영상의 채널을 RGB 채널에서 BGR 채널로 변환합니다.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 미디어파이프 라이브러리로부터 획득한 결과 값(result.multi_hand_landmarks)을 mp_return_value 변수에 할당합니다.
        # 결과 값(result.multi_hand_landmarks)의 형태는 총 21개 랜드마크의 x, y, z 좌표를 가지고 있습니다.
        # visibility 데이터는 라이브러리 이슈가 있어서 제외하도록 합니다.
        mp_return_value = result.multi_hand_landmarks

        # mp_return_value 타입이 "None"이 아닌 경우에만 이하 스크립트를 실행합니다.
        if mp_return_value is not None:
            # 아래의 출력문(print)은 각 랜드마크의 x, y, z 좌표를 확인할 수 있습니다.
            """ print(result.multi_hand_landmarks) """
            # 1개 프레임(frame)에 대해서 각 랜드마크(최대 21개)에 대한 분석을 진행합니다.
            for landmark_index in mp_return_value:
                # 총 21개의 랜드마크가 갖고 있는 좌표(x, y) 데이터를 저장할 객체를 생성합니다.
                segment = np.zeros((21, 2))
                # 각 랜드마크(최대 21개)의 좌표 데이터 중 해상도를 벗어나는 값이 발생하는 경우, 이벤트 처리를 위한 플래그입니다.
                none_flag = False
                # 각 랜드마크의 좌표(x, y) 데이터를 행렬(segment)에 할당합니다.
                for idx, landmark in enumerate(landmark_index.landmark):
                    segment[idx] = landmark.x, landmark.y
                    # 랜드마크의 좌표가 해상도의 최대, 최소값을 벗어나는 경우, 플래그(none_flag)를 활성화 시켜서 이벤트 발생을 억제합니다.
                    if int(landmark.x * cam_width) < 0 or int(landmark.x * cam_width) > cam_width:
                        none_flag = True
                    if int(landmark.y * cam_height) < 0 or int(landmark.y * cam_height) > cam_height:
                        none_flag = True

                    # 비 정상적인 좌표값이 확인되었을 경우, 이후 프로세스를 진행하지 않고 다음 프레임(frame)을 분석합니다.
                    if none_flag:
                        break

                # 각 랜드마크의 좌표값이 정상인 경우에만 이하 프로세스를 실행합니다.
                if not none_flag:
                    # 구성된 segment 변수는 N 차원으로 구성된 데이터이므로, 1차원 데이터로 변형합니다.
                    segment = segment.flatten()
                    # 좌표 함수 추가(42차원)    .reshape(-1, 42)
                    # 각도 함수 추가(14차원)    .reshape(-1, 14)
                    # 비율 함수 추가(5차원)     .reshape(-1, 5)
                    # 모델의 입력 형태(42, )에 맞추기 위해 데이터(segment)의 형태를(1, 42)로 변형합니다.
                    segment = segment.reshape(-1, 42)
                    # 멀티스레드의 동작을 활성화 하기 위해 플래그를 설정합니다.
                    PREDICT_FLAG = True
                # print(type(segment), len(segment))
                # print(segment)
                ###
                """
                None값이 들어갈 수 있으므로
                여기서 None이 있나 없나를 검사 할 예정임.
                None이 없으면, 정상적으로 변환 함수 호출 -> 분류를 하겠다.
                None이 있으면, 호출하지 않고 그냥 Pass -> 분류를 하지 않겠다.
                """

        # 좌측 하단에 모델이 분류한 결과값을 표출합니다.
        # text = ''.join([predict_result[0], '   ', predict_result[1]])
        # 초당 프레임(FPS)수를 계산합니다.
        fps = int(1. / (time.time() - start_time))
        # 가장 큰 확률값을 cv2.imshow()에서 확인하기 위해 자릿수 표현 및 문자열 변환 과정을 수행합니다.
        # predict_result['ratio'] = ''.join([str(np.around(np.max(predict_value), 4)), '%'])
        # ratio = np.around(predict_result['ratio'], 4)
        # print(ratio)
        cv2.putText(frame, predict_result['category'], (20, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    thickness=2)
        cv2.putText(frame, predict_result['ratio'], (100, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                    thickness=2)
        # 좌측 상단에 FPS 를 출력합니다.
        cv2.putText(frame, str(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
        cv2.imshow('test', frame)
        # "Q"버튼을 누르면 프로세스를 종료합니다.
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            break
