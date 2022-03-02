# Release 1.2 by Min-chul
# cv2 객체를 관리하는 함수(draw_frame) 추가 - 더 이상 메인(main)에서 관리하지 않음.
# 마우스 이벤트를 관리하는 함수(execute_mouse_event) 추가

# Release 1.1 by Min-chul
# 멀티스레드 실행하는 코드를 함수화
# 42차원의 좌표(Location) 데이터와 좌표 데이터를 기반으로 산출한 14차원의 각도(Angle) 데이터를 병합하는 알고리즘 추가
# pyautogui 라이브러리 제거 -> GUI 스크립트에서 사용 예정

from tensorflow.keras.models import load_model
from threading import Thread
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import prj_function_directory as pfd
import pyautogui


def initialize_thread():
    # 인공지능 모델로 입력(좌표 + 각도)데이터를 주고 출력(분류된 카테고리) 결과를 획득하는 멀티스레드를 실행합니다.
    try:
        thread_predict_motion = Thread(target=predict_motion, name='predict_motion')
        thread_predict_motion.daemon = True
        thread_predict_motion.start()
    except Exception as E:
        print(f'Cannot launched "{thread_predict_motion.name}" thread..\nError: {E}')
        exit()
    else:
        print(f'"{thread_predict_motion.name}" thread start.')

    # 좌클릭 이벤트 처리를 위한 멀티스레드를 실행합니다.
    try:
        thread_click_trigger = Thread(target=click_trigger, name='click_trigger')
        thread_click_trigger.daemon = True
        thread_click_trigger.start()
    except Exception as E:
        print(f'Cannot launched "{thread_click_trigger.name}" thread..\nError: {E}')
        exit()
    else:
        print(f'"{thread_click_trigger.name}" thread start.')

    # 마우스 이벤트 처리를 위한 멀티스레드를 실행합니다.
    try:
        thread_execute_mouse_event = Thread(target=execute_mouse_event, name='execute_mouse_event')
        thread_execute_mouse_event.daemon = True
        thread_execute_mouse_event.start()
    except Exception as E:
        print(f'Cannot launched "{thread_execute_mouse_event.name}" thread..\nError: {E}')
        exit()
    else:
        print(f'"{thread_execute_mouse_event.name}" thread start.')


# 멀티스레드 - 1
# 영상으로부터 획득한 랜드마크의 좌표 또는 각도 및 비율 데이터를 이용하여 인공지능 모델로부터 출력값을 획득하는 스레드입니다.
def predict_motion():
    global PREDICT_FLAG, CLICK_FLAG, result_asset, event_asset
    while True:
        if PREDICT_FLAG:
            PREDICT_FLAG = False
            # 인공지능 모델에 "segment"변수를 입력으로 주어 분류 결과를 획득합니다.
            # 영상에서 비 정상적인 랜드마크가 수집되는 경향이 있어 예외처리를 하여 시스템 다운을 방지합니다.
            try:
                model_predict = model.predict(segment)
            except ValueError:
                print(f'An invalid input shape{segment.shape} was detected..')
            except Exception as Err:
                print(f'Unknown error occurred.. {Err}')
            else:
                # 학습된 인공지능 모델이 분류한 카테고리 결과 값을 변수에 저장합니다.
                result_asset['category'] = label[np.argmax(model_predict)]
                # 분류된 카테고리의 확률 값을 변수에 저장합니다.
                result_asset['ratio'] = np.max(model_predict)
                # 확률 값이 sensitivity(0.99) 이상인 경우에만 최종 이벤트로서 해당 카테고리를 사용합니다.
                if result_asset['ratio'] >= event_asset['sensitivity']:
                    event_asset['event'] = result_asset['category']
                else:
                    event_asset['event'] = 'default'

                # 최종 이벤트가 "leftclick"이면 클릭 이벤트 처리를 위한 멀티스레드를 활성화합니다.
                if event_asset['event'] == 'leftclick' and not CLICK_FLAG['run']:
                    CLICK_FLAG['detect'] = True
        time.sleep(0.01)


# 멀티스레드 - 2
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
            current_status = 'leftclick'
            # 동작 인식 제한시간을 측정하기 위한 시간 측정을 시작합니다.
            s_time = time.time()
            # 동작 인식 제한시간(0.5초)동안 반복하여 상태 변화가 얼마나 발생했는지 계산하는 반복문을 시행합니다.
            while time.time() - s_time < click_interval:
                # 실시간으로 분류하고 있는 모델의 결과가 스레드 내에서 선언한 현재 상태와 다를 경우, 상태의 변화가 발생하였다고 처리합니다.
                if event_asset['event'] != current_status:
                    # 상태가 변화해도 "Move" 또는 "Click"상태로 변화하였을 때만 상태변화로 처리합니다.
                    if event_asset['event'] == 'leftclick' or event_asset['event'] == 'move':
                        # 상태가 변화하였으므로, 현재 상태값을 모델이 분류한 결과값으로 대체합니다.
                        current_status = event_asset['event']
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
                event_asset['event'] = 'drag'
                pass
            # 상태 변화 횟수가 1이면 제한 시간내에 "Move"로 변화하였으므로 "Click"이벤트로 연결합니다.
            elif status_count == 1:
                print(f'Click')
                # pyautogui.() 여기서 실행을 할거냐?
                # GUI 임포트해서 특정 함수에 파라미터를 전달해서 GUI에서 마우스 컨트롤을 하게 한다.
                pass
            # 상태 변화 횟수가 3회 이상이면, 제한 시간내에 "Move" -> "Click" -> "Move"로 변화하였으므로 "Double-Click"이벤트로 연결합니다.
            elif status_count >= 3:
                print(f'Double Click')
                pass

            # 스레드 내부 함수 실행이 종료되었으므로, 부분 동작 처리를 활성화 할 수 있도록 CLICK_FLAG['detect']를 "False"로 변경합니다.
            CLICK_FLAG['detect'] = False
            # 스레드 내부 함수 실행이 종료되었으므로, CLICK_FLAG['run'] 변수를 "False"로 변경하여 비활성화 상태임을 명시합니다.
            CLICK_FLAG['run'] = False
        time.sleep(0.01)


# 멀티스레드 - 3
def execute_mouse_event():
    while True:
        # print(f'event: {event_asset["event"]}')
        pointer = tuple([int(momentum_position[0] * screen_width), int(momentum_position[1] * screen_height)])
        pyautogui.moveTo(pointer)
        time.sleep(0.01)


def draw_frame(pointer):
    # 초당 프레임(FPS)수를 계산합니다.
    fps = str(int(1. / (time.time() - start_time)))
    # 좌측 상단에 FPS 를 출력합니다.
    cv2.putText(img=frame, text=fps, org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0),
                thickness=2)
    # 인공지능 모델의 분류 결과를 우측 하단에 표시합니다.
    cv2.putText(img=frame, text=event_asset['event'], org=(20, 670), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(255, 255, 255), thickness=2)
    # 인공지능 모델이 분류한 결과의 확률 값을 우측 하단에 표시할 수 있도록 데이터 타입을 변환하고, 문자열을 추가합니다.
    classification_ratio = ''.join([str(np.around(np.max(result_asset["ratio"]), 4)), '%'])
    # 분류 결과의 확률 값이 0.9 이상이면 녹색으로 우측 하단에 표시합니다.
    if result_asset['ratio'] >= 0.9:
        cv2.putText(img=frame, text=classification_ratio, org=(20, 700), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=(0, 255, 0), thickness=2)
    # 분류 결과의 확률 값이 0.9 미만에서 0.8 이상인 경우 노란색으로 우측 하단에 표시합니다.
    elif 0.8 <= result_asset['ratio'] < 0.9:
        cv2.putText(img=frame, text=classification_ratio, org=(20, 700), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=(0, 255, 255), thickness=2)
    # 분류 결과의 확률 값이 0.8 미만이면 빨간색으로 우측 하단에 표시합니다.
    else:
        cv2.putText(img=frame, text=classification_ratio, org=(20, 700), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=(0, 0, 255), thickness=2)

    # 서브 프레임(Sub-frame)을 추가합니다. 서브프레임은 랜드마크의 정상 검출 유무에 따라 프레임의 색상이 변경됩니다.
    if detect_landmark_flag:
        cv2.rectangle(img=frame, pt1=(128, 72), pt2=(1152, 648), color=(0, 255, 0), thickness=2)
        # 메인 프레임에 중심 좌표(momentum_position)를 원(Circle)형으로 출력합니다.
        try:
            draw_pointer = tuple([int(pointer[0] * cam_width), int(pointer[1] * cam_height)])
            cv2.circle(img=frame, center=draw_pointer, radius=3, color=(255, 0, 0), thickness=3)
        except:
            print(f'error')
    else:
        cv2.rectangle(img=frame, pt1=(128, 72), pt2=(1152, 648), color=(0, 0, 255), thickness=2)

    # 설정된 영상을 출력합니다. (Setup)
    cv2.imshow('frame', frame)
    # "Q"버튼을 누르면 프로세스를 종료합니다.
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()


if __name__ == '__main__':
    # 학습된 인공지능 모델을 불러옵니다.
    model = load_model(filepath='./models/model_seq5_loc_angle(default).h5')
    # 카테고리와 관련된 인코더 정보를 불러옵니다.
    with open(file='./resources/encoder_loc_angle_data_lbl_d56.pickle', mode='rb') as f:
        encoder = pickle.load(f)
        label = encoder.classes_
    # 미디어파이프 라이브러리에서 제공하는 함수를 사용하기 위한 객체를 생성합니다.
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 외부입력 카메라의 해상도 값을 다음과 같은 크기로 설정합니다.
    cam_width, cam_height = 1280, 720
    # 사용자 모니터의 실제 해상도 값을 획득합니다.
    screen_width, screen_height = pyautogui.size()

    # 학습된 인공지능 모델이 분류한 카테고리와 분류 확률 값을 관리하는 변수(predict_result)를 초기화합니다.
    """
    category:   한 개 프레임(frame)을 분류한 카테고리 값을 저장합니다.
    ratio:      한 개 프레임(frame)을 분류한 확률 값을 저장합니다.   
    """
    result_asset = {'category': 'default', 'ratio': np.float(0.0), 'event': 'default', 'event_switch': False}
    # 이벤트 처리를 종합적으로 관리하는 변수(event_asset)를 초기화합니다.
    """
    event:          연속적인 프레임(현재 3개)을 취합하여 최종적으로 실행하고자 하는 이벤트 값을 정의합니다.
    switch:         일시정지(Pause) 동작에 의한 이벤트 처리 유무를 관리하는 값을 정의합니다. (False: 실행, True: 실행하지 않음)
    sensitivity:    인공지능 모델이 분류한 카테고리의 확률값이 일정 수치 이상에서만 이벤트로 처리하도록 민감도(Sensitivity)를 정의합니다.
    """
    event_asset = {'event': 'default', 'switch': False, 'sensitivity': 0.99}

    # 클릭 이벤트, 더블 클릭 이벤트, 드래그 이벤트에 사용할 변수를 초기화합니다.
    click_interval = 0.5
    click_count = 0

    # 랜드마크 정보를 이용하여 마우스 포인터 이동을 하고자 하는 기준좌표 정보를 갖고있는 변수를 선언합니다.
    momentum_position = [0.1, 0.1]

    # 랜드마크가 메인프레임(frame) 안에서 감지 되었는지를 확인하는 플래그 변수를 선언합니다. (False: 비정상 감지, True: 정상 감지)
    detect_landmark_flag = False

    # 멀티 스레드에서 사용하는 변수를 선언합니다.
    PREDICT_FLAG = False
    CLICK_FLAG = {'detect': False, 'run': False}

    # 멀티 스레드를 관리하는 함수를 호출합니다.
    initialize_thread()

    # OpenCV 라이브러리를 이용하여 비디오 객체를 생성합니다.
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # 초당 프레임(FPS) 계산을 위해 현재 시간을 획득합니다.
        start_time = time.time()
        # 비디오 객체로부터 리턴 값(ret)과 영상(frame)을 가져옵니다.
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
                    # 비정상적인 랜드마크 좌표값이 확인되었을 경우에는 이후 프로세스를 진행하지 않고 다음 프레임(frame)을 분석합니다.
                    if none_flag:
                        break

                # 각 랜드마크의 좌표값이 정상인 경우에만 이하 프로세스를 실행합니다.
                if not none_flag:
                    # 42차원의 좌표 데이터를 각도 변환 함수(convert_angle())의 입력에 사용할 수 있도록 리스트(loc_x, loc_y)로 재구성합니다.
                    loc_x, loc_y = [], []
                    for idx in range(len(segment)):
                        loc_x.append(segment[idx][0])
                        loc_y.append(segment[idx][1])

                    # 재구성된 리스트(loc_x, loc_y)를 이용하여 14차원의 각도(Angle)로 변환해주는 함수(convert_angle())를 호출합니다.
                    angle_segment = np.array(pfd.convert_angle(x=loc_x, y=loc_y))
                    # 기존에 구성된 42차원의 좌표데이터를 갖고있는 변수(segment)를 1차원 데이터로 변형합니다.
                    segment = segment.flatten()
                    # 변수(segment)를 42차원의 좌표 데이터와 14차원의 각도 데이터를 병합한 형태로 구성합니다.
                    segment = np.concatenate([segment, angle_segment])
                    # 인공지능 모델의 입력 형태에 맞추기 위해 변수(segment)의 차원을 (1, n)으로 변형합니다.
                    segment = segment.reshape(-1, segment.shape[0])
                    # 랜드마크를 이용한 중심좌표(momentum_position)를 산출합니다.
                    momentum_position = pfd.reference_loc(x=loc_x, y=loc_y)
                    # 멀티스레드의 동작을 활성화 하기 위해 플래그를 설정합니다.
                    PREDICT_FLAG = True
                    detect_landmark_flag = True
                else:
                    detect_landmark_flag = False

        # 영상 위젯(frame)에 다양한 정보를 노출시키는 함수를 실행합니다.
        draw_frame(pointer=momentum_position)
