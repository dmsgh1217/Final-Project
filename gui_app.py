import sys
import copy
import time
import pandas as pd
import numpy as np

import cv2
import os

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QMovie #이미지 파일 보여줄 때 사용하는 패키지
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic


# 바탕화면에 실행 버튼 띄우기 위한 클래스 정의
import refactoring_handmouse


class GUI(QtWidgets.QMainWindow):
    def __init__(self, img_path, xy, size=1.0, on_top=True):
        super(GUI, self).__init__()

        # self.timer = QtCore.QTimer(self)
        self.img_path = img_path
        self.xy = xy
        self.size = size
        self.on_top = on_top
        self.localPos = None
        self.setupUi()
        self.show()

    def setupUi(self):
        centralWidget = QtWidgets.QWidget(self) # centralWidget을 정의해야만 만든 widget이 보임
        self.setCentralWidget(centralWidget)

        flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
                                      if self.on_top else QtCore.Qt.FramelessWindowHint)
        # FramelessWindowHint = 닫기 버튼 등 상단에 표시되는 것들이 안보이게 됨
        # WindowStaysOnTopHint = 윈도를 '항상 위에' 노출

        self.setWindowFlags(flags)

        # 투명 배경을 사용하게 위한 Attribute 설정
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        label = QtWidgets.QLabel(centralWidget) #이미지를 올릴 공간 설정은 label 정의
        movie = QMovie(self.img_path) #img_path(이미지 파일)을 QMovie로 불러옴
        label.setMovie(movie) #label에 불러온 moive를 설정
        movie.start() #실제 파일의 너비, 높이를 구하기 위해 한번 실행/정지 시켜줘야함
        movie.stop()

        # 사이즈를 원하는 값에 맞춰 조정하기 위해서 width / height 구함
        w = int(movie.frameRect().size().width() * self.size)
        h = int(movie.frameRect().size().height() * self.size)
        # 지정한 대로 사이즈 적용 시켜줌
        movie.setScaledSize(QtCore.QSize(w, h))
        movie.start()

        self.setGeometry(self.xy[0], self.xy[1], w, h)


def thread_cam():
    # 미디어파이프 라이브러리에서 제공하는 함수를 사용하기 위한 객체를 생성합니다.
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 플래그 값을 초기화 합니다.
    flag = {'momentum': True}
    # 이벤트 값을 저장하는 변수를 초기화합니다.
    event_val = 'default'
    # 랜드마크의 무게 중심 좌표값을 저장하는 변수를 초기화합니다.
    momentum = (0, 0)
    # 외부입력 카메라의 해상도 값을 다음과 같은 크기로 설정합니다.
    cam_width, cam_height = 1440, 810 #FULL HD의 75% 사이즈
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
                        event_val = 'default'
                        break
                if not none_flag:
                    """
                    momentum:       0번, 5번, 17번 랜드마크의 무게 중심 좌표값을 획득합니다.
                    event_val:      
                    """
                    momentum, event_val, switch = refactoring_handmouse.calculate_loc_info(landmarks=segment)
        # 설정된 영상을 출력합니다. (Setup)
        print(f'event_val: {event_val}', switch)
        if flag['momentum']:
            draw_point = tuple([int(momentum[0] * cam_width), int(momentum[1] * cam_height)])
            cv2.circle(img=frame, center=draw_point, radius=3, color=(255, 0, 0), thickness=3)

        cv2.imshow('frame', frame)
        cv2.moveWindow(winname='frame', x=int((screen_width / 2) - (cam_width / 2)), y=int((screen_height / 2) - (cam_height / 2)))
        # "Q"버튼을 누르면 프로세스를 종료합니다.
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()


def execute_event(event):
    import pyautogui
    if event == 'move':
        pyautogui.moveTo()



def main():
    app = QtWidgets.QApplication(sys.argv) #앱 생성

    # 화면 상에 띄워질 이미지 정의

    # 버튼
    Btn1 = GUI('img/1_mouse_btn_img.png', xy=[1630, 370], size=0.15, on_top=True)
    # Btn1_2 = GUI('img/1_mouse_btn_img_2.png', xy=[1760, 380], size=0.1, on_top=True)
    Btn2 = GUI('img/2_painter_btn_img.png', xy=[1630, 470], size=0.15, on_top=True)
    # Btn2_2 = GUI('img/2_painter_btn_img_2.png', xy=[1760, 510], size=0.2, on_top=True)
    Btn3 = GUI('img/3_keyboard_btn_img.png', xy=[1630, 570], size=0.15, on_top=True)
    # Btn3_2 = GUI('img/3_keyboard_btn_img_2.png', xy=[1760, 660], size=0.2, on_top=True)
    Btn4 = GUI('img/4_exit_btn_img_3.png', xy=[1630, 670], size=0.15, on_top=True)
    # Btn4_2 = GUI('img/4_exit_btn_img_2.png', xy=[1760, 810], size=0.2, on_top=True)

    # 외곽선(컨투어)
    Contour1 = GUI('img/green_contour.png', xy=[0, 0], size=1.0, on_top=True)
    # Contour1.hide()
    Contour2 = GUI('img/red_contour.png', xy=[0, 0], size=1.0, on_top=True)
    Contour2.hide()
    Contour3 = GUI('img/yellow_contour_for_control_region.png', xy=[400, 200], size=1.0, on_top=True)
    Contour3.hide()

    Contour4 = GUI('img/grad_contour_for_control_region.png', xy=[400, 200], size=0.7, on_top=True)
    Contour4.hide()
    # Contour4 = GUI('img/test_gif.gif', xy=[400, 200], size=1.0, on_top=True)
    # Contour5 = GUI('img/inner_contour_for_control_region.png', xy=[540, 270], size=1.0, on_top=True)

    ####################################################################################################################
    # 백엔드 모듈을 가져옵니다.
    refactoring_handmouse.initialize()

    import pyautogui
    global screen_width, screen_height
    screen_width, screen_height = pyautogui.size()

    Contour4 = GUI(img_path='./img/grad_contour_for_control_region.png',
                   xy=[(screen_width / 2) - 640, (screen_height / 2) - 360], size=1.0, on_top=True)

    # 외부입력 카메라를 사용하기 위한 멀티스레드를 실행합니다.
    from threading import Thread
    thread_module_cam = Thread(target=thread_cam, name='thread_module_cam')
    thread_module_cam.daemon = True
    thread_module_cam.start()
    ####################################################################################################################

    sys.exit(app.exec_()) #앱 종료

main()


# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv) #앱 생성
#
#     # 화면 상에 띄워질 이미지 정의
#
#     # 버튼
#     Btn1 = GUI('img/1_mouse_btn_img.png', xy=[1760, 380], size=0.2, on_top=True)
#     # Btn1.hide()
#     Btn1_2 = GUI('img/1_mouse_btn_img_2.png', xy=[1760, 380], size=0.2, on_top=True)
#     Btn2 = GUI('img/2_painter_btn_img.png', xy=[1760, 510], size=0.2, on_top=True)
#     # Btn2_2 = GUI('img/2_painter_btn_img_2.png', xy=[1760, 510], size=0.2, on_top=True)
#     Btn3 = GUI('img/3_keyboard_btn_img.png', xy=[1760, 660], size=0.2, on_top=True)
#     # Btn3_2 = GUI('img/3_keyboard_btn_img_2.png', xy=[1760, 660], size=0.2, on_top=True)
#     Btn4 = GUI('img/4_exit_btn_img.png', xy=[1760, 810], size=0.2, on_top=True)
#     # Btn4_2 = GUI('img/4_exit_btn_img_2.png', xy=[1760, 810], size=0.2, on_top=True)
#
#     # 외곽선(컨투어)
#     Contour1 = GUI('img/green_contour.png', xy=[0, 0], size=1.0, on_top=True)
#     # Contour1.hide()
#     Contour2 = GUI('img/red_contour.png', xy=[0, 0], size=1.0, on_top=True)
#     Contour2.hide()
#     Contour3 = GUI('img/yellow_contour_for_control_region.png', xy=[400, 200], size=1.0, on_top=True)
#     Contour3.hide()
#
#     Contour4 = GUI('img/grad_contour_for_control_region.png', xy=[400, 200], size=1.0, on_top=True)
#     # Contour4 = GUI('img/test_gif.gif', xy=[400, 200], size=1.0, on_top=True)
#     # Contour5 = GUI('img/inner_contour_for_control_region.png', xy=[540, 270], size=1.0, on_top=True)
#
#     sys.exit(app.exec_()) #앱 종료
#
#
