import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from threading import Thread
import prj_function_directory as pfd
import handmouse, subprocess
import cv2, hand_painter
import os
import pyautogui
import sys


draw_point = (0, 0) #중심좌표(momentum) draw point
cam_width, cam_height = 1280, 720
margin = 175
smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, MainWindow, on_top=True):
        super(Ui_MainWindow, self).__init__()
        self.on_top = on_top
        self.MainWindow = MainWindow
        self.localPos = None
        self.status_label = None
        self.setupUi(self.MainWindow)
        # self.keyboard_trigger = False


    def setupUi(self, MainWindow):
        mainframe_info = {'start_x': int(screen_width / 2) - 640, 'start_y': int(screen_height / 2) - 420,
                          'end_x': int(screen_width / 2) + 640, 'end_y': int(screen_height / 2) + 300}

        self.MainWindow.setObjectName("MainWindow")
        self.MainWindow.resize(screen_width, screen_height)
        self.MainWindow.move(0, 0)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        flags = QtCore.Qt.WindowFlags(QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
                                      if self.on_top else QtCore.Qt.FramelessWindowHint)
        # FramelessWindowHint = 닫기 버튼 등 상단에 표시 되는 것들이 안 보이게 됨
        # WindowStaysOnTopHint = 윈도를 '항상 위에' 노출

        self.MainWindow.setWindowFlags(flags)

        # 투명 배경을 사용하게 위한 Attribute 설정
        self.MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        self.centralwidget.setObjectName("centralwidget")
        # self.btn_set('img/1_mouse_btn_img', [int(screen_width / 2) + 740, int(screen_height / 2), 80, 80], 'pushButton')


        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(mainframe_info['start_x'], mainframe_info['start_y'], mainframe_info['end_x'], mainframe_info['end_y']))
        self.label.setPixmap(QtGui.QPixmap("img/grad_contour_for_control_region.png"))
        self.label.setObjectName("label")
        self.status_label = QtWidgets.QLabel(self.centralwidget)
        self.status_label.setGeometry(screen_width - 120, screen_height - 120, 64, 64)
        self.status_label.setPixmap(QtGui.QPixmap('./img/red.png'))
        self.btn_set('img/1_mouse_btn_img', [int(screen_width / 2) + 690, int(screen_height / 2) - 110, 80, 80], 'pushButton')
        self.btn_set_2('img/2_painter_btn_img', [int(screen_width / 2) + 690, int(screen_height / 2) + 10, 80, 80], 'pushButton_2')
        self.btn_set_3('img/3_keyboard_btn_img', [int(screen_width / 2) + 690, int(screen_height / 2) + 130, 80, 80], 'pushButton_3')
        self.btn_set_4('img/4_exit_btn_img', [int(screen_width / 2) + 690, int(screen_height / 2) + 250, 80, 80], 'pushButton_4')
        self.MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton_2.clicked.connect(self.start_paint)
        self.pushButton_3.clicked.connect(self.link_keyboard)
        self.pushButton_4.clicked.connect(self.closeEvent)

    def btn_set(self, path, xy, name):
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(xy[0], xy[1], xy[2], xy[3]))
        # self.pushButton.setGeometry(int(screen_width / 2) + 740, int(screen_height / 2), 80, 80)
        self.pushButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton.setStyleSheet("QPushButton{\n"
                                      "border:0px;\n"
                                      f"border-image: url(./{path}.png);\n"
                                      "padding:7px;\n"
                                      "}\n"
                                      "QPushButton:disabled{\n"
                                      f"border-image: url(./{path}.png);\n"
                                      "}\n"
                                      "QPushButton:hover{\n"
                                      f"border-image: url(./{path}_2.png);\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      f"border-image: url(./{path}_3.png);\n"
                                      "}")
        self.pushButton.setText("")
        self.pushButton.setFlat(True)
        self.pushButton.setObjectName(name)

    def btn_set_2(self, path, xy, name):
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(xy[0], xy[1], xy[2], xy[3]))
        self.pushButton_2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_2.setStyleSheet("QPushButton{\n"
                                        "border:0px;\n"
                                        f"border-image: url(./{path}.png);\n"
                                        "padding:7px;\n"
                                        "}\n"
                                        "QPushButton:disabled{\n"
                                        f"border-image: url(./{path}.png);\n"
                                        "}\n"
                                        "QPushButton:hover{\n"
                                        f"border-image: url(./{path}_2.png);\n"
                                        "}\n"
                                        "QPushButton:pressed{\n"
                                        f"border-image: url(./{path}_3.png);\n"
                                        "}")
        self.pushButton_2.setText("")
        self.pushButton_2.setFlat(True)
        self.pushButton_2.setObjectName(name)

    def btn_set_3(self, path, xy, name):
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(xy[0], xy[1], xy[2], xy[3]))
        self.pushButton_3.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_3.setStyleSheet("QPushButton{\n"
                                        "border:0px;\n"
                                        f"border-image: url(./{path}.png);\n"
                                        "padding:7px;\n"
                                        "}\n"
                                        "QPushButton:disabled{\n"
                                        f"border-image: url(./{path}.png);\n"
                                        "}\n"
                                        "QPushButton:hover{\n"
                                        f"border-image: url(./{path}_2.png);\n"
                                        "}\n"
                                        "QPushButton:pressed{\n"
                                        f"border-image: url(./{path}_3.png);\n"
                                        "}")
        self.pushButton_3.setText("")
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName(name)

    def btn_set_4(self, path, xy, name):
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(xy[0], xy[1], xy[2], xy[3]))
        self.pushButton_4.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_4.setStyleSheet("QPushButton{\n"
                                        "border:0px;\n"
                                        f"border-image: url(./{path}.png);\n"
                                        "padding:7px;\n"
                                        "}\n"
                                        "QPushButton:disabled{\n"
                                        f"border-image: url(./{path}.png);\n"
                                        "}\n"
                                        "QPushButton:hover{\n"
                                        f"border-image: url(./{path}_2.png);\n"
                                        "}\n"
                                        "QPushButton:pressed{\n"
                                        f"border-image: url(./{path}_3.png);\n"
                                        "}")
        self.pushButton_4.setText("")
        self.pushButton_4.setFlat(True)
        self.pushButton_4.setObjectName(name)

    def start_paint(self):
        # th_painter = Thread(target=hand_painter.main, args=(param=event_val,), name='hand_painter')
        # th_painter.daemon = True
        # th_painter.start()
        th_call_painter = Thread(target=thread_call_painter, name='th_call_painter')
        th_call_painter.daemon = True
        th_call_painter.start()

    def link_keyboard(self):
        print('input keyboard')
        subprocess.Popen('C:\WINDOWS\system32\osk.exe')

    def activate_mouse(self):
        print('mouse activate')

    def closeEvent(self, QCloseEvent):
        QCloseEvent.accept()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

    def widget_controller(self, show):
        if show:
            self.MainWindow.show()
        else:
            self.MainWindow.hide()

    def show_status(self, val):
        if val == 2:
            self.status_label.setPixmap(QtGui.QPixmap('./img/green.png'))
        elif val == 1:
            self.status_label.setPixmap(QtGui.QPixmap('./img/yellow.png'))
        else:
            self.status_label.setPixmap(QtGui.QPixmap('./img/red.png'))


def thread_cam():
    global execute_flag, execute_parameter, draw_point, clocX, clocY, plocX, plocY, execute_flag, event_val
    # 미디어파이프 라이브러리에서 제공하는 함수를 사용하기 위한 객체를 생성합니다.
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    # mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 플래그 값을 초기화 합니다.
    flag = {'momentum': True}
    # 이벤트 값을 저장하는 변수를 초기화합니다.
    event_val = 'default'
    # 랜드마크의 무게 중심 좌표값을 저장하는 변수를 초기화합니다.
    momentum = [.0, .0]
    # 외부입력 카메라의 해상도 값을 다음과 같은 크기로 설정합니다.
    cam_width, cam_height = 1280, 720
    # OpenCV 라이브러리를 이용하여 비디오 객체를 생성합니다.
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # 비디오 객체로부터 리턴 값(ret)과 영상(frame)을 가져옵니다.
        ret, frame = cap.read()
        # 영상을 좌우 반전하여 출력합니다.
        frame = cv2.flip(frame, 1)
        # 영상의 출력되는 사이즈를 조절합니다.
        frame = cv2.resize(frame, (cam_width, cam_height), interpolation=cv2.INTER_LINEAR)
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
                        ui.show_status(val=0)
                        break
                if not none_flag:
                    """
                    momentum:       0번, 5번, 17번 랜드마크의 무게 중심 좌표값을 획득합니다.
                    event_val:
                    """
                    if not run_flag:
                        momentum, event_val, switch = handmouse.calculate_loc_info(landmarks=segment)
                        if switch:
                            ui.show_status(val=1)
                        else:
                            ui.show_status(val=2)
                        print(f'event_val({switch}): {event_val}')
                        execute_parameter = [momentum, event_val, screen_width, screen_height]
                        execute_flag = True

                    # tf_result = icon_in(momentum[0], momentum[1], 1630, 370)
                    # icon_control(tf_result, event_val)

        # 설정된 영상을 출력합니다. (Setup)
        if flag['momentum']:
            draw_point = tuple([int(momentum[0] * cam_width), int(momentum[1] * cam_height)])
            cv2.circle(img=frame, center=draw_point, radius=3, color=(255, 0, 0), thickness=3)

        cv2.rectangle(frame, (margin, margin), (cam_width - margin, cam_height - margin), (255, 0, 255), 2)

        cv2.imshow('frame', frame)
        cv2.moveWindow(winname='frame', x=int((screen_width / 2) - (cam_width / 2)), y=int((screen_height / 2) - (cam_height / 2)))

        # "Q"버튼을 누르면 프로세스를 종료합니다.
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            break
    cv2.destroyAllWindows()


plocX, plocY = clocX, clocY


def thread_call_painter():
    while True:
        hand_painter.main(param=event_val)
        time.sleep(0.05)


def thread_execute_event():
    global execute_flag, plocX, plocY, clocX, clocY, run_flag
    scroll_asset = {'previous': -1, 'current': -1}
    while True:
        if execute_flag:
            execute_flag = False
            run_flag = True
            event = execute_parameter[1]
            # 좌표 변환
            xy = (draw_point[0], draw_point[1])
            win_xy = [execute_parameter[2], execute_parameter[3]]
            cam_xy = [cam_width, cam_height]
            loc_x, loc_y = pfd.convert_loc(xy, win_xy, cam_xy, margin)
            scroll_asset['current'] = loc_y

            clocX = plocX + (loc_x - plocX) / smoothening
            clocY = plocY + (loc_y - plocY) / smoothening

            try:
                if event == 'move' or event == 'default':
                    pfd.move_event(clocX, clocY)
                    pfd.drag_event(drag_flag=False)
                elif event == 'leftclick':
                    pfd.leftclick_event(clocX, clocY)
                elif event == 'doubleclick':
                    pfd.doubleclick_event(clocX, clocY)
                elif event == 'drag':
                    pfd.drag_event(drag_flag=True)
                    # pfd.move_event(clocX, clocY)
                elif event == 'rightclick':
                    pfd.rightclick_event(clocX, clocY)
                elif event == 'screenshot':
                    ui.widget_controller(show=False)
                    time.sleep(0.5)
                    pfd.screenshot_event()
                    time.sleep(0.5)
                    ui.widget_controller(show=True)
                elif event == 'scroll':
                    if not scroll_asset['previous'] == -1:
                        scroll_vector = (scroll_asset['current'] - scroll_asset['previous']) / 5
                        pfd.scroll_event(vector=scroll_vector)
                    scroll_asset['previous'] = scroll_asset['current']
                else:
                    pass
            except Exception as E:
                print(f'Error occurred.. {E}')
            finally:
                run_flag = False
        plocX, plocY = clocX, clocY
        time.sleep(0.01)


if __name__ == "__main__":
    screen_width, screen_height = pyautogui.size()
    handmouse.initialize()

    # 스크린샷을 저장할 디렉토리를 생성합니다. (이미 있는경우 무시합니다.)
    os.makedirs('./screenshot_img', exist_ok=True)

    # th_execute_event 에서 사용할 플래그 변수, 파라미터를 선언합니다.
    event_val = 'default'
    execute_flag = False
    run_flag = False
    execute_parameter = []

    th_module_cam = Thread(target=thread_cam, name='th_module_cam')
    th_module_cam.daemon = True
    th_module_cam.start()
    th_execute_event = Thread(target=thread_execute_event, name='th_execute_event')
    th_execute_event.daemon = True
    th_execute_event.start()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
