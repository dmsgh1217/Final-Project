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
class GUI(QtWidgets.QMainWindow):
    def __init__(self, img_path, xy, size=1.0, on_top=False):
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


def main():
    app = QtWidgets.QApplication(sys.argv) #앱 생성

    # 화면 상에 띄워질 이미지 정의

    # 버튼
    Btn1 = GUI('img/1_mouse_btn_img.png', xy=[1760, 380], size=0.2, on_top=True)
    # Btn1.hide()
    Btn1_2 = GUI('img/1_mouse_btn_img_2.png', xy=[1760, 380], size=0.2, on_top=True)
    Btn2 = GUI('img/2_painter_btn_img.png', xy=[1760, 510], size=0.2, on_top=True)
    # Btn2_2 = GUI('img/2_painter_btn_img_2.png', xy=[1760, 510], size=0.2, on_top=True)
    Btn3 = GUI('img/3_keyboard_btn_img.png', xy=[1760, 660], size=0.2, on_top=True)
    # Btn3_2 = GUI('img/3_keyboard_btn_img_2.png', xy=[1760, 660], size=0.2, on_top=True)
    Btn4 = GUI('img/4_exit_btn_img.png', xy=[1760, 810], size=0.2, on_top=True)
    # Btn4_2 = GUI('img/4_exit_btn_img_2.png', xy=[1760, 810], size=0.2, on_top=True)

    # 외곽선(컨투어)
    Contour1 = GUI('img/green_contour.png', xy=[0, 0], size=1.0, on_top=True)
    # Contour1.hide()
    Contour2 = GUI('img/red_contour.png', xy=[0, 0], size=1.0, on_top=True)
    Contour2.hide()
    Contour3 = GUI('img/yellow_contour_for_control_region.png', xy=[400, 200], size=1.0, on_top=True)
    Contour3.hide()

    Contour4 = GUI('img/grad_contour_for_control_region.png', xy=[400, 200], size=1.0, on_top=True)
    # Contour4 = GUI('img/test_gif.gif', xy=[400, 200], size=1.0, on_top=True)
    # Contour5 = GUI('img/inner_contour_for_control_region.png', xy=[540, 270], size=1.0, on_top=True)

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
