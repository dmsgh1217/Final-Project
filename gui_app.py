import sys
import copy
import time
import pandas as pd
import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic


# 사용자 정의 시그널 사용을 위한 클래스 정의
#
# class CustomSignal(QObject):
#       signal = pyqtSignal()

# 바탕화면에 실행 버튼 띄우기 위한 클래스 정의

class Button(QtWidgets.QMainWindow):
    def __init__(self, img_path, xy, size=1.0, on_top=False):
        super(Button, self).__init__()
        # click_event = pyqtSignal()

        # self.timer = QtCore.QTimer(self)
        self.img_path = img_path
        self.xy = xy
        self.from_xy = xy
        self.from_xy_diff = [0, 0]
        self.to_xy = xy
        self.to_xy_diff = [0, 0]
        self.speed = 60
        self.direction = [0, 0] # x: 0(left), 1(right), y: 0(up), 1(down)
        self.size = size
        self.on_top = on_top
        self.localPos = None
        self.setupUi()
        self.show()



    # button class 안에 별도로 시그널을 만들어주어 이벤트 처리
    #
    # def hamsu(self):
    #     Btn1 = self.sender()
    #     Btn1.hide()


    def setupUi(self):
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint if self.on_top else QtCore.Qt.FramelessWindowHint)
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        label = QtWidgets.QLabel(centralWidget)
        movie = QMovie(self.img_path)
        label.setMovie(movie)
        movie.start()
        movie.stop()

        w = int(movie.frameRect().size().width() * self.size)
        h = int(movie.frameRect().size().height() * self.size)
        movie.setScaledSize(QtCore.QSize(w, h))
        movie.start()

        self.setGeometry(self.xy[0], self.xy[1], w, h)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    #Btn1 = Button('Button_img/1_mouse_btn_img.png', xy=[1760, 380], size=0.2, on_top=True)
    # Btn1.hide()
    Btn1_2 = Button('Button_img/1_mouse_btn_img_2.png', xy=[1760, 380], size=0.2, on_top=True)
    Btn2 = Button('Button_img/2_painter_btn_img.png', xy=[1760, 510], size=0.2, on_top=True)
    # Btn2_2 = Button('Button_img/2_painter_btn_img_2.png', xy=[1760, 510], size=0.2, on_top=True)
    Btn3 = Button('Button_img/3_keyboard_btn_img.png', xy=[1760, 660], size=0.2, on_top=True)
    # Btn3_2 = Button('Button_img/3_keyboard_btn_img_2.png', xy=[1760, 660], size=0.2, on_top=True)
    Btn4 = Button('Button_img/4_exit_btn_img.png', xy=[1760, 810], size=0.2, on_top=True)
    # Btn4_2 = Button('Button_img/4_exit_btn_img_2.png', xy=[1760, 810], size=0.2, on_top=True)

    Contour1 = Button('Button_img/green_contour.png', xy=[0, 0], size=1.0, on_top=True)
    Contour1.hide()
    Contour2 = Button('Button_img/red_contour.png', xy=[0, 0], size=1.0, on_top=True)
    Contour2.hide()
    Contour3 = Button('Button_img/yellow_contour_for_control_region.png', xy=[400, 200], size=1.0, on_top=True)
    Contour3.hide()
    Contour4 = Button('Button_img/grad_contour_for_control_region.png', xy=[400, 200], size=1.0, on_top=True)
    Contour5 = Button('Button_img/inner_contour_for_control_region.png', xy=[540, 270], size=1.0, on_top=True)
    sys.exit(app.exec_())

