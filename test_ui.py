# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1495, 782)
        # window = QtWidgets.QWidget()
        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1360, 380, 100, 100))
        self.pushButton_2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_2.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"border-image: url(:/painter/2_painter_btn_img.png);\n"
"padding:7px;\n"
"}\n"
"QPushButton:disabled{\n"
"border-image: url(:/painter/2_painter_btn_img.png);\n"
"}\n"
"QPushButton:hover{\n"
"border-image: url(:/painter/2_painter_btn_img_2.png);\n"
"}\n"
"QPushButton:pressed{\n"
"border-image: url(:/painter/2_painter_btn_img_3.png);\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"")
        self.pushButton_2.setText("")
        self.pushButton_2.setFlat(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1360, 640, 100, 100))
        self.pushButton_4.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_4.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"border-image: url(:/exit/4_exit_btn_img.png);\n"
"padding:7px;\n"
"}\n"
"QPushButton:disabled{\n"
"border-image: url(:/exit/4_exit_btn_img.png);\n"
"}\n"
"QPushButton:hover{\n"
"border-image:url(:/exit/4_exit_btn_img_2.png);\n"
"}\n"
"QPushButton:pressed{\n"
"border-image:url(:/exit/4_exit_btn_img_3.png);\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"\n"
"")
        self.pushButton_4.setText("")
        self.pushButton_4.setFlat(True)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1360, 250, 100, 100))
        self.pushButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"border-image: url(:/mouse_click/1_mouse_btn_img.png);\n"
"padding:7px;\n"
"}\n"
"QPushButton:disabled{\n"
"border-image: url(:/mouse_click/1_mouse_btn_img.png);\n"
"}\n"
"QPushButton:hover{\n"
"border-image: url(:/mouse_click/1_mouse_btn_img_2.png);\n"
"}\n"
"QPushButton:pressed{\n"
"border-image: url(:/mouse_click/1_mouse_btn_img_3.png);\n"
"}\n"
"")
        self.pushButton.setText("")
        self.pushButton.setFlat(True)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1360, 510, 100, 100))
        self.pushButton_3.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_3.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"border-image: url(:/keyboard/3_keyboard_btn_img.png);\n"
"padding:7px;\n"
"}\n"
"QPushButton:disabled{\n"
"border-image: url(:/keyboard/3_keyboard_btn_img.png);\n"
"}\n"
"QPushButton:hover{\n"
"border-image:url(:/keyboard/3_keyboard_btn_img_2.png);\n"
"}\n"
"QPushButton:pressed{\n"
"border-image: url(:/keyboard/3_keyboard_btn_img_3.png);\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"")
        self.pushButton_3.setText("")
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 40, 1291, 721))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("img/grad_contour_for_control_region.png"))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

