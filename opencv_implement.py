# -*- coding: utf-8 -*-

import sys
from opencv_implement_ui import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication

import numpy as np


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)

    # button for problem 1.1
    def on_btn1_1_click(self):
        #load image
        img = cv2.imread('images/dog.bmp')

        #print image size
        print('Height: ', img.shape[0])
        print('Width: ', img.shape[1])

        #show image
        cv2.imshow('ShowImage', img)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn1_2_click(self):
        #load image
        img = cv2.imread('images/color.png')

        #color conversion
        b, g, r = cv2.split(img)
        img_cc = cv2.merge((g,r,b))

        #show image
        cv2.imshow('ShowImage', img)
        cv2.imshow('ShowImage(Converted)', img_cc)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn1_3_click(self):
        #load image
        img = cv2.imread('images/dog.bmp')

        #image flip(0 = vertical, 1 = horizontal, -1 = both)
        img_flip = cv2.flip(img, 1)

        #show image
        cv2.imshow('ShowImage', img)
        cv2.imshow('ShowImage(Flipped)', img_flip)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn1_4_click(self):
        #load image
        img = cv2.imread('images/dog.bmp')
        img_flip = cv2.flip(img, 1)

        #Blending
        def blend(x):
            blend_value = cv2.getTrackbarPos("Blend", "ShowImage")
            img_blend = cv2.addWeighted(img, (100 - blend_value)/100, img_flip, blend_value/100, 0);
            cv2.imshow("ShowImage", img_blend)

        #New window
        cv2.namedWindow("ShowImage")
        cv2.createTrackbar("Blend", "ShowImage", 0, 100, blend)
        cv2.setTrackbarPos("Blend", "ShowImage", 50)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn2_1_click(self):
        #load image and grayscale
        img = cv2.imread("images/M8.jpg", 0)

        #Gaussian smooth 3*3
        kernel_size = (3, 3)
        img = cv2.GaussianBlur(img, kernel_size, 0)
        cv2.imshow("ShowImage(Smooth)", img)

        #Sobel edge detection
        dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        row = img.shape[0] - 2
        col = img.shape[1] - 2

        img_verti = np.zeros((row, col))
        img_hori = np.zeros((row, col))
        img_megni = np.zeros((row, col))
        img_angle = np.zeros((row, col))

        for i in range(row):
            for j in range(col):
                mask = img[i:i+3, j:j+3]
                img_verti[i, j] = np.sum(np.multiply(mask, dx))
                img_hori[i, j] = np.sum(np.multiply(mask, dy))
                img_megni[i, j] = np.sqrt(np.square(img_verti[i, j]) + np.square(img_hori[i, j]))
                img_angle[i, j] = np.arctan2(img_verti[i, j], img_hori[i, j])

        #nomalize to 0~255
        verti_abs = cv2.convertScaleAbs(img_verti)
        hori_abs = cv2.convertScaleAbs(img_hori)
        megni_abs = cv2.convertScaleAbs(img_megni)

        img_angle = np.rad2deg(img_angle)
        img_angle = (img_angle < 0)*360 + img_angle

        #threshold set 40
        ret, verti_out = cv2.threshold(verti_abs, 40, 255, cv2.THRESH_TOZERO)
        ret, hori_out = cv2.threshold(hori_abs, 40, 255, cv2.THRESH_TOZERO)
        ret, megni_out = cv2.threshold(hori_abs, 40, 255, cv2.THRESH_TOZERO)

        #trackbar func for megnitude
        def megni_trackbar(x):
            megni_val = cv2.getTrackbarPos("Megnitude", "Megnitude")
            ret, megni_out = cv2.threshold(megni_abs, megni_val, 255, cv2.THRESH_TOZERO)
            cv2.imshow("Megnitude", megni_out)

        #trackbar func for angle
        def angle_trackbar(x):
            angle_val = cv2.getTrackbarPos("Angle", "Direction")
            angle_out = (img_angle > angle_val - 10) * (img_angle < angle_val + 10) * megni_out
            angle_out = cv2.convertScaleAbs(angle_out)

            cv2.imshow("Direction", angle_out)

        #show image
        cv2.imshow("Vertical", verti_out)
        cv2.imshow("Horizontal", hori_out)
        cv2.namedWindow("Megnitude")
        cv2.namedWindow("Direction")
        cv2.createTrackbar("Megnitude", "Megnitude", 0, 255, megni_trackbar)
        cv2.createTrackbar("Angle", "Direction", 0, 360, angle_trackbar)

        #initial TrackbarPos
        cv2.setTrackbarPos("Megnitude", "Megnitude", 40)
        cv2.setTrackbarPos("Angle", "Direction", 10)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn3_1_click(self):
        #Gaussian pyramids
        gp_0 = cv2.imread("images/pyramids_Gray.jpg")
        gp_1 = cv2.pyrDown(gp_0)
        gp_2 = cv2.pyrDown(gp_1)

        #Laplacian pyramids
        lp_0 = cv2.subtract(gp_0, cv2.pyrUp(gp_1))
        lp_1 = cv2.subtract(gp_1, cv2.pyrUp(gp_2))

        #Inverse pyramids
        Inv_1 = cv2.add(lp_1, cv2.pyrUp(gp_2))
        Inv_0 = cv2.add(lp_0, cv2.pyrUp(Inv_1))

        #show image
        cv2.imshow("Gaussian pyramids lv1", gp_1)
        cv2.imshow("Laplacian pyramids lv0", lp_0)
        cv2.imshow("Inverse pyramids lv1", Inv_1)
        cv2.imshow("Inverse pyramids lv0", Inv_0)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn4_1_click(self):
        #Global Threshold
        #load image
        img = cv2.imread("images/QR.png", 0)

        ret, img_out = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

        #show image
        cv2.imshow("Original image", img)
        cv2.imshow("Threshold image", img_out)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn4_2_click(self):
        #Local Threshold
        #load image
        img = cv2.imread("images/QR.png", 0)

        img_out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -1)
        
        #show image
        cv2.imshow("Original image", img)
        cv2.imshow("Threshold image", img_out)        

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    def on_btn5_1_click(self):
        # Rotate + Scale + Translation
        #load image
        img = cv2.imread("images/OriginalTransform.png")

        #get value from textBox
        if self.edtAngle.text() == "" or self.edtScale.text() == "" or self.edtTx.text() == "" or self.edtTy.text() == "":
            angle = 0
            scale = 1
            Tx = 0
            Ty = 0
        else:
            angle = float(self.edtAngle.text())
            scale = float(self.edtScale.text())
            Tx = float(self.edtTx.text())
            Ty = float(self.edtTy.text())

        #Transform
        M = np.float32([[1, 0, Tx], [0, 1, Ty]])
        img_out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        #Rotate + Scale
        center = (130 + Tx, 125 + Ty)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        img_out = cv2.warpAffine(img_out, M, (img.shape[1], img.shape[0]))

        #show image
        cv2.imshow("Original Image", img)
        cv2.imshow("Rotate + Scale + Translation Image", img_out)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()



    def on_btn5_2_click(self):
        #Perspective Transformation 
        #load image
        img = cv2.imread("images/OriginalPerspective.png")

        global num, pt_start
        pt_start = np.zeros((4, 2), np.float32)
        num = 0
        pt_end = np.float32([[20,20], [450, 20], [450, 450], [20, 450]])


        #mouse callback func
        def mouse_func(event, x, y, flags, param):
            global num, pt_start
            if event == cv2.EVENT_LBUTTONUP and num < 4:
                pt_start[num] = [x, y]
                num = num+1
                print(num)
            elif num >= 4:
                #Perspective transform
                M = cv2.getPerspectiveTransform(pt_start, pt_end)
                img_out = cv2.warpPerspective(img, M, (450, 450))
                cv2.imshow("Perspective Result Image", img_out)

        #new Windows
        cv2.namedWindow("Original Image")
        #show image
        cv2.imshow("Original Image", img)
        #mouse callback
        cv2.setMouseCallback("Original Image", mouse_func)

        #destroy
        cv2.waitKey()
        cv2.destroyAllWindows()

    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
