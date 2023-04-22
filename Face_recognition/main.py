from PyQt5 import QtCore, QtGui, QtWidgets
from gui import Ui_MainWindow
from scipy import ndimage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import qimage2ndarray
import ntpath
import copy
import sys
import cv2
import numpy as np
import imageio
import os.path
from PIL import Image, ImageDraw
import time
import timeit
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import face_detection

class App(QtWidgets.QMainWindow):
    
    path1 = ""
    path2 = ""

    
    
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.b3.clicked.connect(self.loaddata2)
        self.ui.b3_2.clicked.connect(self.loaddata1)
        self.ui.b4_2.clicked.connect(self.detect_faces)


    def loaddata1(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path1 = filename[0]
        
        image = cv2.imread(self.path1)
        new = np.flip(image, axis=-1) 
        self.ui.w3_2.show()
        self.ui.w3_2.setImage(np.rot90(new,1))

    def loaddata2(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path2 = filename[0]
        
        image = cv2.imread(self.path2)
        new = np.flip(image, axis=-1) 
        self.ui.w3.show()
        self.ui.w3.setImage(np.rot90(new,1))      

    def detect_faces(self):
        im = cv2.imread(self.path1)
        start_time = timeit.default_timer()
        face_detection.apply_detection(im)
        # print("detect: "+str(num_of_faces)+" faces") 
        end_time = timeit.default_timer()
        elapsed_time = format(end_time - start_time, '.5f')
        img = cv2.imread("detected.jpg")
        self.ui.label_4.setText(str(elapsed_time))       
        new = np.flip(img, axis=-1)
        self.ui.w4_2.show()
        self.ui.w4_2.setImage(np.rot90(new,1))           

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = App()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()        