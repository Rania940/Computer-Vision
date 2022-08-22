from PyQt5 import QtCore, QtGui, QtWidgets
from ui2 import Ui_MainWindow
from scipy import ndimage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import qimage2ndarray
from scipy import signal
from scipy.signal import convolve2d
import ntpath
import copy
import sys
import cv2
from math import sqrt, pi, cos, sin, atan2,exp
import numpy as np
import imageio
import os.path
#import pyqtgraph as pg
from skimage.transform import resize
import timeit
from PIL import Image, ImageDraw
import time
# import harris_corner_detection
# from Feature_matching import feature_matching


class App(QtWidgets.QMainWindow):
    
    path1 = ""
    path2 = ""
    path3 = ""
    
    
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.b1.clicked.connect(self.loaddata1)
        self.ui.b3.clicked.connect(self.loaddata2)
        self.ui.b4.clicked.connect(self.loaddata3)
        self.ui.b2.clicked.connect(self.harris)
        self.ui.b5.clicked.connect(self.matching)
    
    
    
    def loaddata1(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path1 = filename[0]
        
        image = cv2.imread(self.path1)
        new = np.flip(image, axis=-1) 
        self.ui.w1.show()
        self.ui.w1.setImage(np.rot90(new,1))
        
    def loaddata2(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path2 = filename[0]
        
        image = cv2.imread(self.path2)
        new = np.flip(image, axis=-1)
        self.ui.w3.show()
        self.ui.w3.setImage(np.rot90(image,1))
        self.ui.w5.show()
        self.ui.w5.setImage(np.rot90(image,1))
        
    def loaddata3(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path3 = filename[0]
        
        image = cv2.imread(self.path3)
        new = np.flip(image, axis=-1)
        self.ui.w4.show()
        self.ui.w4.setImage(np.rot90(image,1))
        self.ui.w6.show()
        self.ui.w6.setImage(np.rot90(image,1))

    def harris(self):
        # starting time
        start = time.time()
        image = cv2.imread(self.path1)   
        harris_corner_detection.harris_corner_detector(image,5, 0.035, 0.3)
        out_harris = cv2.imread("harris.jpg")
        new = np.flip(out_harris, axis=-1) 
        # cvt_harris = cv2.cvtColor("harris.jpg", cv2.COLOR_BGR2RGB) 


        # end time
        end = time.time()
        # total time taken
        print(f"The computation time of harris corner detection is {end - start} seconds")
        self.ui.w2.show()
        self.ui.w2.setImage(np.rot90(new,1))           
 
          
    def matching(self):
        image1 = cv2.imread(self.path2)
        image1 = cv2.resize(image1, (1000,1000), interpolation = cv2.INTER_AREA)

        image2 = cv2.imread(self.path3)
        image2 = cv2.resize(image2, (300,300), interpolation = cv2.INTER_AREA) 

        feature_matching(image1, image2, type='ssd')     



        out_ssd = cv2.imread("matched_image.png")
        self.ui.w3.show()
        self.ui.w3.setImage(np.rot90(out_ssd,1)) 
           



def main():
    app = QtWidgets.QApplication(sys.argv)
    application = App()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
