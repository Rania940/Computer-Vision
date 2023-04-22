import cv2
import numpy as np
import  matplotlib.pyplot as plt

def optimal_thresholding(image):
    background_sum = (image[0,0] + image[0,-1] + image[-1,0] + image[-1,-1])
    foreground_sum = np.sum(image) - background_sum

    background_mean = background_sum / 4
    foreground_mean = foreground_sum / (np.size(image) - 4)
    threshold = (background_mean + foreground_mean) / float(2)
    flag=True
    while flag:
        old_thresh = threshold
        foreground = np.extract(image > threshold, image)
        background = np.extract(image < threshold, image)

        if foreground.size:
            foreground_mean = np.mean(foreground)
        else:
            foreground_mean = 0

        if background.size:
            background_mean = np.mean(background)
        else:
            background_mean = 0

        threshold = (background_mean + foreground_mean) / float(2)

        if old_thresh == threshold:
            flag = False

    return threshold


