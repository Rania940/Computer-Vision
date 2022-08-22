import numpy as np
import cv2
from Otsu import otsu_thresholding
from optimal import optimal_thresholding
from spectral import spectral_thresholding
import matplotlib.pyplot as plt


def Global_threshold(image , thresh_typ = "Optimal"):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh_img = np.zeros(image.shape)
    if thresh_typ == "Otsu":
        threshold = otsu_thresholding(image)
        thresh_img = np.uint8(np.where(image > threshold, 255, 0))
    elif thresh_typ == "Optimal":
        threshold = optimal_thresholding(image)
        thresh_img = np.uint8(np.where(image > threshold, 255, 0))
    else:
        # De-noise the image with gaussian blur
        image = cv2.GaussianBlur(image, (5, 5), 0)
        thresholds, thresh_img=spectral_thresholding(image)
    return thresh_img

def Local_threshold(image, block_size , thresh_typ = "Optimal"):
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if (thresh_typ == 'Spectral'):
        thresh_img = np.zeros(image.shape,dtype=np.uint8)
        for row in range(0, image2.shape[0], block_size):
            for col in range(0, image2.shape[1], block_size):
                mask = image2[row:min(row+block_size,image2.shape[0]),col:min(col+block_size,image2.shape[1])]
                thresh_img[ row:min(row+block_size,image2.shape[0]),col:min(col+block_size,image2.shape[1]),:] = Global_threshold(mask, thresh_typ)
        thresh_img=cv2.cvtColor(thresh_img, cv2.COLOR_RGB2GRAY)
    else:
        thresh_img = np.zeros(image2.shape)
        for row in range(0, image2.shape[0], block_size):
            for col in range(0, image2.shape[1], block_size):
                mask = image2[row:min(row+block_size,image2.shape[0]),col:min(col+block_size,image2.shape[1])]
                thresh_img[ row:min(row+block_size,image2.shape[0]),col:min(col+block_size,image2.shape[1])] = Global_threshold(mask, thresh_typ)

    return thresh_img




source_image = cv2.imread("pic/apple.jpg")
# optimal_local = Local_threshold(source_image, 50,  "Optimal")
# optimal_global = Global_threshold(source_image, "Optimal")

spectral_global = Global_threshold(source_image, "Spectral")
spectral_local = Local_threshold(source_image, 10,"Spectral")
# # print(otsu)
#cv2.imshow('Original image', source_image)
# cv2.imshow('Local optimal thresholding', optimal_local)
#cv2.imshow('Global thresholding', optimal_global)
cv2.imshow('local spectral thresholding', spectral_local)
cv2.imshow('global spectral thresholding', spectral_global)

cv2.waitKey(0)

# cv2.destroyAllWindows()