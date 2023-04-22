import cv2
import numpy as np
import  matplotlib.pyplot as plt


def otsu_thresholding(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
   # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(256)

    fn_min = np.inf
    threshold = -1

    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        b1, b2 = np.hsplit(bins, [i])  # weights

        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2

        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            threshold = i
    return threshold

















# # SEGMENTATION Otsu's binary threshold
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread("pic/rice.png")
# b,g,r = cv2.split(img)
# rgb_img = cv2.merge([r,g,b])
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # noise removal
# kernel = np.ones((2,2),np.uint8)
# #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
# # sure background area
# sure_bg = cv2.dilate(closing,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
# # Threshold
# ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]
# plt.subplot(211),plt.imshow(rgb_img)
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(212),plt.imshow(thresh, 'gray')
# plt.imsave(r'thresh.png',thresh)
# plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.show()
