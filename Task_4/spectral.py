import  cv2
import numpy as np
import matplotlib.pyplot as plt

def normalised_histogram_binning(hist, M=32, L=256):
    
    norm_hist = np.zeros((M, 1), dtype=np.float32)
    N = L // M
    counters = [range(x, x+N) for x in range(0, L, N)]
    for i, C in enumerate(counters):
        norm_hist[i] = 0
        for j in C:
            norm_hist[i] += hist[j]
    norm_hist = (norm_hist / norm_hist.max()) * 100
    return norm_hist


def valleys_estimation(hist, M=32, L=256):
    # stage one find valleys in the groupped histogram

    #  bin groupping to speed up the searching of sets that contain thresholds
    H = normalised_histogram_binning(hist, M, L)

    # find valley locations by determining the probability of the bin to become a valley in the normalized histogram distribution
    hsize = H.shape[0]
    probs = np.zeros((hsize, 1), dtype=int)

    for i in range(1, hsize-1):
        if H[i] > H[i-1] or H[i] > H[i+1]:
            probs[i] = 0

        elif H[i] < H[i-1] and H[i] == H[i+1]:
            probs[i] = 1

        elif H[i] == H[i-1] and H[i] < H[i+1]:
            probs[i] = 3

        elif H[i] < H[i-1] and H[i] < H[i+1]:
            probs[i] = 4

        elif H[i] == H[i-1] and H[i] == H[i+1]:
            probs[i] = probs[i-1]

    for i in range(1, hsize-1):
        if probs[i] != 0:
            probs[i] = (probs[i-1] + probs[i] + probs[i+1]) // 4

    valleys = [i for i, x in enumerate(probs) if x > 0]

    return valleys

def optimized_otsu(hist):
    """"
    Adapted from https://github.com/subokita/Sandbox/blob/master/otsu.py
    """
    num_bins = hist.shape[0]
    total = hist.sum()
    sum_total = np.dot(range(0, num_bins), hist)

    weight_background = 0.0
    sum_background = 0.0

    optimum_value = 0
    maximum = -np.inf

    for t in range(0, num_bins):
        # background weight will be incremented, while foreground's will be reduced
        weight_background += hist.item(t)
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist.item(t)
        mean_foreground = (sum_total - sum_background) / weight_foreground
        mean_background = sum_background / weight_background

        inter_class_variance = weight_background * weight_foreground * \
            (mean_background - mean_foreground) ** 2

        # find the threshold with maximum variances between classes
        if inter_class_variance > maximum:
            optimum_value = t
            maximum = inter_class_variance
    return optimum_value, maximum


def threshold_valleys(hist, valleys, N):
    # stage two perform Otsu threshold to the estimated valley regions
    
    thresholds = []
    for valley in valleys:
        start_pos = (valley * N) - N
        end_pos = (valley + 2) * N
        h = hist[start_pos:end_pos]
        sub_threshold, val = optimized_otsu(hist=h)
        thresholds.append((start_pos + sub_threshold, val))
    
    #sort based on between class variance
    thresholds.sort(key=lambda x: x[1], reverse=True)
    thresholds = list(map(lambda x: x[0], thresholds))
    #print(thresholds)
    return thresholds



def multithreshold_image(img, thresholds):
    # multithreshold a single-channel image using provided thresholds

    masks = np.zeros((len(thresholds) + 1, img.shape[0], img.shape[1]), bool)
    
    for i, t in enumerate(sorted(thresholds)):
        masks[i+1] = (img > t)
    
    #background
    if len(thresholds)>1:
        masks[0] = ~masks[1]
    
    for i in range(1, len(masks) - 1):
        masks[i] = masks[i] ^ masks[i+1]

    
    return masks
    
def show_thresholds(src_img, dst_img ,thresholds):
    colors = [(255, 255, 0) , (128,255,0) , (0,255,255) , (0, 128, 255), (127,0,255) , (255, 0, 255)  , (255,0, 127), (153,0,76), (153, 0, 153), (76, 0, 153) , (0,0,102),(255, 155, 0) , (128,255,128) , (100,0,255) , (200, 0, 200)  , (250,0, 107), (153,50,76), (255,255,255) , (0, 50, 200), (50,50,255) ,(0,0,0),(10,70,50),(90,200,170),(140,60,200),(80,80,80),(10,255,155),(0,0,70),(70,0,0),(80,80,255),(0,70,0),(100,200,100),(200,1170,80)]
    
    masks = multithreshold_image(src_img, thresholds)
    
    for i, mask in enumerate(masks):
        dst_img[mask] = colors[i]

    # plt.figure()
    # ax = plt.subplot(1, 2, 1)
    # ax.set_title('Original image')
    # plt.imshow(src_img, cmap='gray')
    # ax = plt.subplot(1, 2, 2)
    # ax.set_title('{} levels'.format(len(thresholds)))
    # plt.imshow(dst_img)
    # plt.show()
    
    return dst_img


def spectral_thresholding(img, L=256, M=32):
    """ Two Stage Multithreshold Otsu thresholding , based on Automatic multilevel thresholding based on two-stage Otsu method with cluster
        determination by valley estimation. International Journal of Innovative Computing, Information and Control, 7(10), 56315644
    """
    hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[L], ranges=[0, L] )

    #get num of vals in each bin
    N = L // M
    valleys = valleys_estimation(hist, M, L)
    thresholds = threshold_valleys(hist, valleys, N)

    #show thresholds
    img_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    thresholded_im=show_thresholds(img, img_1, thresholds)
    # plt.figure()
    # plt.imshow(thresholded_im)
    # plt.show()

    return thresholds , thresholded_im


    


if __name__ == "__main__":

    # read the image in grey scale
    image = cv2.imread("pic/apple.jpg",0)

    # De-noise the image with gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    #spectral thresholding
    thresholds , thresholded_im = spectral_thresholding(image)

    #show thresholds
    cv2.imshow('spectral thresholding', thresholded_im)

    cv2.waitKey(0)
    


