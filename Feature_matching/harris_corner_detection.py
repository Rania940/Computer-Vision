import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import timeit

def harris_corner_detector(img, window_size, k, threshold):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    height = img.shape[0]  # .shape[0] outputs height
    width = img.shape[1]  # .shape[1] outputs width .shape[2] outputs color channels of image
    matrix_R = np.zeros((height, width))

    # Calculate the x and y image derivatives (Ix & Iy)
    Ix = cv2.Sobel(gaussian_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gaussian_img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate product and second derivatives (Ixx, Iyy & Ixy)
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = Ix * Iy

    offset = int(window_size / 2)  # Shifted the window by an offset value

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            #   Calculate a harris matrix. H(x,y)=[[Sxx,Sxy],[Sxy,Syy]]
            H = np.array([[Sxx, Sxy], [Sxy, Syy]])

            #   Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R = det - k * (tr ** 2)
            matrix_R[y - offset, x - offset] = R

    #  Apply a threshold
    ## the response results were normalized and only vary between 0 and 1,
    ## this way the threshold value should also be a value between 0 and 1.
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            value = matrix_R[y, x]
            if value > threshold:
                # this is a corner
               cv2.circle(img, (x, y), 1, (125, 0, 255))



    plt.figure(" Harris corner detector")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Harris corner detector")
    plt.show()


def harris(input_file, window_size=5, k=0.05, threshold=0.5):
    # starting time
    start = time.time()
    image = cv2.imread(input_file,)
    harris_corner_detector(image,window_size, k, threshold)
    # end time
    end = time.time()
    # total time taken
    print(f"The computation time of harris corner detection is {end - start} seconds")


def main():
  # starting time
  start = time.time()
  image = cv2.imread("pic/lena.jpg")
  harris_corner_detector(image,5, 0.035, 0.3)
  # end time
  end = time.time()
  # total time taken
  print(f"The computation time of harris corner detection is {end - start} seconds")

if __name__ == "__main__":
   main()

