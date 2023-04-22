import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


def hough_line(edgeImage):
    # Theta 0 - 180 degree
    # Calculate 'cos' and 'sin' value ahead to improve running time
    theta = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))

    # Generate a accumulator matrix to store the values
    rho_range = round(math.sqrt(edgeImage.shape[0] ** 2 + edgeImage.shape[1] ** 2))
    accumulator = np.zeros((2 * rho_range, len(theta)), dtype=np.uint8)

    # Threshold to get edges pixel location (x,y)
    edge_pixels = np.where(edgeImage != 0)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    # Calculate rho value for each edge location (x,y) with all the theta range
    for p in range(len(coordinates)):
        for t in range(len(theta)):
            rho = int(round(coordinates[p][1] * cos[t] + coordinates[p][0] * sin[t]))
            accumulator[rho, t] += 2  # Suppose add 1 only, Just want to get clear result

    return accumulator


#

# Draw the lines represented in the hough accumulator on the original image
def drawhoughLinesOnImage(image, coordinates):
    for i in range(0, len(coordinates)):
        a = np.cos(np.deg2rad(coordinates[i][1]))
        b = np.sin(np.deg2rad(coordinates[i][1]))
        x0 = a * coordinates[i][0]
        y0 = b * coordinates[i][0]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 127), 2)

        # Different weights are added to the image to give a feeling of blending


def blend_images(image, final_image, alpha=0.7, beta=1., gamma=0.):
    return cv2.addWeighted(final_image, alpha, image, beta, gamma)


if __name__ == "__main__":
    image = cv2.imread("pic/line4.png")  # load image in grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurredImage = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edgeImage = cv2.Canny(blurredImage, 50, 120)
    accumulator = hough_line(edgeImage)
    # Threshold some high values then draw the line
    edge_pixels = np.where(accumulator > 220)

    houghLinesImage = np.zeros_like(image)  # create and empty image

    drawhoughLinesOnImage(houghLinesImage, accumulator)  # draw the lines on the empty image
    orginalImageWithHoughLines = blend_images(houghLinesImage, image)  # add two images together, using image blending
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(edgeImage, cmap='gray')
    ax2.set_title('Edge Image')
    ax2.axis('off')

    ax3.imshow(orginalImageWithHoughLines, cmap='gray')
    ax3.set_title("Original Image with Hough lines")
    ax3.axis('off')
    plt.show()