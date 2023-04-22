import numpy as np
import matplotlib.pyplot as plt
import cv2



class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1),
                    Point(1, 0), Point(1, 1), Point(0, 1),
                    Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]

    return connects


def regionGrow(img, seeds, thresh, p = 1):
    height, weight = img.shape
    belogToRegion = np.zeros(img.shape)

    label = 1
    connects = selectConnects(p)

    while (len(seeds) > 0):
        currentPoint = seeds.pop(0)
        belogToRegion[currentPoint.x, currentPoint.y] = label

        for i in range(len(connects)):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y

            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue

            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))

            if grayDiff < thresh and belogToRegion[tmpX, tmpY] == 0:
                belogToRegion[tmpX, tmpY] = label
                seeds.append(Point(tmpX, tmpY))

    return belogToRegion



if __name__ == "__main__":

    img = cv2.imread('pic/shapes.jpg')
    # img = cv2.imread('pic/rice.png')
    # img = cv2.imread('pic/scene.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    seeds = []
    for i in range(3):
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])
        seeds.append(Point(x, y))

    binaryImg = regionGrow(img_gray, seeds, 10)
    # binaryImg = regionGrow(img_gray, seeds, 10, 0)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(binaryImg, cmap='gray')
    plt.axis('off')
    plt.title(f'Segmented image')

    plt.show()