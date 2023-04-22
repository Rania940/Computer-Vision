import os
from re import L
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import Eigen_Faces as ef
import project_new_faces as projectf

def dist_metric(p,q):
    p = np.asarray(p).flatten()
    q = np.asarray (q).flatten()
    return np.sqrt (np.sum (np. power ((p-q) ,2)))

def predict (W, mu , projections, y, X):
    minDist = float("inf")
    minClass = -1
    Q = projectf.project(W, X.reshape (1 , -1) , mu)
    for i in range (len(projections)):
        dist = dist_metric( projections[i], Q)
        if dist < minDist:
            minDist = dist
            minClass = i
    return minClass





if __name__ == "__main__":

    [X, y] = ef.read_images(image_path=ef.IMAGE_DIR)
    [eigenvalues, eigenvectors, mean] = ef.pca(ef.as_row_matrix(X), y)
    projections = []
    for xi in X:
        projections.append(projectf.project(eigenvectors, xi.reshape(1 , -1) , mean))

    image = Image.open("test_data/Amanda_Beard_0002.jpg")
    image = image.convert ("L")
    if (ef.DEFAULT_SIZE is not None ):
        image = image.resize (ef.DEFAULT_SIZE , Image.ANTIALIAS )
    test_image = np. asarray (image , dtype =np. uint8 )
    predicted = predict(eigenvectors, mean , projections, y, test_image)

    ef.draw_eigenfaces( title ="Prediction", images =[test_image, X[predicted]], rows =1, cols =2,
             sptitles = ["Unknown image", "Prediction :{0}".format(y[predicted])] , colormap =plt.cm.gray ,
             filename ="prediction_test.png", figsize = (5,5))
