import os
from re import L
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import Eigen_Faces as ef 
 
     
     
def project (W , X , mu):
    return np.dot (X - mu , W)
def reconstruct (W , Y , mu) :
    return np.dot (Y , W.T) + mu



if __name__ == "__main__":

    [x, y] = ef.read_images(image_path=ef.IMAGE_DIR) 
    [eigenvalues, eigenvectors, mean] = ef.pca (ef.as_row_matrix(x), y)

    steps =[i for i in range (eigenvectors.shape[1])]
    E = []
    for i in range (len(steps)):
        numEvs = steps[i]
        P = project(eigenvectors[: ,0: numEvs ], x[0].reshape (1 , -1) , mean)
        R = reconstruct(eigenvectors[: ,0: numEvs ], P, mean)
        # reshape and append to plots
        R = R.reshape(x[0].shape )
        E.append(np.asarray(R))

    # print(len(E))
    # ef.draw_eigenfaces( title ="Reconstruction", images =E[:16], rows =4, cols =4, colormap =plt.cm.gray , filename ="pca_reconstruction.png")      
    ef.draw_eigenfaces( title ="Reconstruction", images =E[:16], rows =4, cols =4, colormap =plt.cm.gray , filename =None)