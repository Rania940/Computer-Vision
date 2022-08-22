from unittest import skip
import numpy as np
import cv2
import itertools as IT
from collections import defaultdict

def detect_circles(image,r_min=None, r_max=None, delta_r=1, num_thetas=360, min_votes=0.44):
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.GaussianBlur(img_grey, (5, 5), 1.3)
    edges = cv2.Canny(img_grey, 100,150)

    cv2.imshow("Edges",edges)
    cv2.waitKey(0)
    w,h= img_grey.shape

    # thetas array 
    delta_theta = int(360 / num_thetas)
    thetas = np.arange(0, 360, delta_theta)
    # Calculate Cos(theta) and Sin(theta) it will be required later
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    #radius array
    if r_min == None:
        r_min = 2
    if r_max == None:
        r_max = w if (w < h)  else h 

    r = np.arange(r_min,r_max,delta_r,dtype=np.int32)

    Acc = defaultdict(int)  
    edges = np.argwhere(edges[:, :])

    #r_ = r[:,np.newaxis]
    #sin_update= np.multiply(r_,sin_thetas).astype(int)
    #cos_update= np.multiply(r_,cos_thetas).astype(int)

    # for x,y in edges:
    #     for rad in r:
    #         a = (x - cos_update[rad])
    #         b = (y - sin_update[rad])
    #         Acc[a , b, rad] = Acc[a, b, rad] + 1

    for x,y in edges:
        for rad in r:
            for s, c in zip(sin_thetas, cos_thetas):
                a = x - int(rad*s)
                b = y - int(rad*c)
                if (a>0 and a<w and b>0 and b<h):
                    Acc[(a, b, rad)] += 1 

    #Acc=dict(np.ndenumerate(Acc))
    out_circles = []
    cpy = np.copy(image)

    for candidate_circle, votes in sorted(Acc.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas

        # Filter upove minvote percentage and remove nearby circles
        if current_vote_percentage >= min_votes and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in out_circles):
            out_circles.append((x, y, r))
            cv2.circle(cpy, (y, x), r, color=(0, 255, 0), thickness=1)


    
    cv2.imshow("circles",cpy)
    cv2.waitKey(0)

       
    
def detect_ellipse(image,min_votes=3):
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.GaussianBlur(img_grey, (5, 5), 0.75)
    edges = cv2.Canny(img_grey, 60,100)

    w,h= img_grey.shape
    edges = np.argwhere(edges[:, :])
    cpy = np.copy(image)
    Acc = np.zeros((1, (max(w, h))//2,1))
    #ij1, ij2 are indexes of (x1, y1) and (x2, y2), following the reference [1].
    for i in range(0,w):
        for k in range (w-1,(i+1),-1):
            x1 = edges[i] [0]
            y1 = edges[i] [1]
            x2 = edges[k] [0]
            y2 = edges[k] [1]
            d12 = np.linalg.norm(np.subtract([x1,y1],[x2,y2])) 
            Acc = Acc * 0
            if  x1 - x2 and d12 > 3:
                #Center
                x0 = (x1 + x2)/2
                y0 = (y1 + y2)/2
                #Half-length of the major axis
                a = d12/2
                #Orientation
                alpha = np.arctan((y2 - y1)/(x2 - x1))
                #Distances between the two points and the center
                d01 = np.linalg.norm(np.subtract([x1,y1],[x0,y0]))
                d02 = np.linalg.norm(np.subtract([x2,y2],[x0,y0]))
                for l in range(0,w):
                    if (l == i) and (l == k):
                        continue
                     
                    x3 = edges[l][0]
                    y3 = edges[l][1]
                    d03 = np.linalg.norm(np.subtract([x3,y3],[x0,y0]))
                    
                    if  d03 >= a:
                        continue

                    f = np.linalg.norm(np.subtract([x3,y3],[x2,y2]))
                    #estimating the half length of b
                    cos2 = ((a**2 + d03**2 - f**2) / (2 * a * d03))**2
                    sin2 = 1 - cos2
                    b = round(np.sqrt((a**2 * d03**2 * sin2) /(a**2 - d03**2 * cos2)))

                    if b > 0 and b < len(Acc):
                        Acc[b]+=1
                print(Acc)
                bv,bi = max(Acc)
                if bv > min_votes:
                    parameters = [x0,y0,a,bi,alpha]
                    cv2.ellipse(cpy,(x0,y0),(a,bi),alpha,color=(0, 255, 0), thickness=1)
                    # print(parameters)
    cv2.imshow("ellipses",cpy)
    cv2.waitKey(0)



img=cv2.imread("img4.PNG")
detect_circles(img,r_min =4, r_max=15,min_votes=0.5)
#detect_ellipse(img)
