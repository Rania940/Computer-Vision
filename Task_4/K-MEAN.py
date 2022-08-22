import pandas as pd
import cv2
import math
import numpy as np
import random
import matplotlib.pyplot as plt




#to calculate the distance between pixels and centroids
def distance_calc(a, b):
    dis=sum(np.square((a-b)))
    dis=np.sqrt(dis)
    #dis=math.dist(a, b)
    return(dis)



 #assign the minimum distance to make the clusters   
def closest_centroids(c, im):
    assigned_centroid=[]
    for i in im:
        dis_centroid_img=[]
        for j in c:
             dis_centroid_img.append(distance_calc(i,j))
    
        assigned_centroid.append(np.argmin(dis_centroid_img)) 
        #print(assigned_centroid)
    return assigned_centroid 


 
#function to update the centroids to converge 
def centroids_update(clusters, im) :
    updated_centroids=[] 
    df_img=pd.concat([pd.DataFrame(im), pd.DataFrame(clusters, columns=['CLUSTER'])], axis=1)
    for k in set(df_img['CLUSTER']):
        current_clusters=df_img[df_img['CLUSTER']==k][df_img.columns[:-1]]
        mean_of_cluster=current_clusters.mean(axis=0)
        updated_centroids.append(mean_of_cluster)
        print("updated",updated_centroids)
    return  updated_centroids 


   
        
im = cv2.imread('pic/flower.jpg')
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#convert rgb image to luv
img1 = cv2.cvtColor(rgb,cv2.COLOR_RGB2Luv)
 
 
 
        

img = (img1/255).reshape(img1.shape[0]*img1.shape[1], 3)
print(img.shape)

#firsly choose centroids randomly
random_index = random.sample(range(0, len(img)), 2)

centroids = []
for i in random_index:
    centroids.append(img[i])
centroids = np.array(centroids)


#then, update 
for i in range(2):
    get_centroids =closest_centroids(centroids, img)
    centroids = centroids_update(get_centroids, img)
#..................

img_recovered = img.copy()
for i in range(len(img)):
    img_recovered[i] = centroids[get_centroids[i]]



img_recovered = img_recovered.reshape(img1.shape[0],img1.shape[1], 3)



fig , ax = plt.subplots(1,2)
ax[0].imshow(rgb)
ax[0].title.set_text("original_rgb")
ax[1].imshow(img_recovered)
ax[1].title.set_text("segmented_luv")
plt.show()        