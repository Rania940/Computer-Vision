import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from SIFT import sift 


im1="pic\lena.jpg"
im2="pic\lena_rot.jpg"
img1 = cv2.imread("pic\cow1.jpg")
img2= cv2.imread("pic\cow2.jpg")


img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img1_grey=cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2_grey=cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)


# def opencv_kp(kp_list):
#     cv_kp_list = []
#     for kp in kp_list:
#         opencv_kp = cv2.KeyPoint(x=kp[1] * (2**(kp[2]-1)), y=kp[0] * (2**(kp[2]-1)) , size=kp[3] , angle=kp[4])
#         cv_kp_list += [opencv_kp]
    
    #return tuple(cv_kp_list)

def get_kp_desc(img_grey):

    KP,DESC , del_t = sift(img_grey)
    #KP = opencv_kp(KP)
    #DESC=np.asarray(DESC)
    #print(DESC[0].shape)

    # i = cv2.drawKeypoints(img1, KP, None, color=(255,0,0))
    # plt.figure(figsize=(20,20))
    # plt.imshow(i)
    # plt.show()

    return KP, DESC,del_t


kp1, desc1 , del_t1=get_kp_desc(im1)
kp2, desc2 , del_t2=get_kp_desc(im2)
desc1=np.array(desc1)
desc2=np.array(desc2)


kp1 , desc1 , del_t1=get_kp_desc(im1) 
kp2 , desc2 , del_t2=get_kp_desc(im2)
# keypoints_1, descriptors_1 , del_t1= get_kp_desc(im1)
# keypoints_2, descriptors_2 , del_t2= get_kp_desc(im2)

#1.....calculate ssd of descritors vectors 
def sum_square_difference (desc1, desc2):

    sum_square = 0
    
    for i in range(len(desc1)):
        sum_square = sum_square + (desc1[i] - desc2[i]) ** 2

    sum_square = - (np.sqrt(sum_square)) 

    return sum_square

#2.......calculate ncc of descriptors
def Normalized_cross_correlation(desc1 , desc2 ): 
    
    # Normalize the 2 vectors
    out1_normalized = (desc1 - np.mean(desc1)) / (np.std(desc1))
    out2_normalized = (desc2 - np.mean(desc2)) / (np.std(desc2))

    # Apply similarity product between the 2 normalized vectors
    correlation_vector = np.multiply(out1_normalized, out2_normalized)

    # Get mean of the result vector
    ncc = float(np.mean(correlation_vector))

    return ncc


def apply_feature_matching(desc_img_1 , desc_img_2, match_cal): 
    
    
    # number of key points in each image
    num_key_points_img1 = desc_img_1.shape[0]
    num_key_points_img2 = desc_img_2.shape[0]
    
    matches = []

    
    for kp1 in range(num_key_points_img1):
        
        distance = -np.inf
        y_index = -1

        # calculate similarity with each key point in image2
        
        for kp2 in range(num_key_points_img2):

            # Match features between the 2 vectors
            val = match_cal(desc_img_1[kp1], desc_img_2[kp2])

            if val > distance:
                distance = val
                y_index = kp2

        # Create a cv2.DMatch object 
        match = cv2.DMatch()
        match.queryIdx = kp1          
        match.trainIdx = y_index       
        match.distance = distance     
        matches.append(match)

    return matches

# def feature_matching(input_img1,input_img2,type='ssd'):

#     img1 = cv2.imread(input_img1)
#     img2= cv2.imread(input_img2)
#     img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#     img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#     # sift to apply the featuresmatching of ssd and ncc and use keypoints to draw the matches
#     sift__ = cv2.SIFT_create()

#     keypoints_1, descriptors_1 = sift__.detectAndCompute(img1, None)
#     keypoints_2, descriptors_2 = sift__.detectAndCompute(img2, None)

#     if type == 'ssd':
#         start_ssd = time.time()  #ssd and calculate computation time
#         matches_ssd = apply_feature_matching(descriptors_1 , descriptors_2 , sum_square_difference)
#         matches_ssd = sorted(matches_ssd, key=lambda x: x.distance, reverse=True)
#         matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches_ssd[:30], img2, flags=2)
#         end_ssd = time.time()
#         print ("Computation time of Sum Square Distance =  " , end_ssd - start_ssd, " seconds")

#     else:
#         start_ncc = time.time() # ncc and calculate computation time
#         matches_ncc = apply_feature_matching(descriptors_1, descriptors_2, Normalized_cross_correlation)
#         matches_ncc = sorted(matches_ncc, key=lambda x: x.distance, reverse=True)
#         matched_image = cv2.drawMatches(img1,keypoints_1, img2,keypoints_2, matches_ncc[:30], img2, flags=2)
#         end_ncc = time.time()
#         print ("Computation time of Normalized Cross Correlation =  " , end_ncc - start_ncc, " seconds")


#     #plot matches
#     plt.figure(figsize=(40, 40))

#     # showing image
#     plt.imshow(matched_image)
#     plt.axis('off')
#     plt.title("ssd_result")
#     plt.show()
    

# sift to apply the featuresmatching of ssd and ncc and use keypoints to draw the matches
def draw_matches():
    sift__ = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift__.detectAndCompute(img1, None) 
    keypoints_2, descriptors_2 = sift__.detectAndCompute(img2, None)
    start_ssd = time.time()  #ssd and calculate computation time
    matches_ssd = apply_feature_matching(descriptors_1 , descriptors_2 , sum_square_difference)
    matches_ssd = sorted(matches_ssd, key=lambda x: x.distance, reverse=True)
    matched_image_ssd = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches_ssd[:30], img2, flags=2)
    end_ssd = time.time()
    print ("Computation time of Sum Square Distance =  " , end_ssd - start_ssd, " seconds")


    start_ncc = time.time() # ncc and calculate computation time
    matches_ncc = apply_feature_matching(descriptors_1, descriptors_2, Normalized_cross_correlation)
    matches_ncc = sorted(matches_ncc, key=lambda x: x.distance, reverse=True)
    matched_image_ncc = cv2.drawMatches(img1,keypoints_1, img2,keypoints_2, matches_ncc[:30], img2, flags=2)
    end_ncc = time.time()
    print ("Computation time of Normalized Cross Correlation =  " , end_ncc - start_ncc, " seconds")


keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp1]
cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp2]

keypoints_1, descriptors_1 = sift__.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift__.detectAndCompute(img2, None)




    #plot matches
    figure_matching = plt.figure(figsize=(40, 40))
    figure_matching.add_subplot(2, 1, 1)


    # showing image1
    plt.imshow(matched_image_ssd)
    plt.axis('off')
    plt.title("ssd_result")

start_ncc = time.time() # ncc and calculate computation time
matches_ncc = apply_feature_matching(descriptors_1, descriptors_2, Normalized_cross_correlation)
matches_ncc = sorted(matches_ncc, key=lambda x: x.distance, reverse=True)
matched_image_ncc = cv2.drawMatches(img1,keypoints_1, img2, keypoints_2, matches_ncc[:30], img2, flags=2)
end_ncc = time.time()
print ("Computation time of Normalized Cross Correlation =  " , end_ncc - start_ncc, " seconds")



    # Adds a subplot at the 2nd position
    figure_matching.add_subplot(2, 1, 2)

    # showing image2
    plt.imshow(matched_image_ncc)
    plt.axis('off')
    plt.title("ncc_result")


    plt.show()
    
    
draw_matches()