
from cv2 import resize
import numpy as np
import  matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2
import time 
from skimage.transform import resize

octaves=[]
dogs = []
octaves_keypoints = []
nb_octaves = 4
nb_scales = 5 
sigma = 1.6 
K = np.sqrt(2)

sigmas = [(K**i)*sigma for i in range(nb_scales)]
kernlen = [(2*int(round(s))) for s in sigmas]

#slice that dealwith out of bounds indices
def slice(img, sl):
    output_shape = np.asarray(np.shape(img))
    output_shape[0] = sl[1] - sl[0]
    output_shape[1] = sl[3] - sl[2]
    src = [max(sl[0], 0), min(sl[1], img.shape[0]), max(sl[2], 0), min(sl[3], img.shape[1])]
    dst = [src[0] - sl[0], src[1] - sl[0], src[2] - sl[2], src[3] - sl[2]]
    output = np.zeros(output_shape, dtype=img.dtype)
    output[dst[0]:dst[1],dst[2]:dst[3]] = img[src[0]:src[1],src[2]:src[3]]
    return output

# Get gradient at defined angles
def angle_histogram(phase, magnitude, bins):
    gradientMagnitudes = []
    angles = range(0, 360, 360//bins) 
    for angle in angles:
        indices = np.where(phase == angle)
        gradients = np.sum(magnitude[indices])
        gradientMagnitudes.append(gradients)
    return gradientMagnitudes


def is_extrema( below_im, at_im, above_im ):
    value = at_im[1,1]

    if value > 0:
        return all([np.all( value >= img ) for img in [below_im,at_im,above_im]]) 
    else:
        return all([np.all( value <= img ) for img in [below_im,at_im,above_im]]) 


def scale_space(img, nb_octaves = 4, nb_scales = 5 , sigma = 1.6 ,K = np.sqrt(2) ,show=False):

    #upsampling first picture
    image = cv2.resize(img, (0,0), fx=2, fy=2,interpolation=cv2.INTER_NEAREST )
    #image = img.repeat(2, axis=0).repeat(2, axis=1)
    #image  = rescale( img, 2, anti_aliasing=False) 
    for octave in range( nb_octaves ):
        scales=[]
        for i in range(nb_scales):
            #blurred = cv2.GaussianBlur(image,(2*kernlen[i]+1,2*kernlen[i]+1),sigmas[i])
            kernel=cv2.getGaussianKernel(2*kernlen[i]+1,sigmas[i])
            blurred = convolve2d( image , np.outer( kernel,kernel), 'same', 'symm') 
            scales.append(blurred)
        
        #append scales of the octave
        octaves.append(scales)
        
        #append dog of scales in the octave
        dogs.append([ s2 - s1 for (s1,s2) in zip( octaves[octave][:-1], octaves[octave][1:])])
        
        #subsampling the second to last image
        image = octaves[octave][2][::2,::2]
        #image = cv2.resize(octaves[octave][2], (0,0), fx=0.5, fy=0.5,interpolation=cv2.INTER_LINEAR )
  
    
    if show:
    
        fig1, ax1 = plt.subplots(nb_octaves,nb_scales,figsize = (15, 10))
        for octave_idx in range(nb_octaves):
            img_octave = octaves[octave_idx]
            for scale_idx in range(nb_scales):
                subplot = ax1[octave_idx,scale_idx]
                img_scale = img_octave[scale_idx]
                subplot.imshow(img_scale, cmap = 'gray')

        fig2, ax2 = plt.subplots(nb_octaves,nb_scales-1,figsize = (15, 10))
        for octave_idx in range(nb_octaves):
            img_octave_dogs = dogs[octave_idx]
            for dog_idx in range(len(img_octave_dogs)):
                subplot = ax2[octave_idx,dog_idx]
                img_dog = img_octave_dogs[dog_idx]
                subplot.imshow(img_dog, cmap = 'gray')

        plt.show()

def locate_keypoints(hess_ratio=10, contrast_threshold =0.0002, im_max=255.0):
    count =0
    hess_threshold = ((hess_ratio + 1.0)**2)/hess_ratio
    
    for octave in range(nb_octaves):
        octave_keypoints= []
        octave_dogs = dogs[octave]
        for idx in range(1, len(octave_dogs) -1):
            dog_image = octave_dogs[idx]
            dog_keypoints = []
            
            #corners only from keypoints using Hessian Matrix
            gradX =cv2.Sobel(dog_image,ddepth=cv2.CV_64F,dx=1,dy=0)
            gradY =cv2.Sobel(dog_image,ddepth=cv2.CV_64F,dx=0,dy=1)
            gradXX =np.square(gradX)
            gradYY =np.square(gradY)
            gradXY =gradXX*gradYY
            tr = gradXX + gradYY
            det = gradXX * gradYY - gradXY ** 2
            response = ( tr**2 +0.00000001 ) / (det+0.00000001)
            
            corners = list(map( tuple , np.argwhere( response < hess_threshold ).tolist() ))
           
            #prune low contrast
            dog_norm =  dog_image/im_max
            high_contrast = list(map( tuple , np.argwhere( np.abs( dog_norm ) > contrast_threshold ).tolist() ))
            
            #non-border pixels only
            non_borders = set((i,j) for i in range(1, dog_image.shape[0] - 1) for j in range(1, dog_image.shape[1] - 1))

            #candidate pixels intersection
            search_pixels = non_borders & set(corners) & set(high_contrast)
            for i,j in search_pixels:
                down = octave_dogs[idx-1][i-1:i+2, j-1:j+2]
                mid = octave_dogs[idx][i-1:i+2, j-1:j+2]
                up = octave_dogs[idx+1][i-1:i+2, j-1:j+2]
                if is_extrema( down, mid, up ):
                    dog_keypoints.append((i,j))
                    count +=1

            octave_keypoints.append(dog_keypoints)
        octaves_keypoints.append(octave_keypoints)
       # print(count)

def keypoints_orientation( num_bins = 36) :
    new_keypoints = []
    bin_width =360//num_bins
    
    for octave in range(nb_octaves):
        octave_scales = octaves[octave]
        octave_keypoints = octaves_keypoints[octave]
        sigmas_ = np.multiply( 1.5,sigmas)
        radiuses = [(2*int(round(s))) for s in sigmas_]

        for scale_idx in range(len(octave_keypoints)):
            scale_keypoints = octave_keypoints[scale_idx]
            scale_img = octave_scales[ scale_idx ] 
            sigma = sigmas_[scale_idx]
            radius= radiuses[scale_idx]

            kernel = np.outer(cv2.getGaussianKernel(2 * radius + 1, sigma), cv2.getGaussianKernel(2 * radius + 1, sigma))
            gradX = cv2.Sobel(scale_img,-1,dx=1,dy=0)
            gradY = cv2.Sobel(scale_img,-1,dx=0,dy=1)
            mag = np.sqrt( gradX * gradX + gradY * gradY )
            dir = np.rad2deg( np.arctan2( gradY , gradX )) % 360

            #Sampling  the angle histogram to 36 bins
            direction_bins = np.array(np.floor(dir)//bin_width,dtype =np.int16)       
            for i,j in scale_keypoints :
                window = [i-radius, i+radius+1, j-radius, j+radius+1]
                mag_win = slice( mag , window )
                dir_idx = slice( direction_bins, window )
                weight = mag_win * kernel 
                hist = np.zeros(num_bins, dtype=np.float32)
                
                # hist = np.bincount(dir_idx.ravel())
                # hist=hist*weight
                for bin_idx in range(num_bins):
                    hist[bin_idx] = np.sum( weight[ dir_idx == bin_idx ] )
            
                for bin_idx in np.argwhere( hist == hist.max() ).tolist():
                    angle = (bin_idx[0]+0.5) * (360./num_bins) % 360
                    new_keypoints.append( (i,j,octave,scale_idx,angle))

    return new_keypoints
    
def sift_descriptors( keypoints, num_bins = 8 ):
    features = []
    kps=[]
    infos = {}
    bin_width =360//num_bins

    for i,j,oct,sc, orientation in keypoints :
        #to calculate scale gradient only one
        if 'index' not in infos or infos['index'] != (oct,sc):
            infos['index'] = (oct,sc)
            octave_scales = octaves[oct]
            sigma = 1.5*sigmas[sc]
            scale_img = octave_scales[ sc ] 

            infos['kernel'] = np.outer(cv2.getGaussianKernel(16, sigma), cv2.getGaussianKernel(16, sigma))
            gradX = cv2.Sobel(scale_img,-1,dx=1,dy=0)
            gradY = cv2.Sobel(scale_img,-1,dx=0,dy=1)
            infos['mag'] = np.sqrt( gradX * gradX + gradY * gradY )
            infos['dir'] = np.rad2deg( np.arctan2( gradY , gradX )) % 360

        #Sampling the angle histogram to 8 bins
        radius=8
        features_desc = []
        window = [i-radius, i+radius, j-radius, j+radius]
        feature_mag_win = slice( infos['mag'] , window )
        
        #weighted gradients magnitude and by gaussian kernel
        feature_mag_weighted = feature_mag_win*infos['kernel']
        
        #subtract orientation of key point 
        feature_phase = slice( infos['dir'], window )
        featurePhaseAdjusted = feature_phase - orientation
        feature_phase_bins  = np.array(np.floor(featurePhaseAdjusted)//bin_width,dtype =np.int16)

        #angle histogram 
        for x in range(0, 13, 4):
            for y in range(0, 4):
                featurePhaseQuad = feature_phase_bins[x:x + 4, 4 * y: 4 * (y + 1)]
                featureMagnitudeWeightedQuad = feature_mag_weighted[x:x + 4, 4 * y: 4 * (y + 1)]
                featureVector = angle_histogram(featurePhaseQuad, featureMagnitudeWeightedQuad, 8)
                features_desc = features_desc + featureVector      
        
        if max(features_desc) != 0:
            features_desc /= max(features_desc)
            features.append(features_desc)
            kps.append((i,j,oct,sc, orientation))

    
    return kps , features

def drawKeypoints(image, keypoints ):
    copy = image.copy()
    copy = resize(copy,octaves[0][0].shape)
    plt.figure(figsize=(40,40)) 
    plt.axis('off')
    for x, y, octave_idx, scale_idx, angle in keypoints:
        #adjust x,y, r depending on location in scale space
        radius = sigma * ( 2 ** octave_idx ) * ( K ** (scale_idx)) *3
        y *= 2 ** (octave_idx)
        x *= 2 ** (octave_idx)
        cv2.circle(copy, (y, x), int(round(radius)), (0,255,255), 1)
        cv2.circle(copy, (y, x), 2, (0,255,0), 3)

    plt.imshow( copy)
    plt.show()


def sift(input_mg,contrast_threshold=0.06):
    img_color = cv2.imread(input_mg )
    img_color=cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_grey  = cv2.imread(input_mg, cv2.IMREAD_GRAYSCALE )

    start = time.time()
    scale_space(img_grey,show=False)
    locate_keypoints(hess_ratio=10, im_max=img_grey.max(),contrast_threshold=contrast_threshold)
    key_points = keypoints_orientation()

    drawKeypoints(img_color,key_points)
    
    kps , feature_descs= sift_descriptors(key_points,8)
    end = time.time()

    print(feature_descs)

    print(end-start)
    return kps ,feature_descs ,end-start


    return kps ,feature_descs ,end-start
    
#sift("pic\lena.jpg")
#sift("pic\lena_rot.jpg")
#sift("pic\Cow1.jpg",0.07)
#sift("pic\Cow2.jpg",0.065)

sift("pic\Cow2.jpg",0.035)


