[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7226721&assignment_repo_type=AssignmentRepo)

# *Submitted by*:
|              Name              |   Sec. | B.N.|
|:------------------------------:|:------:|:---:|
| Aya Abdullah Farag             |    1   | 19
| Aya Mohamed Abdulrazzaq        |    1   | 20
| Rania Atef Omar                |    1   | 31 
| Salma Haytham                  |    1   | 37
| Nouran Khaled                  |    2   | 41

____________________________
</br>
</br>

# 1-Adding additive noise 
Noise is generally considered to be a random variable . Consider a noisy pixel, p=p0+n where p0 is the true value of pixel and n is the noise in that pixel. 
Here, we add three basic types of noise that are common in image processing applications:

- Gaussian noise.
- Uniform noise
- Salt and Pepper noise 

## *Outputs Samples* :
### Gaussian noise 
Gaussian noise has a uniform distribution throughout the signal. A noisy image has pixels that are made up of the sum of their original pixel values plus a random Gaussian noise value. The probability distribution function for a Gaussian distribution has a bell shape.
This type of noise adds more Noise to the midtones and less noise to the shadows and highlight regions of the image.
<p float="left">
  <img src="pic/gaussian.PNG" width="275" />
  <img src="pic/gaussian2.PNG" width="275" /> 
  <img src="pic/gaussian3.PNG" width="275" />
</p>

### Uniform noise
 As the name suggests, using this option adds random colour noise of equal intensity all over the image. it means the different “values” of noise are equally probably.

<p float="left">
  <img src="pic/uniform.PNG" width="275" />
  <img src="pic/uniform2.PNG" width="275" /> 
  <img src="pic/uniform3.PNG" width="275" />
</p>

### Salt and Pepper noise 
A type of noise commonly seen in photographs is salt and pepper noise. It manifests as white and black pixels that appear at random intervals. 
<p float="left">
  <img src="pic/s&p.PNG" width="275" />
  <img src="pic/s&p_colored.jpg" width="275" /> 
</p>

# 2-Filters
Filter_Noise function smooth the image and reduce high frequency components (Noise). It implements : Average , Gaussian , Median filters with adjustable kernel size and standard deviation
## Common Algorithm:
- Creating kernel
- Zero padd the image 
- Convolve kernel with the image 

## *Outputs Samples* :
### Median filter applied on salt & pepper noise , with kernel size 3 and 5 
<p float="left">
  <img src="pic/s&p.PNG" width="275" />
  <img src="pic/MedianFilter 3.png" width="275" /> 
  <img src="pic/MedianFilter 5.png" width="275" />
</p>

### Gaussian filter kernel size = 5 , std = 1 // kernel size 3 , std = 5  
<p float="left">
  <img src="pic/Lenna.png" width="270" /> 
  <img src="pic/gaussian_filter.png" width="270" />
  <img src="pic/gaussian.PNG" width="270" /> 
  <img src="pic/GaussianFilter.png" width="277" />
</p>

### Average filter kernel size = 5 , std = 1 // kernel size 3 , std = 5  
<p float="left">
  <img src="pic/uniform2.PNG" width="275" /> 
  <img src="pic/Avg_filter.png" width="275" />
  <img src="pic/Avg2_filter.png" width="275" />

</p>

# 3- Edge Detection
DetectEdges function implements 4 Edge detection masks: Sobel , Roberts , Prewitt and Canny 
## Common Algorithm:
- Convert image to grey scale 
- Apply gaussian filter to smooth the image out
- Define kernels 
-Convolution 

## Canny edge detection:
### - canny filter works by:
- Convert image to grey scale 
- Apply gaussian filter 
- Apply Sobel filter
- Aply non max suppresion algorithm
- Double thresholding for defining weak and strong edges 
- Hysterisis 

## *Outputs Samples* :
### Canny:  
![output](pic/Hysterisis.png)

#### Canny Algorithm Output : Sobel , non max , thresholded image , hysterisis 
<p float="left">
  <img src="pic/Sobel.png" width="200" /> 
  <img src="pic/non_max.png" width="200" />
  <img src="pic/thresholded.png" width="200" />
  <img src="pic/Hysterisis.png" width="200" />

</p>

### Sobel:
![output](pic/Sobel.png)

### Roberts:
![output](pic/Roberts.png)

### Prewitt:
![output](pic/Prewitt.png)


# 4&8-Histogram

Breaking this up into 2 functions - imhist() and im_rgbhist(). Each would has a double loop where we add up values in the different vector color arrays and generate the Histogram Plot images, create the windows and show the images.

  ## *Output Samples*:

  ![output](pic/hany-gray.jpg)
  ![output](pic/hany-color.jpg)



# 5-Image_Equalization
histogram qualitation improves the contrast of images by clustering the pixels and flattinning the curve of histogram  -applying it in gray and color.

  ## *used functions*:
   1- imhist()
   2-cumhist()
   3-equalization_Algorithm_GRAYSCALE()
   4-equalization_Algorithm_COLOUR()

 ## *Outputs Samples* :

   <img src="pic/original_gray.PNG" width="280" /> 
   <img src="pic/eq_gray.PNG" width="280" /> 
   <img src="pic/original_color.PNG" width="280" /> 
   <img src="pic/equalize.PNG" width="280" /> 

   
# 6-Image Normalization
This is another technique to enhance contrast. It is done by stretching the histogram to take all values.

 ## *Output* ##
![](pic/norm.PNG)

# 7-Thresholding
Separate out regions of an image corresponding to objects which we want to analyze. This separation is based on the variation of intensity between the object pixels and the background pixels.
## *Outputs Samples* :

   <p float="left">
  <img src="pic/outsuthresholding with Gaussian filtering.PNG" width="275" />
  <img src="pic/outsuthresholding.PNG" width="275" /> 
  <img src="pic/adaptive gussian thresholding.PNG" width="275" />
  <img src="pic/adaptive mean thresholding.PNG" width="275" />
</p>

# 9-Frequency Domain Filters
 we can choose the desired filter and D0 by using *function Construct_H* which applied in the  Frequency_Domain Filters file .
  ## *Low pass filter*: H(u,v)
   reduces the high frequency content(blurring or smoothing)
   passes all frequencies withn a circle of radius from the origin and cut off all frequencies outside the circle.

  ## ideal_lowpass: D0 is a constant
  <img src="pic/ideal_lpf.PNG" width="280"/>   

  ## the equation D of ideal:
  <img src="pic/ideal_lowequation.PNG" width="280"/>

  ## Gaussian_lowpass: 
  <img src="pic/gaussian_lpf.PNG" width="280"/> 

  ## equation of gaussian:
  <img src="pic/LOW_Gaussian.PNG" width="280"/> 

  ## *High pass filter*:  1 - H(u,v)
  sharpening the images 


  ## *Outputs Samples* :

  ##original in GRayscale:

  <img src="pic/original_horse.PNG" width="280"/>  

  ##IDEAL_LOW ,,,, D0=30

  <img src="pic/ideallow30.PNG" width="280"/> 

  ##IDEAL_LOW ,,,, D0=85

  <img src="pic/ideal_low85.PNG" width="280"/>     

  ##GAUSSIAN_LOW ,,,, D0=30

  <img src="pic/gaussianlow30.PNG" width="280"/>    

  ##GAUSSIAN_LOW ,,,, D0=85

  <img src="pic/gaussianlow85.PNG" width="280"/>    
 
  ##IDEAL_HIGH ,,,, D0=15

  <img src="pic/idealhigh15.PNG" width="280"/>

  ##IDEAL_HIGH ,,,, D0=30

  <img src="pic/idealhigh30.PNG" width="280"/>  

  ##GAUSSIAN_HIGH ,,,, D0=15

  <img src="pic/gaussianhigh15.PNG" width="280"/>

  ##GAUSSIAN_HIGH ,,,, D0=30

  <img src="pic/gaussianhpf30.PNG" width="280"/>  

<br>
<br>

# 10- Hybrid Images

 ## *First Image* ##
 ![](pic/marylin.PNG)

 ## *Second Image* ##
 ![](pic/eins.PNG)

 ## *Zoomed In Output* ##
 ![](pic/hybrid_eins.PNG)

 ## *Zoomed Out Output* ##
 ![](pic/hybrid_marylin.PNG)






 
  







