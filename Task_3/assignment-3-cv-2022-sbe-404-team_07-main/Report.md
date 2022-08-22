[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7719704&assignment_repo_type=AssignmentRepo)

# *Submitted by*:
|              Name              |   Sec. | B.N.|
|:------------------------------:|:------:|:---:|
| Aya Abdullah Farag             |    1   | 19
| Aya Mohamed Abdulrazzaq        |    1   | 20
| Rania Atef Omar                |    1   | 31 
| Salma Haytham                  |    1   | 37
| Nouran Khaled                  |    2   | 41

contact email: nkhaledsoliman@gmail.com
____________________________
</br>
</br>

# 1- Harris corner detection

## *Outputs Samples* :
<p float="left">
  <img src="pic/harris1.PNG" width="200" />
  <img src="pic/harris2.PNG" width="200" /> 
  <img src="pic/harris3.PNG" width="200" /> 
  <img src="pic/harris4.PNG" width="200" /> 
</p>

# 2- SIFT 

##main_functions:
1. scale_space() constructs the scale space of the image and the DOGs images of each octave 
2. locate_keypoints() locate keypoints from dog images in each octave and filter out low contrast and edge pixels
3. keypoints_orientation () appends orientation angle for keypoints
4. sift_descriptors() generate feature descriptor vectors for the keypoints

## *Outputs Samples* :
<img src="pic/cow1_kp.png"/> 

### lowering contrast threshold , more keypoints are added
<p float="left">
  <img src="pic/cow2_kp.png" width="400"/> 
  <img src="pic/cow2_low_th.png"width="400" /> 
</p>

<p float="left">
  <img src="pic/high_sift.png"width="400" /> 
  <img src="pic/lwth_sift.png" width="400"/> 
</p>

## Harris vs SIFT

#### 5.5 seconds - 7.2 secons
<p float="left">
  <img src="pic/lena_harris.png" width="400" /> 
  <img src="pic/lena_sift.png" width="400"/> 
</p>


# 3- Feature_Matching

##main_functions:
1. sum squared difference() to minimize the difference 
2. normalized cross correlation() to maximize the corrrelation
3. apply_feature_matching() to Create a cv2.DMatch object 

## *Outputs Samples* :
<p float="left">
  <img src="pic/feature_matching_results.PNG" /> 
</p>



