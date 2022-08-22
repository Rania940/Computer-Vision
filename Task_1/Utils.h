#ifndef _Utils_
#define _Utils_


#include<iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;





//.....................image_equalization....................//

void imhist(Mat image, int histogram[]);
void cumhist(int histogram[], int cumhistogram[]);

int get_max(int arr[], int size);
int get_min(int arr[], int size);










#endif