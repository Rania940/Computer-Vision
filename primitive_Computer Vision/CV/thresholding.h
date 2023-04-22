#ifndef th_h
#define th_h

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <random>
#include <cstdlib>
#include <vector>
#include <numeric> // std::inner_product

using namespace cv;



void SimpleThresholding(Mat& image);

void AdaptiveThresholding(Mat& image);

void OtsuThresholding(Mat& image);


#endif // !
