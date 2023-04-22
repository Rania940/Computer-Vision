#ifndef noise_h
#define noise_h

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <random>
#include <cstdlib>

#include "utils.h"

//using namespace cv;
//using namespace std;

//1-Add additive noise to the image


void gaussianNoise(cv::Mat& image, const unsigned char mean, const unsigned char sigma);
void saltAndPepperNoise(cv::Mat& image, float saltProbability, float pepperProbability);
void uniformNoise(cv::Mat& image, const unsigned char a, const unsigned char b);

#endif
