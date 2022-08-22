#ifndef utils_h
#define utils_h

#include <opencv2/opencv.hpp>
#include <iostream>



cv::Mat ToGreyScale(cv::Mat src);

cv::Mat ZeroPadding(cv::Mat img, int kernal_size);

std::vector<std::vector<double>>  Kernel2D(int kernel_size, float sigma);

double Median(cv::Mat kernel_elements);

cv::Mat Convolution(cv::Mat src, std::vector<std::vector<double>>  kernel);

int get_max(int arr[], int size);

int get_min(int arr[], int size);

#endif // !utils_h
