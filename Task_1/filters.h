#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat AvgFilter(cv::Mat src, int  kernel_size);
cv::Mat GaussianFilter(cv::Mat src, int  kernel_size, float sigma = 1.0);
cv::Mat MedianFilter(cv::Mat src, int  kernel_size);
