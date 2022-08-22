#ifndef filter_h
#define filter_h
#include "utils.h"

cv::Mat AvgFilter(cv::Mat src, int  kernel_size);

cv::Mat GaussianFilter(cv::Mat src, int  kernel_size, float sigma);

cv::Mat MedianFilter(cv::Mat src, int  kernel_size);


cv::Mat Filter_Noise(std::string filter_name, cv::Mat src, int kernel_size, float sigma = 1.0);


#endif // !filter_h
