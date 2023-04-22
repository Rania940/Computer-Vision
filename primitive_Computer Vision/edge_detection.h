#include <opencv2/opencv.hpp>
#include <iostream>

void Sobel(cv::Mat src, cv::Mat& dst, cv::Mat& angles_sobel, int gauss_size = 3, float sigma = 1.0);
void Prewitt(cv::Mat src, cv::Mat& dst, cv::Mat& angles_prewitt, int gauss_size = 3, float sigma = 1.0);
void Roberts(cv::Mat src, cv::Mat& dst, cv::Mat& angles_roberts, int gauss_size = 3, float sigma = 1.0);
