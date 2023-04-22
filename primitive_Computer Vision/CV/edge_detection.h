#ifndef edge_h
#define edge_h
#include "utils.h"
#include "filters.h"



void Sobel(cv::Mat src, cv::Mat& dst, cv::Mat& angles_sobel, int gauss_size, float sigma);

void Prewitt(cv::Mat src, cv::Mat& dst, cv::Mat& angles_prewitt, int gauss_size, float sigma);


void Roberts(cv::Mat src, cv::Mat& dst, cv::Mat& angles_roberts, int gauss_size, float sigma);


void NonMaxSuppression(cv::Mat non_max, cv::Mat& mag_sobel, cv::Mat angles_sobel);

void DThresholding(cv::Mat threshold, cv::Mat& non_max, uchar l_th, uchar h_th);


void Hysterisis(cv::Mat& edges, cv::Mat& threshold);


void Canny(cv::Mat src, cv::Mat& edges, cv::Mat& angles_sobel, uchar l_th, uchar h_th, int gauss_size, float sigma);

void DetectEdges(std::string filter_name, cv::Mat src, cv::Mat& dst, cv::Mat& angles, int gauss_size = 5, float sigma = 1.0, int l_th = 20, int h_th = 70);




#endif