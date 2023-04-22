#ifndef _FrequencyDomain_Filters_
#define _FrequencyDomain_Filters_



#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;


void calculateDFT(Mat& scr, Mat& dst);
Mat construct_H(Mat& scr, String type, float D0);
void filtering(Mat& scr, Mat& dst, Mat& H);
void fftshift(const Mat& input_img, Mat& output_img);
void display_high_ideal();
void display_low_ideal();
void display_high_gaussian();
void display_low_gaussian();









#endif

