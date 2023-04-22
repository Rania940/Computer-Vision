#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.h"
#include "filters.h"
#include "edge_detection.h"
#include "histogram.h"
#include "noise.h"
#include "FrequencyDomain_Filters.h"
#include "equalization.h"
#include "thresholding.h"
#include "hybrid.h"
#include "normalize.h"



int main() {
	
	////1-Add additive noise to the image

	//// Reading the image file
	//cv::Mat image_g = cv::imread("..\\pic\\Lenna.png");
	//cv::Mat image_u = cv::imread("..\\pic\\Lenna.png");
	//cv::Mat image_s = cv::imread("..\\pic\\Lenna.png" );


	//resize(image_g, image_g, { 500,500 });
	//resize(image_u, image_g, { 500,500 });
	//resize(image_s, image_g, { 500,500 });


	////if (image.empty())
	///* {
	//	cout << "Could not read the image:" << endl;
	//	return 1;
	//}*/
	////1......................................
	//	//(Add gaussian noise to image)
	//gaussianNoise(image_g, 128, 20);
	////(Add uniform noise to image)
	//uniformNoise(image_u, 20, 128);
	////(Add salt and pepper noise to image after add gaussian noise to image )
	//saltAndPepperNoise(image_s, 0.05, 0.05);





	//2-Filters
	// // Read the image file
	//cv::Mat image = cv::imread("..\\pic\\lenna.png");
	// Show our image inside a window.
	//cv::imshow("original", image);
	//cv::Mat Gaussian = Filter_Noise("Gaussian", image, 3, 20);
	//cv::Mat Avg = Filter_Noise("Avg", image, 3);
	//cv::Mat Median = Filter_Noise("Median", image, 3);
	//cv::imshow("Avg", Avg);


	//3-Edge detectors
	// // Read the image file
	//cv::Mat image = cv::imread("..\\pic\\lenna.png");
	// Show our image inside a window.
	//cv::imshow("original", image);
	//cv::Mat mag = cv::Mat::Mat(image.rows, image.cols, CV_8UC1);
	//cv::Mat angles = cv::Mat::Mat(image.rows, image.cols, CV_64FC3);
	//DetectEdges("Sobel", image , mag , angles);
	//DetectEdges("Prewitt", image, mag, angles);
	//DetectEdges("Roberts", image, mag, angles);
	//DetectEdges("Canny", image, mag, angles);
	//cv::imshow("Canny", mag);


	////3-histogram
	// // Read the image file
	//cv::Mat image = cv::imread("..\\pic\\lenna.png");
	// 
	////for greyscale image add 0
	//if (image.channels() == 3) {
	//	im_rgbhist(image);
	//}
	//else if (image.channels() == 1) {
	//	int * his  = imhist(image);
	//}



	////q5. show the difference between the two images and effect of equalization histogram on the contrast of the image:
	//// Display the original Image
	////load the image in grayscale
	//Mat image = imread("..\\pic\\lenna.png", cv::ImreadModes::IMREAD_GRAYSCALE);

	////load the image in colorscale
	//Mat image1 = imread("..\\pic\\lenna.png");

	//namedWindow("Original gray Image");
	//imshow("Original gray Image", image);

	//namedWindow("Original color Image");
	//imshow("Original color Image", image1);



	//// Display the equalized  gray image 
	//Mat new_image = equalization_Algorithm_GRAYSCALE(image);
	//namedWindow("Equilized GRAY Image");
	//imshow("Equilized GRAY Image", new_image);


	//// Display the equalized color image
	//Mat newcolor_image = equalization_Algorithm_COLOUR(image1);
	//namedWindow("Equilized COLOR Image");
	//imshow("Equilized COLOR Image", newcolor_image);





	//Q6 - Image Normalization
	//Mat image_norm = imread("..\\pic\\lenna.png");
	//normalize_image(image_norm);
	//imshow("Normalized", image_norm);



	//q-9.....//frequencydomain_filters......//

	//
	//Mat imgIn = imread("..\\pic\\Lenna.png", 0);


	//int down_width = 500;
	//int down_height = 500;

	//Mat resized_down_original;
	////resize down
	//resize(imgIn, resized_down_original, Size(down_width, down_height), INTER_LINEAR);


	//imshow("Original", resized_down_original);
	//imgIn.convertTo(imgIn, CV_32F);

	//// DFT
	//Mat DFT_image;
	//calculateDFT(imgIn, DFT_image);

	//// construct H(u,v)
	//Mat H;
	////choose the desired filter here....& the desired D0 .....
	//H = construct_H(imgIn, "Gaussian_high", 35);

	//// filtering
	//Mat complexIH;
	//filtering(DFT_image, complexIH, H);

	//// IDFT
	//Mat imgOut;
	//dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);

	//normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);


	//Mat resized_down_result;
	////resize down
	//resize(imgOut, resized_down_result, Size(down_width, down_height), INTER_LINEAR);

	//// options depend on the desired filter

	////imshow("Ideal_lowpass", resized_down_result);
	////imshow("Ideal_highpass", resized_down_result);
	////imshow("Gaussian_lowpass", resized_down_result);
	//imshow("Gaussian_highpass", resized_down_result);


	////Thresholding
	//// Reading the image file
	//cv::Mat image = cv::imread("..\\pic\\lenna.png", IMREAD_GRAYSCALE);
	//resize(image, image, { 500,500 });

	//if (image.empty())
	//{
	//	std::cout << "Could not read the image:" << std::endl;
	//	return 1;
	//}

	////adaptiveThresholdingSegmentation(image,31);
	//SimpleThresholding(image);
	//AdaptiveThresholding(image);
	//OtsuThresholding(image);
	//// Wait for any keystroke
	//cv::waitKey(0);





	//Q10 - Hybrid Image

	cv::Mat hybrid_image1 = cv::imread("../pic/marylin.png");
	cv::Mat hybrid_image2 = cv::imread("../pic/eins.png");
	cv::Mat hybrid_image = hybridize(hybrid_image1, hybrid_image2);
	cv::Mat resized;


	int size = 60;
	cv::resize(hybrid_image, resized, cv::Size(size, size), cv::INTER_LINEAR);

	size = 1000;
	cv::resize(hybrid_image, hybrid_image, cv::Size(size, size), cv::INTER_LINEAR);


	cv::imshow("Hybrid Image Zoomed in", hybrid_image);

	cv::imshow("Hybrid Image Zoomed out", resized);



	// Wait for any keystroke in the window
	cv::waitKey(0);
	return 0;

}
