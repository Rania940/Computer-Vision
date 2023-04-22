#include "hybrid.h"

cv::Mat hybridize(cv::Mat image1, cv::Mat image2)
{
	//image1 = ToGreyScale(image1);
	//image2 = ToGreyScale(image2);

	cv::Mat resized_im1, resized_im2, hybrid_image;

	//first resize the 2 images
	cv::resize(image1, resized_im1, cv::Size(500, 500));
	cv::resize(image2, resized_im2, cv::Size(500, 500));



	// Low-pass filter on first image
	resized_im1 = GaussianFilter(resized_im1, 23, 15);
	//cv::imshow("low", resized_im1);

	//std::cout << resized_im1.rows << std::endl;
	//std::cout << resized_im1.cols << std::endl;
	//std::cout << resized_im1.channels() << std::endl;

	// Get high frequencies of second image (or get edges)
	cv::Mat low_image2 = GaussianFilter(resized_im2, 35, 15);
	cv::Mat high_image2;
	cv::subtract(resized_im2, low_image2, high_image2);

	//std::cout << high_image2.rows << std::endl;
	//std::cout << high_image2.cols << std::endl;
	//std::cout << high_image2.channels() << std::endl;
	//cv::imshow("high", high_image2);

	/*
	cv::Mat mag_im2 = cv::Mat::Mat(resized_im2.rows, resized_im2.cols, CV_8UC1);
	cv::Mat angles_im2 = cv::Mat::Mat(resized_im2.rows, resized_im2.cols, CV_64FC3);
	DetectEdges("Sobel", resized_im2, mag_im2, angles_im2);
	

	std::cout << mag_im2.rows << std::endl;
	std::cout << mag_im2.cols << std::endl;
	std::cout << mag_im2.channels() << std::endl;


	cv::imshow("high", mag_im2);
	*/


	//cv::waitKey(0);

	// adding the 2 images together
	cv::add(resized_im1, high_image2, hybrid_image);
	//cv::add(resized_im1, mag_im2, hybrid_image);


	return hybrid_image;
}
