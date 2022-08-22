#include "filters.h"

cv::Mat AvgFilter(cv::Mat src, int  kernel_size) {
	//create kernal 
	std::vector<std::vector<double>> kernel = Kernel2D(kernel_size, 0);

	//convolve filter with img 
	cv::Mat convolved = Convolution(src, kernel);

	return convolved;
}

cv::Mat GaussianFilter(cv::Mat src, int  kernel_size, float sigma = 1.0) {
	//create kernal 
	std::vector<std::vector<double>> kernel = Kernel2D(kernel_size, sigma);

	//convolve filter with img 
	cv::Mat convolved = Convolution(src, kernel);

	return convolved;
}

cv::Mat MedianFilter(cv::Mat src, int  kernel_size) {

	cv::Mat img = src.clone();
	img.convertTo(img, CV_64FC(img.channels()));

	//zero padding img  
	cv::Mat img_padded = ZeroPadding(img, kernel_size);

	//initialize convolved mat
	cv::Mat out = cv::Mat::Mat(img.rows, img.cols, CV_64FC(img.channels()));


	int padd_size = (kernel_size - 1) / 2;
	std::cout << img.channels() << "\n";

	std::vector<cv::Mat> channels;
	split(img_padded, channels);

	//loop to get kernel from img
	cv::Mat kernel;
	for (int i = padd_size; i < img_padded.rows - padd_size; i++) {
		for (int j = padd_size; j < img_padded.cols - padd_size; j++) {
			for (int c = 0; c < img_padded.channels(); c++) {
				// extract kernel as sub matrix
				kernel = channels[c](cv::Range(i - padd_size, i - padd_size + kernel_size - 1), cv::Range(j - padd_size, j - padd_size + kernel_size - 1));
				// get median of the kernel
				double median = Median(kernel);

				// Assign median 
				if (img_padded.channels() > 1) {
					out.at<cv::Vec3d>(i - padd_size, j - padd_size)[c] = median;
				}
				else {
					out.at<double>(i - padd_size, j - padd_size) = median;
				}
			}
		}
	}
	out.convertTo(out, CV_8UC(img.channels()));
	return out;

}


cv::Mat Filter_Noise(std::string filter_name, cv::Mat src, int kernel_size, float sigma) {
	if (filter_name == "Avg") {
		return AvgFilter(src, kernel_size);
	}

	else if (filter_name == "Gaussian") {
		return GaussianFilter(src, kernel_size, sigma = 1);
	}

	else if (filter_name == "Median") {
		return MedianFilter(src, kernel_size);
	}
}
