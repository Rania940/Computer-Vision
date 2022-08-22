#include "edge_detection.h"

void Sobel(cv::Mat src, cv::Mat& dst, cv::Mat& angles_sobel, int gauss_size = 3, float sigma = 1.0) {
	//convert to grey scale
	cv::Mat grey_img = ToGreyScale(src);
	//cv::cvtColor(src, grey_img, cv::COLOR_BGR2GRAY);	

	//gaussian flter to remove high freq comp
	grey_img = GaussianFilter(grey_img, gauss_size, sigma);

	//Sobel kernels
	std::vector<std::vector<double>> Gx = { {-1.0, 0.0, 1.0}, {-2.0, 0.0, 2.0}, {-1.0, 0.0, 1.0} };
	std::vector<std::vector<double>> Gy = { {1.0, 2.0, 1.0},{ 0.0, 0.0, 0.0},{ -1.0, -2.0, -1.0} };

	//horizontal and vertical conv
	cv::Mat out_h = Convolution(grey_img, Gx);
	cv::Mat out_v = Convolution(grey_img, Gy);
	double max = 0;

	//magnitude
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double sumh = (double)out_h.at<uchar>(i, j) * out_h.at<uchar>(i, j);
			double sumv = (double)out_v.at<uchar>(i, j) * out_v.at<uchar>(i, j);
			double mag = sqrt(sumh + sumv);
			double angle = atan2(sumv, sumh);
			max = mag > max ? mag : max;

			dst.at<uchar>(i, j) = (uchar)mag; //mag

			angles_sobel.at<double>(i, j) = angle > 0 ? (angle * (180.0 / 3.14159)) : ((angle + 2 * 3.14159) * (180.0 / 3.14159)); //angles in degree 0:360


		}
	}


}

void Prewitt(cv::Mat src, cv::Mat& dst, cv::Mat& angles_prewitt, int gauss_size = 5, float sigma = 1.0) {
	//convert to grey scale
	cv::Mat grey_img = ToGreyScale(src);

	//gaussian flter to remove high freq comp
	grey_img = GaussianFilter(grey_img, gauss_size, sigma);

	//Prewitt kernels
	std::vector<std::vector<double>> Gx = { {-1.0, 0, 1.0}, {-1.0, 0, 1.0}, {-1.0, 0, 1.0} };
	std::vector<std::vector<double>> Gy = { {1.0, 1.0, 1.0}, {0, 0, 0}, {-1.0, -1.0, -1.0} };

	//horizontal and vertical conv
	cv::Mat out_h = Convolution(grey_img, Gx);
	cv::Mat out_v = Convolution(grey_img, Gy);

	//magnitude
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double sumh = (double)out_h.at<uchar>(i, j) * out_h.at<uchar>(i, j);
			double sumv = (double)out_v.at<uchar>(i, j) * out_v.at<uchar>(i, j);
			double mag = sqrt(sumh + sumv);
			//thresholding
			//mag = (mag > 100) ? 255 : mag;
			dst.at<uchar>(i, j) = (uchar)mag;
			angles_prewitt.at<double>(i, j) = (atan(sumv / sumh)) * (180.0 / 3.14159); //angles in degree

		}
	}

}

void Roberts(cv::Mat src, cv::Mat& dst, cv::Mat& angles_roberts, int gauss_size = 5, float sigma = 1.0) {
	//convert to grey scale
	cv::Mat grey_img = ToGreyScale(src);

	//gaussian flter for noise reduction
	grey_img = GaussianFilter(grey_img, gauss_size, sigma);

	//Prewitt kernels
	std::vector<std::vector<double>> Gx = { {1.0, 0.0,0.0},{0.0,-1.0,0.0} ,{0,0,0} };
	std::vector<std::vector<double>> Gy = { {0.0, 1.0,0.0},{-1.0,0.0,0.0} ,{0,0,0} };

	//horizontal and vertical conv
	cv::Mat out_h = Convolution(grey_img, Gx);
	cv::Mat out_v = Convolution(grey_img, Gy);

	//magnitude
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double sumh = (double)out_h.at<uchar>(i, j) * out_h.at<uchar>(i, j);
			double sumv = (double)out_v.at<uchar>(i, j) * out_v.at<uchar>(i, j);
			double mag = sqrt(sumh + sumv);
			//thresholding
			//mag = (mag > 150) ? 255 : mag;
			dst.at<uchar>(i, j) = (uchar)mag;
			angles_roberts.at<double>(i, j) = (atan(sumv / sumh)) * (180.0 / 3.14159); //angles in degree
		}
	}

}


void NonMaxSuppression(cv::Mat non_max, cv::Mat& mag_sobel, cv::Mat angles_sobel) {

	double gradient_angle = 0;
	double before_val = 0;
	double after_val = 0;

	//non max suppression , loop for peaks 
	for (int i = 1; i < mag_sobel.rows - 1; i++)
	{
		for (int j = 1; j < mag_sobel.cols - 1; j++)
		{
			gradient_angle = angles_sobel.at<double>(i, j);

			//0
			if (((0 <= gradient_angle) && (gradient_angle < 22.5)) || ((337.5 <= gradient_angle) && (gradient_angle <= 360)))
			{
				before_val = mag_sobel.at<uchar>(i, j - 1);
				after_val = mag_sobel.at<uchar>(i, j + 1);
			}
			//45
			if (((22.5 <= gradient_angle) && (gradient_angle < 67.5)) || ((202.5 <= gradient_angle) && (gradient_angle < 247.5)))
			{
				before_val = mag_sobel.at<uchar>(i + 1, j - 1);
				after_val = mag_sobel.at<uchar>(i - 1, j + 1);
			}

			//90
			if (((67.5 <= gradient_angle) && (gradient_angle < 112.5)) || ((247.5 <= gradient_angle) && (gradient_angle < 292.5)))
			{
				before_val = mag_sobel.at<uchar>(i - 1, j);
				after_val = mag_sobel.at<uchar>(i - +1, j);
			}

			//135
			if (((112.5 <= gradient_angle) && (gradient_angle < 157.5)) || ((292.5 <= gradient_angle) && (gradient_angle < 337.5)))
			{
				before_val = mag_sobel.at<uchar>(i - 1, j - 1);
				after_val = mag_sobel.at<uchar>(i - +1, j + 1);
			}

			//non-max suppression (keep val zero if not max else get pixel value)
			if ((mag_sobel.at<uchar>(i, j) >= before_val) && (mag_sobel.at<uchar>(i, j) >= after_val)) {
				non_max.at<uchar>(i, j) = mag_sobel.at<uchar>(i, j);
			}
		}
	}

}


void DThresholding(cv::Mat threshold, cv::Mat& non_max, uchar l_th, uchar h_th) {

	for (int i = 0; i < non_max.rows; i++) {
		for (int j = 0; j < non_max.cols; j++) {
			//strong pixels
			if (non_max.at<uchar>(i, j) >= h_th) {
				threshold.at<uchar>(i, j) = 255;
			}
			//weak pixels
			else if (non_max.at<uchar>(i, j) >= l_th) {
				threshold.at<uchar>(i, j) = 60;  //weak is set to 60 
			}
			else if (non_max.at<uchar>(i, j) < l_th) {
				threshold.at<uchar>(i, j) = 0;
			}
			//<l_threshold are kept zero
			//std::cout << threshold.at<uchar>(i, j)<<"\n";
		}
	}
}


void Hysterisis(cv::Mat& edges, cv::Mat& threshold) {

	for (int i = 1; i < threshold.rows - 1; i++) {
		for (int j = 1; j < threshold.cols - 1; j++) {

			if ((threshold.at<uchar>(i, j)) == 60) {

				//check for strong neightbour in the 8 neighbours				
				if (((threshold.at<uchar>(i, j - 1)) == 255) || ((threshold.at<uchar>(i, j + 1)) == 255) ||
					((threshold.at<uchar>(i + 1, j)) == 255) || ((threshold.at<uchar>(i - 1, j)) == 255) ||
					((threshold.at<uchar>(i - 1, j + 1)) == 255) || ((threshold.at<uchar>(i + 1, j - 1)) == 255) ||
					((threshold.at<uchar>(i + 1, j + 1)) == 255) || ((threshold.at<uchar>(i - 1, j - 1)) == 255))
				{
					edges.at<uchar>(i, j) = 255;
				}
				else
				{
					edges.at<uchar>(i, j) = 0;
				}
			}

		}
	}

}


void Canny(cv::Mat src, cv::Mat& edges, cv::Mat& angles_sobel, uchar l_th = 20, uchar h_th = 70, int gauss_size = 5, float sigma = 1) {

	//Sobel filtering
	cv::Mat mag_sobel = cv::Mat::Mat(src.rows, src.cols, CV_8UC1);
	Sobel(src, mag_sobel, angles_sobel, gauss_size, sigma);
	//cv::imshow("Sobel", mag_sobel);

	//non max suppresion 
	//init with zeros
	cv::Mat non_max = cv::Mat::zeros(mag_sobel.rows, mag_sobel.cols, CV_8UC1);
	NonMaxSuppression(non_max, mag_sobel, angles_sobel);
	//cv::imshow("non_max", non_max);

	//double thresholding 
	cv::Mat threshold = cv::Mat::zeros(non_max.rows, non_max.cols, CV_8UC1);
	DThresholding(threshold, non_max, l_th, h_th);
	//cv::imshow("thresholded", threshold);

	//Hysteresis
	//Hysterisis(edges, threshold);
	edges = threshold.clone();
	Hysterisis(edges, threshold);
	//cv::imshow("Hysterisis", edges);

}


void DetectEdges(std::string filter_name, cv::Mat src, cv::Mat& dst, cv::Mat& angles, int gauss_size, float sigma, int l_th, int h_th) {

	if (filter_name == "Sobel") {
		return Sobel(src, dst, angles, gauss_size = 5, sigma = 1);
	}

	else if (filter_name == "Prewitt") {
		return Prewitt(src, dst, angles, gauss_size = 5, sigma = 1);
	}

	else if (filter_name == "Roberts") {
		return Roberts(src, dst, angles, gauss_size = 5, sigma = 1);
	}

	else if (filter_name == "Canny") {
		return Canny(src, dst, angles, l_th = 20, h_th = 70, gauss_size = 5, sigma = 1);
	}
}


