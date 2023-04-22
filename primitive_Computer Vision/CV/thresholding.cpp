#include "thresholding.h"

using namespace cv;
using namespace std;
Mat thresh1, thresh2, thresh3, thresh4, thresh5, blurred;

/*void adaptiveThresholdingSegmentation(Mat& image, const int kernelSize)
{
	Mat tempImage;

	if (image.channels() != 1)
	{
		cvtColor(image, tempImage, COLOR_BGR2GRAY);

		tempImage.copyTo(image);
	}
	else
	{
		image.copyTo(tempImage);
	}
	int totalKernelElements = kernelSize * kernelSize;

	vector<double> kernel(totalKernelElements, 1.0 / totalKernelElements);

	vector<double> values;

	int halfSize{ kernelSize / 2 };

	for (int i{ halfSize }; i < image.rows - halfSize; i++)
	{
		for (int j{ halfSize }; j < image.cols - halfSize; j++)
		{
			values.clear();

			for (int x = { -halfSize }; x <= halfSize; x++)
			{
				for (int y = { -halfSize }; y <= halfSize; y++)
				{
					unsigned char* pixelValuePtr = image.ptr(i + x) + (j + y);

					values.push_back(*pixelValuePtr);
				}
			}

			long averageValue = std::inner_product(begin(values), end(values),begin(kernel), 0.0);

			unsigned char* pixelValuePtr = image.ptr(i) + j;

			*pixelValuePtr = *pixelValuePtr > averageValue ? 0 : 255;
		}
	}
	cv::imshow("adaptiveThresholdingSegmentation", image);
	cv::imwrite("adaptiveThresholdingSegmentation", image);
}
*/

void SimpleThresholding(Mat& image) 
{
	Mat thresh1, thresh2, thresh3, thresh4, thresh5;
	// First type of Simple Thresholding is Binary Thresholding
	// After thresholding the image with this type of operator, we will
	// have image with only two values, 0 and 255.
	threshold(image, thresh1, 127, 255, THRESH_BINARY);

	// Inverse binary thresholding is just the opposite of binary thresholding.
	threshold(image, thresh2, 127, 255, THRESH_BINARY_INV);

	// Truncate Thresholding is type of thresholding where pixel
	// is set to the threshold value if it exceeds that value.
	// Othervise, it stays the same.
	threshold(image, thresh3, 127, 255, THRESH_TRUNC);

	// Threshold to Zero is type of thresholding where pixel value stays the same
	// if it is greater than the threshold. Otherwise it is set to zero.
	threshold(image, thresh4, 127, 255, THRESH_TOZERO);

	// Inverted Threshold to Zero is the opposite of the last one.
	// Pixel value is set to zero if it is greater than the threshold.
	// Otherwise it stays the same.
	threshold(image, thresh5, 127, 255, THRESH_TOZERO_INV);
	// Displaying the result
	cv::imshow("THRESH_BINARY", thresh1);
	cv::imshow("THRESH_BINARY_INV", thresh2);
	cv::imshow("THRESH_TRUNC", thresh3);
	cv::imshow("THRESH_TOZERO", thresh4);
	cv::imshow("THRESH_TOZERO_INV", thresh5);

}

void AdaptiveThresholding(Mat& image) {

	adaptiveThreshold(image, thresh1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
	adaptiveThreshold(image, thresh2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

	// Displaying the result
	cv::imshow("Adaptive Mean Thresholding", thresh1);
	cv::imshow("Adaptive Gaussian Thresholding", thresh2);


}
void OtsuThresholding(Mat& image) {
	// Otsu's thresholding
	threshold(image, thresh1, 0, 255, THRESH_BINARY + THRESH_OTSU);

	// Otsu's thresholding with Gaussian filtering
	GaussianBlur(image, blurred, Size(5, 5), 0);
	threshold(blurred, thresh2, 0, 255, THRESH_BINARY + THRESH_OTSU);

	// Displaying the result
	cv::imshow("Otsu's thresholding", thresh1);
	cv::imshow("Otsu's thresholding with Gaussian filtering", thresh2);


}

