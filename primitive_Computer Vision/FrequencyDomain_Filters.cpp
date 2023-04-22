#include "FrequencyDomain_Filters.h"



void fftshift(const Mat& input_img, Mat& output_img)
{
	output_img = input_img.clone();
	int cx = output_img.cols / 2;
	int cy = output_img.rows / 2;
	Mat q1(output_img, Rect(0, 0, cx, cy));
	Mat q2(output_img, Rect(cx, 0, cx, cy));
	Mat q3(output_img, Rect(0, cy, cx, cy));
	Mat q4(output_img, Rect(cx, cy, cx, cy));

	Mat temp;
	q1.copyTo(temp);
	q4.copyTo(q1);
	temp.copyTo(q4);
	q2.copyTo(temp);
	q3.copyTo(q2);
	temp.copyTo(q3);
}


void calculateDFT(Mat& scr, Mat& dst)
{
	// define mat consists of two mat, one for real values and the other for complex values
	Mat planes[] = { scr, Mat::zeros(scr.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);
	dst = complexImg;
}


Mat construct_H(Mat& scr, String type, float D0)
{
	Mat H(scr.size(), CV_32F, Scalar(1));
	float D_L = 0;
	
	if (type == "Ideal_LPF")
	{
		for (int u = 0; u < H.rows; u++)
		{
			for (int v = 0; v < H.cols; v++)
			{
				D_L = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
				if (D_L > D0)
				{
					H.at<float>(u, v) = 0;
				}
			}
		}
		return H;
	}
	if (type == "Ideal_HPF")
	{
		for (int u = 0; u < H.rows; u++)
		{
			
			for (int v = 0; v < H.cols; v++)
			{
				D_L = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
				if (D_L > D0)
				{
					H.at<float>(u, v) = 0;
				}
				
			}
		}
		return (1 - H);
	}
	
	
	else if (type == "Gaussian_low")
	{
		for (int u = 0; u < H.rows; u++)
		{
			for (int v = 0; v < H.cols; v++)
			{
				D_L = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
				H.at<float>(u, v) = exp(-D_L * D_L / (2 * D0 * D0));
			}
		}
		return H;
	}
	else if (type == "Gaussian_high")
	{
		for (int u = 0; u < H.rows; u++)
		{
			for (int v = 0; v < H.cols; v++)
			{
				D_L = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
				H.at<float>(u, v) = exp(-D_L * D_L / (2 * D0 * D0));
			}
		}
		return (1 - H);
	}
	
	
}



void filtering(Mat& scr, Mat& dst, Mat& H)
{
	fftshift(H, H);
	Mat planesH[] = { Mat_<float>(H.clone()), Mat_<float>(H.clone()) };

	Mat planes_dft[] = { scr, Mat::zeros(scr.size(), CV_32F) };
	split(scr, planes_dft);

	Mat planes_out[] = { Mat::zeros(scr.size(), CV_32F), Mat::zeros(scr.size(), CV_32F) };
	planes_out[0] = planesH[0].mul(planes_dft[0]);
	planes_out[1] = planesH[1].mul(planes_dft[1]);

	merge(planes_out, 2, dst);

}

void display_high_ideal()
{
	Mat imgIn = imread("horse1.JPG", 0);




	int down_width = 500;
	int down_height = 500;
    Mat resized_down_original;
	//resize down
	resize(imgIn,  resized_down_original, Size(down_width, down_height), INTER_LINEAR);


	imshow("Original", resized_down_original);
	imgIn.convertTo(imgIn, CV_32F);

	// DFT
	Mat DFT_image;
	calculateDFT(imgIn, DFT_image);

	// construct H(u,v)
	Mat H;
	//choose the desired filter here....& the desired D0 .....
	H = construct_H(imgIn, "Ideal_HPF", 35);

	// filtering
	Mat complexIH;
	filtering(DFT_image, complexIH, H);

	// IDFT
	Mat imgOut;
	dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);

	normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);


	Mat resized_down_result;
	//resize down
	resize(imgOut, resized_down_result, Size(down_width, down_height), INTER_LINEAR);

	// options depend on the desired filter

	
	imshow("Ideal_highpass", resized_down_result);
	




}
void display_low_ideal()
{
	Mat imgIn = imread("horse1.JPG", 0);




	int down_width = 500;
	int down_height = 500;
	Mat resized_down_original;
	//resize down
	resize(imgIn, resized_down_original, Size(down_width, down_height), INTER_LINEAR);


	imshow("Original", resized_down_original);
	imgIn.convertTo(imgIn, CV_32F);

	// DFT
	Mat DFT_image;
	calculateDFT(imgIn, DFT_image);

	// construct H(u,v)
	Mat H;
	//choose the desired filter here....& the desired D0 .....
	H = construct_H(imgIn, "Ideal_LPF", 35);

	// filtering
	Mat complexIH;
	filtering(DFT_image, complexIH, H);

	// IDFT
	Mat imgOut;
	dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);

	normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);


	Mat resized_down_result;
	//resize down
	resize(imgOut, resized_down_result, Size(down_width, down_height), INTER_LINEAR);

	// options depend on the desired filter

	imshow("Ideal_lowpass", resized_down_result);
	









}
void display_high_gaussian()
{
	Mat imgIn = imread("horse1.JPG", 0);




	int down_width = 500;
	int down_height = 500;
	Mat resized_down_original;
	//resize down
	resize(imgIn, resized_down_original, Size(down_width, down_height), INTER_LINEAR);


	imshow("Original", resized_down_original);
	imgIn.convertTo(imgIn, CV_32F);

	// DFT
	Mat DFT_image;
	calculateDFT(imgIn, DFT_image);

	// construct H(u,v)
	Mat H;
	//choose the desired filter here....& the desired D0 .....
	H = construct_H(imgIn, "Gaussian_high", 35);

	// filtering
	Mat complexIH;
	filtering(DFT_image, complexIH, H);

	// IDFT
	Mat imgOut;
	dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);

	normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);


	Mat resized_down_result;
	//resize down
	resize(imgOut, resized_down_result, Size(down_width, down_height), INTER_LINEAR);

	// options depend on the desired filter

	
	imshow("Gaussian_highpass", resized_down_result);




}

void display_low_gaussian()
{


	Mat imgIn = imread("horse1.JPG", 0);




	int down_width = 500;
	int down_height = 500;
	Mat resized_down_original;
	//resize down
	resize(imgIn, resized_down_original, Size(down_width, down_height), INTER_LINEAR);


	imshow("Original", resized_down_original);
	imgIn.convertTo(imgIn, CV_32F);

	// DFT
	Mat DFT_image;
	calculateDFT(imgIn, DFT_image);

	// construct H(u,v)
	Mat H;
	//choose the desired filter here....& the desired D0 .....
	H = construct_H(imgIn, "Gaussian_low", 35);

	// filtering
	Mat complexIH;
	filtering(DFT_image, complexIH, H);

	// IDFT
	Mat imgOut;
	dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);

	normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);


	Mat resized_down_result;
	//resize down
	resize(imgOut, resized_down_result, Size(down_width, down_height), INTER_LINEAR);

	// options depend on the desired filter

	
	imshow("Gaussian_lowpass", resized_down_result);
	

}












