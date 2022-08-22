#include "Utils.h"






//5....................image_equalization............
//1. get histogram of te grayscale image 
void imhist(Mat image, int histogram[])
{
    // initialize all intensity values to 0
    for (int i = 0; i < 256; i++)
    {
        histogram[i] = 0;
    }
    // calculate the no of pixels for each intensity values
    for (int y = 0; y < image.rows; y++)
        for (int x = 0; x < image.cols; x++)
            histogram[(int)image.at<uchar>(y, x)]++;
}
//2. cumulative distribution function (CDF)
void cumhist(int histogram[], int cumhistogram[])
{
    cumhistogram[0] = histogram[0];
    for (int i = 1; i < 256; i++)
    {
        cumhistogram[i] = histogram[i] + cumhistogram[i - 1];
    }
}






int get_max(int arr[], int size)
{
	int max = 0;
	for (int i = 0; i < size; i++) 
	{
		if (arr[i] > max)
			max = arr[i];
	}
	return max;
}


int get_min(int arr[], int size)
{
	int min = 0;
	for (int i = 0; i < size; i++)
	{
		if (arr[i] < min)
			min = arr[i];
	}
	return min;
}

