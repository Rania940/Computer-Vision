#include "equalization.h"
#include "Utils.h"





//5...............

 
Mat equalization_Algorithm_GRAYSCALE()
{
    //load the image in grayscale
    Mat image = imread("pic2.JPG", cv::ImreadModes::IMREAD_GRAYSCALE);


    
    
    //1.. Generate the histogram
    int histogram[256];
    imhist(image, histogram);
    // Caluculate the size of image
    int size = image.rows * image.cols;
    float alpha = 255.0 / size;
    // Calculate the probability of each intensity
    float Pr_int[256];
    for (int i = 0; i < 256; i++)
    {
        Pr_int[i] = (double)histogram[i] / size;
    }
    //2... Generate cumulative frequency histogram
    int cumhistogram[256];
    cumhist(histogram, cumhistogram);
    //3... Scale the histogram
    int scale[256];
    for (int i = 0; i < 256; i++)
    {
        scale[i] = cvRound((double)cumhistogram[i] * alpha);
    }
    //4... Generate the equalized histogram
    float eq[256];
    for (int i = 0; i < 256; i++)
    {
        eq[i] = 0;
    }
    for (int i = 0; i < 256; i++)
    {
        eq[scale[i]] += Pr_int[i];
    }
    int equalized_hist[256];
    for (int i = 0; i < 256; i++)
    {
        equalized_hist[i] = cvRound(eq[i] * 255);
    }
    //5... Generate the equalized image
    Mat equalized_image = image.clone();
    for (int y = 0; y < image.rows; y++)
        for (int x = 0; x < image.cols; x++)
            equalized_image.at<uchar>(y, x) = saturate_cast<uchar>(scale[image.at<uchar>(y, x)]);



    return equalized_image;

}

Mat equalization_Algorithm_COLOUR()
{
    Mat image1 = imread("pic2.JPG");

    // Convert the image from BGR to YCrCb color space
    Mat hist_equalized_image;
    cvtColor(image1, hist_equalized_image, COLOR_BGR2YCrCb);

    //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
    vector<Mat> vec_channels;
    split(hist_equalized_image, vec_channels);

    //1.. Generate the histogram
    int histogram[256];
    imhist(vec_channels[0], histogram);
    // Caluculate the size of image
    int size = vec_channels[0].rows * vec_channels[0].cols;
    float alpha = 255.0 / size;
    // Calculate the probability of each intensity
    float Pr_int[256];
    for (int i = 0; i < 256; i++)
    {
        Pr_int[i] = (double)histogram[i] / size;
    }
    //2... Generate cumulative frequency histogram
    int cumhistogram[256];
    cumhist(histogram, cumhistogram);
    //3... Scale the histogram
    int scale[256];
    for (int i = 0; i < 256; i++)
    {
        scale[i] = cvRound((double)cumhistogram[i] * alpha);
    }
    //4... Generate the equalized histogram
    float eq[256];
    for (int i = 0; i < 256; i++)
    {
        eq[i] = 0;
    }
    for (int i = 0; i < 256; i++)
    {
        eq[scale[i]] += Pr_int[i];
    }
    int equalized_hist[256];
    for (int i = 0; i < 256; i++)
    {
        equalized_hist[i] = cvRound(eq[i] * 255);
    }
    //5... Generate the equalized image1
    vec_channels[0] = vec_channels[0].clone();
    for (int y = 0; y < vec_channels[0].rows; y++)
        for (int x = 0; x < vec_channels[0].cols; x++)
            vec_channels[0].at<uchar>(y, x) = saturate_cast<uchar>(scale[vec_channels[0].at<uchar>(y, x)]);

    //merge the channels
    merge(vec_channels, hist_equalized_image);

    //Convert the histogram equalized image from YCrCb to BGR color space again
    cvtColor(hist_equalized_image, hist_equalized_image, COLOR_YCrCb2BGR);
    return hist_equalized_image;




}

void dispay_equalization()
{    //5. show the difference between the two images and effect of equalization histogram on the contrast of the image:
    // Display the original Image
    //load the image in grayscale
    Mat image = imread("pic2.JPG", cv::ImreadModes::IMREAD_GRAYSCALE);

    //load the image in colorscale
    Mat image1 = imread("pic2.JPG");

    namedWindow("Original gray Image");
    imshow("Original gray Image", image);

    namedWindow("Original color Image");
    imshow("Original color Image", image1);



    // Display the equalized  gray image 
    Mat new_image = equalization_Algorithm_GRAYSCALE();
    namedWindow("Equilized GRAY Image");
    imshow("Equilized GRAY Image", new_image);


    // Display the equalized color image
    Mat newcolor_image = equalization_Algorithm_COLOUR();
    namedWindow("Equilized COLOR Image");
    imshow("Equilized COLOR Image", newcolor_image);






}
