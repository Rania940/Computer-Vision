#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void imhist(Mat grey_image)
{
    int Hist[256] = {0};
 

    for(int y = 0; y < grey_image.rows; y++)
        for(int x = 0; x < grey_image.cols; x++)
            Hist[(int)grey_image.at<uchar>(y,x)]++;

    Mat HistPlot(500, 256, CV_8UC3, Scalar(0, 0, 0));    
    for (int i = 0; i < 256; i=i+2)
    {
        line(HistPlot, Point(i, 500), Point(i, 500-Hist[i]), Scalar(0, 0, 0),1,8,0);  
    }    

    imshow("Original Image", grey_image);
    imshow("Grey Histogram", HistPlot);
 
}


void im_rgbhist(Mat image){
    //Mat image = imread("hany.jpg");
    int HistR[256] = {0};
    int HistG[256] = {0};
    int HistB[256] = {0};

    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
        {
            Vec3b intensity = image.at<Vec3b>(Point(j, i));
            int Red = intensity.val[0];
            int Green = intensity.val[1];
            int Blue = intensity.val[2];
            HistR[Red] = HistR[Red]+1;
            HistB[Blue] = HistB[Blue]+1;
            HistG[Green] = HistG[Green]+1;
        }
    Mat HistPlotR (500, 256, CV_8UC3, Scalar(0, 0, 0));
    Mat HistPlotG (500, 256, CV_8UC3, Scalar(0, 0, 0));
    Mat HistPlotB (500, 256, CV_8UC3, Scalar(0, 0, 0));
    Mat HistPlot (500, 256, CV_8UC3, Scalar(0, 0, 0));


    for (int i = 0; i < 256; i=i+2)
    {
        line(HistPlotR, Point(i, 500), Point(i, 500-HistR[i]), Scalar(0, 0, 255),1,8,0);
        line(HistPlotG, Point(i, 500), Point(i, 500-HistG[i]), Scalar(0, 255, 0),1,8,0);
        line(HistPlotB, Point(i, 500), Point(i, 500-HistB[i]), Scalar(255, 0, 0),1,8,0);
        
    }

    Mat combined_hist1;
    Mat combined_hist2;
    
    add(HistPlotB,HistPlotG, combined_hist1);
    add(combined_hist1,HistPlotR, combined_hist2);


    /**namedWindow("Red Histogram");
    namedWindow("Green Histogram");
    namedWindow("Blue Histogram");
    imshow("Original Image", image);
    imshow("Red Histogram", HistPlotR);
    imshow("Green Histogram", HistPlotG);
    imshow("Blue Histogram", HistPlotB);**/
    imshow("Original Image", image);
    imshow("RGB Histogram", combined_hist2);

}

int main(){

    Mat image = imread("flower.jpg");
    if (image.channels() == 3){
        im_rgbhist(image);
    }
    else if (image.channels() == 1){
        imhist(image);
    }


    waitKey(0);
    return 0;
}
