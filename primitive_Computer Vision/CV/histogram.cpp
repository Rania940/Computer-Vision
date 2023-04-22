#include "histogram.h"
using namespace cv;

int* imhist(cv::Mat grey_image)
{
    int Hist[256] = { 0 };


    for (int y = 0; y < grey_image.rows; y++)
        for (int x = 0; x < grey_image.cols; x++)
            Hist[(int)grey_image.at<uchar>(y, x)]++;


    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);
    Mat HistPlot(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

    int max = Hist[0];
    for (int i = 1; i < 256; i++) {
        if (max < Hist[i]) {
            max = Hist[i];
        }
    }
    for (int i = 0; i < 255; i++) {
        Hist[i] = ((double)Hist[i] / max) * HistPlot.rows;
    }

    for (int i = 0; i < 255; i++)
    {
        line(HistPlot, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - Hist[i]), Scalar(0, 0, 0), 1, 8, 0);
    }

    imshow("Original Image", grey_image);
    imshow("Grey Histogram", HistPlot);
    return Hist; 
}
 



void im_rgbhist(cv::Mat image) {
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
    
      
    
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w/256);
    Mat HistPlotR (hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat HistPlotG (hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat HistPlotB (hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    int max = HistR[0];
    for(int i = 1; i < 256; i++){
        if(max < HistR[i]){
            max = HistR[i];
        }
    }   
    for(int i = 0; i < 255; i++){
        HistR[i] = ((double)HistR[i]/max)*HistPlotR.rows;
        HistG[i] = ((double)HistG[i]/max)*HistPlotG.rows;
        HistB[i] = ((double)HistB[i]/max)*HistPlotB.rows;
    }

    for (int i = 0; i < 256; i=i+2)
    {
        line(HistPlotR, Point(bin_w*(i), hist_h), Point(bin_w*(i), hist_h - HistR[i]),Scalar(0,0,255), 1, 8, 0);
        line(HistPlotG, Point(bin_w*(i), hist_h), Point(bin_w*(i), hist_h - HistG[i]), Scalar(0, 255, 0),1,8,0);
        line(HistPlotB, Point(bin_w*(i), hist_h), Point(bin_w*(i), hist_h - HistB[i]), Scalar(255, 0, 0),1,8,0);        
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
