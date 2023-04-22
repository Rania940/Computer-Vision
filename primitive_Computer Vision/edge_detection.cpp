#include "edge_detection.h"
#include "utils`.cpp"
#include"filters.h"

void Sobel(cv::Mat src, cv::Mat& dst, cv::Mat& angles_sobel , int gauss_size = 3, float sigma = 1.0) {
    //convert to grey scale
    cv::Mat grey_img = ToGreyScale(src);

    //gaussian flter to remove high freq comp
    grey_img = GaussianFilter(grey_img, gauss_size, sigma);

    //Sobel kernels
    std::vector<std::vector<double>> Gx = { {-1.0, 0.0, 1.0}, {-2.0, 0.0, 2.0}, {-1.0, 0.0, 1.0} };
    std::vector<std::vector<double>> Gy = { {1.0, 2.0, 1.0},{ 0.0, 0.0, 0.0},{ -1.0, -2.0, -1.0} };

    //horizontal and vertical conv
    cv::Mat out_h = Convolution(grey_img, Gx);
    cv::Mat out_v = Convolution(grey_img, Gy);

    //magnitude
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double sumh = (double)out_h.at<uchar>(i, j) * out_h.at<uchar>(i, j);
            double sumv = (double)out_v.at<uchar>(i, j) * out_v.at<uchar>(i, j);
            double mag = sqrt(sumh + sumv);
            //threshold
           // mag = (mag >100) ? 255 : mag;

            dst.at<uchar>(i, j) = mag; //mag
            angles_sobel.at<double>(i, j) = (atan(sumv / sumh)) * (180.0 / 3.14159); //angles in degree
        }
    }
}

void Prewitt(cv::Mat src, cv::Mat& dst, cv::Mat& angles_prewitt, int gauss_size = 3, float sigma = 1.0) {
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
            dst.at<uchar>(i, j) = mag;
            angles_prewitt.at<double>(i, j) = (atan(sumv / sumh)) * (180.0 / 3.14159); //angles in degree

        }
    }

}

void Roberts(cv::Mat src, cv::Mat& dst, cv::Mat& angles_roberts, int gauss_size = 3, float sigma = 1.0) {
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
            dst.at<uchar>(i, j) = mag;
            angles_roberts.at<double>(i, j) = (atan(sumv / sumh)) * (180.0 / 3.14159); //angles in degree
        }
    }

}

void Canny(cv::Mat src, cv::Mat& dst, cv::Mat& angles_sobel,int l_th = 40, int h_th = 100, int gauss_size = 3, float sigma = 1.0) {
    //convert to grey scale
    cv::Mat grey_img = ToGreyScale(src);

    //gaussian flter for noise reduction
    grey_img = GaussianFilter(grey_img, gauss_size, sigma);

    //Sobel filtering
    cv::Mat mag_sobel;
    cv::Mat angles_sobel;
    Sobel(src, mag_sobel, angles_sobel);
    cv::Mat non_max = cv::Mat::Mat(mag_sobel.rows - 2, mag_sobel.cols - 2, CV_8UC1);  //-2
    double gradient_angle;

    //non max suppression , loop for peaks 
    for (int i = 1; i < mag_sobel.rows - 1; i++)
    {
        for (int j = 1; j < mag_sobel.cols - 1; j++)
        {
            gradient_angle = angles_sobel.at<double>(i, j);

            non_max.at<uchar>(i - 1, j - 1) = mag_sobel.at<uchar>(i, j);
            //0
            if (((-22.5 < gradient_angle) && (gradient_angle <= 22.5)) || ((157.5 < gradient_angle) && (gradient_angle <= -157.5)))
            {
                if ((mag_sobel.at<uchar>(i, j) < mag_sobel.at<uchar>(i, j + 1)) || (mag_sobel.at<uchar>(i, j) < mag_sobel.at<uchar>(i, j - 1)))
                    non_max.at<uchar>(i - 1, j - 1) = 0;
            }
            //90
            if (((-112.5 < gradient_angle) && (gradient_angle <= -67.5)) || ((67.5 < gradient_angle) && (gradient_angle <= 112.5)))
            {
                if ((mag_sobel.at<uchar>(i, j) < mag_sobel.at<uchar>(i + 1, j)) || (mag_sobel.at<uchar>(i, j) < mag_sobel.at<uchar>(i - 1, j)))
                    non_max.at<uchar>(i - 1, j - 1) = 0;
            }

            //-45
            if (((-67.5 < gradient_angle) && (gradient_angle <= -22.5)) || ((112.5 < gradient_angle) && (gradient_angle <= 157.5)))
            {
                if ((mag_sobel.at<uchar>(i, j) < mag_sobel.at<uchar>(i - 1, j + 1)) || (mag_sobel.at<uchar>(i, j) < mag_sobel.at<uchar>(i + 1, j - 1)))
                    non_max.at<uchar>(i - 1, j - 1) = 0;
            }

            //45
            if (((-157.5 < gradient_angle) && (gradient_angle <= -112.5)) || ((22.5 < gradient_angle) && (gradient_angle <= 67.5)))
            {
                if ((mag_sobel.at<uchar>(i, j) < mag_sobel.at<uchar>(i + 1, j + 1)) || (mag_sobel.at<uchar>(i, j) < mag_sobel.at<uchar>(i - 1, j - 1)))
                    non_max.at<uchar>(i - 1, j - 1) = 0;
            }
        }
    }

    //double thresholding 

    //Hysteresis
}


void DetectEdges(std::string filter_name, cv::Mat src, cv::Mat& dst, cv::Mat& angles, int gauss_size = 3, float sigma = 1.0 , int l_th = 40, int h_th = 100, ) {
    
    if (filter_name == "Sobel") {
        return Sobel(src, dst, angles, gauss_size = 3, sigma);
    }

    else if (filter_name == "Prewitt") {
        return Prewitt(src, dst, angles, gauss_size = 3, sigma);
    }

    else if (filter_name == "Roberts") {
        return Roberts(src, dst, angles, gauss_size = 3, sigma);
    }

    else if (filter_name == "Canny") {
        return Canny(src, dst, angles, gauss_size = 3, sigma , l_th, h_th);
    }
}