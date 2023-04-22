#include <opencv2/opencv.hpp>
#include <iostream>


cv::Mat ToGreyScale(cv::Mat src) {
    cv::Mat src_grey = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    // luminosity method
    for (int i = 0; i < src.cols; i++) {
        for (int j = 0; j < src.rows; j++)
        {
            cv::Vec3b src_color = src.at<cv::Vec3b>(cv::Point(i, j));
            cv::Scalar converted = (0.11 * src_color.val[0] + 0.59 * src_color.val[1] + 0.3 * src_color.val[2]);
            src_grey.at<uchar>(cv::Point(i, j)) = converted.val[0];
        }
    }
    return src_grey;
}

cv::Mat ZeroPadding(cv::Mat img, int kernal_size) {
    cv::Mat img_c = img.clone();
    img_c.convertTo(img_c, CV_64FC(img_c.channels()));

    cv::Mat img_padded = cv::Mat::Mat(img.rows + kernal_size - 1, img.cols + kernal_size - 1, CV_64FC(img_c.channels()));
    int padd_size = (kernal_size - 1) / 2;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.channels() > 1) {
                for (int c = 0; c < img.channels(); c++) {
                    img_padded.at<cv::Vec3d>(i + padd_size, j + padd_size)[c] = img.at<cv::Vec3d>(i, j)[c];
                }
            }
            else {
                img_padded.at<double>(i + padd_size, j + padd_size) = img.at<double>(i, j);
            }

        }
    }
    return img_padded;
}

std::vector<std::vector<double>>  Kernel2D(int kernel_size, float sigma = 0.0)
{
    //window
    int k = (kernel_size - 1) / 2;
    //Average kernel
    if (sigma == 0) {
        std::vector<std::vector<double>> kernel(kernel_size, std::vector<double>(kernel_size, 1.0 / (kernel_size * kernel_size)));
        return kernel;
    }
    //Gaussian filter
    else {
        std::vector<std::vector<double>> kernel(kernel_size, std::vector<double>(kernel_size));
        double sumXY = 0;
        double s = 2.0 * sigma * sigma;
        double s_pi = 3.14 * s;
        double sum = 0;

        for (int x = -k; x <= k; x++) {
            for (int y = -k; y <= k; y++) {
                sumXY = (double)x * x + (double)y * y;
                kernel[x + k][y + k] = (exp(-(sumXY) / s)) / (s_pi);
                sum += kernel[x + k][y + k];
            }
        }

        //Normalize
        //std::transform

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                kernel[i][j] /= sum;
            }
        }
        return kernel;

    }
}

double Median(cv::Mat kernel_elements) {
    kernel_elements = kernel_elements.clone();
    kernel_elements = kernel_elements.reshape(0, kernel_elements.rows * kernel_elements.cols); // spread Input Mat to single row
    std::vector<double> vec;
    kernel_elements.copyTo(vec); // Copy Input Mat to vector vecFromMat
    std::nth_element(vec.begin(), vec.begin() + vec.size() / 2, vec.end());
    return vec[vec.size() / 2];
}

cv::Mat Convolution(cv::Mat src, std::vector<std::vector<double>>  kernel)
{
    //kernel size 
    int kernel_size = kernel[0].size();

    cv::Mat img = src.clone();
    img.convertTo(img, CV_64FC(img.channels()));

    //zero padding img  
    cv::Mat img_padded = ZeroPadding(img, kernel_size);

    cv::Mat out = cv::Mat::Mat(img.rows, img.cols, CV_64FC(img.channels()));

    //window 
    int k = ((kernel_size - 1) / 2) + 0.5;

    //Performing the convolution
    for (int i = k; i < img_padded.rows - k; i++) {
        for (int j = k; j < img_padded.cols - k; j++) {
            if (img_padded.channels() > 1) {
                for (int c = 0; c < img_padded.channels(); c++) {

                    double comp = 0;
                    for (int u = -k; u <= k; u++) {
                        for (int v = -k; v <= k; v++) {
                            comp = comp + (img_padded.at<cv::Vec3d>(i + u, j + v)[c] * kernel[u + k][v + k]);
                        }
                    }
                    out.at<cv::Vec3d>(i - k, j - k)[c] = comp;

                }
            }
            else {
                double comp = 0;
                for (int u = -k; u <= k; u++) {
                    for (int v = -k; v <= k; v++) {
                        comp = comp + (img_padded.at<double>(i + u, j + v) * kernel[u + k][v + k]);
                    }
                }
                out.at<double>(i - k, j - k) = comp;
            }
        }
    }
    out.convertTo(out, CV_8UC(img.channels()));
    return out;
}

