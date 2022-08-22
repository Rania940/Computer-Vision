#include "noise.h"


std::default_random_engine generator;


//1-Add additive noise to the image

void gaussianNoise(cv::Mat& image, const unsigned char mean, const unsigned char sigma)
{

    std::normal_distribution<double> distribution(mean, sigma);

    int imageChannels = image.channels();

    for (int row = 0; row < image.rows; row++)
    {
        for (int column = 0; column < image.cols; column++)
        {
            for (int channel = 0; channel < imageChannels; channel++)
            {
                unsigned char* pixelValuePtr = image.ptr(row) + (column * imageChannels) + channel;

                long newPixelValue = *pixelValuePtr + distribution(generator);

                *pixelValuePtr = newPixelValue > 255 ? 255 : newPixelValue < 0 ? 0 : newPixelValue;
            }
        }
    }
    // Show Image inside a window 
    cv::imshow("gaussian noise image", image);
}


void saltAndPepperNoise(cv::Mat& image, float saltProbability, float pepperProbability)
{
    int noise1 = image.rows * image.cols * saltProbability;
    int noise2 = image.rows * image.cols * pepperProbability;

    cv::RNG rng; // rand number generate


    for (long i = 0; i < noise1; i++)
    {
        image.at<uchar>(rng.uniform(0, image.rows), rng.uniform(0, image.cols)) = 0;
    }
    for (long i = 0; i < noise2; i++)
    {
        image.at<uchar>(rng.uniform(0, image.rows), rng.uniform(0, image.cols)) = 255;
    }
    // Show Image inside a window 
    cv::imshow("salt and pepper noise image", image);
}


void uniformNoise(cv::Mat& image, const unsigned char a, const unsigned char b)
{

    std::uniform_real_distribution<> distribution(a, b);
    int imageChannels = image.channels();

    for (int row = 0; row < image.rows; row++)
    {
        for (int column = 0; column < image.cols; column++)
        {
            for (int channel = 0; channel < imageChannels; channel++)
            {
                unsigned char* pixelValuePtr = image.ptr(row) + (column * imageChannels) + channel;

                long newPixelValue = *pixelValuePtr + distribution(generator);

                *pixelValuePtr = newPixelValue > 255 ? 255 : newPixelValue < 0 ? 0 : newPixelValue;
            }
        }


    }
    cv::imshow("Uniform noise image", image);
}
