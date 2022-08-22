#include "normalize.h"

void normalize_image(cv::Mat& image)
{
    int* histogram = imhist(image);

    int max_intensity = get_max(histogram, 256);
    int min_intensity = get_min(histogram, 256);

    int image_channels = image.channels();

    // Mat normalized_image;

    for (int channel = 0; channel < image_channels; channel++)
    {
        for (int row = 0; row < image.rows; row++)
        {
            for (int column = 0; column < image.cols; column++)
            {
                unsigned char* pixelValuePtr = image.ptr(row) + (column * image_channels) + channel;
                int newPixelValue = int((*pixelValuePtr - min_intensity) * (255.0 / (max_intensity - min_intensity)));
                *pixelValuePtr = newPixelValue;

            }
        }

    }


}