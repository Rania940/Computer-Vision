
#include "Utils.h"
#include "equalization.h"
#include "FrequencyDomain_Filters.h"










int main(int argc, char** argv)
{
   

    // Reading the image file
    cv::Mat image_g = cv::imread("C:/Users/ayaab/Desktop/Lenna.png", IMREAD_GRAYSCALE);
    cv::Mat image_u = cv::imread("C:/Users/ayaab/Desktop/Lenna.png", IMREAD_GRAYSCALE);
    cv::Mat image_s = cv::imread("C:/Users/ayaab/Desktop/Lenna.png", IMREAD_GRAYSCALE);


    resize(image_g, image_g, { 500,500 });
    resize(image_u, image_g, { 500,500 });
    resize(image_s, image_g, { 500,500 });


    //if (image.empty())
    /* {
        cout << "Could not read the image:" << endl;
        return 1;
    }*/
//1......................................
    //(Add gaussian noise to image)
    gaussianNoise(image_g, 128, 20);
    //(Add uniform noise to image)
    uniformNoise(image_u, 20, 128);
    //(Add salt and pepper noise to image after add gaussian noise to image )
    saltAndPepperNoise(image_s, 0.05, 0.05);

 









    //question 9   we can change D0 from displays function
    display_high_ideal();
    display_low_ideal();
    display_high_gaussian();
    display_low_gaussian();


    //question 5
    dispay_equalization();


   











    // Wait for any keystroke in the window
    waitKey(0);
    return 0;
}