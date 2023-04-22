#ifndef equalization_h
#define equalization_h

#include "utils.h"
#include "equalization.h"

using namespace cv;

void cumhist(int histogram[], int cumhistogram[]);


Mat equalization_Algorithm_GRAYSCALE();
Mat equalization_Algorithm_COLOUR();

#endif 