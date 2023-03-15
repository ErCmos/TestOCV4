#ifndef GEEKSFORGEEKERS_H
#define GEEKSFORGEEKERS_H

//#include "opencv2/opencv.hpp"
// Include required header files from OpenCV directory

#include "opencv2/core/types_c.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

class Geeksforgeekers
{
public:
    Geeksforgeekers();
    void detectAndDraw( cv::Mat& img,
                        cv::CascadeClassifier& cascade,
                        cv::CascadeClassifier& nestedCascade,
                        double scale);
    void inicio();
};

#endif // GEEKSFORGEEKERS_H
