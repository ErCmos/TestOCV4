#ifndef LEARNOPENCV_H
#define LEARNOPENCV_H

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "tuple"
#include "iterator"
#include "iostream"
#include "QMessageBox"

/*
#include "dlib/opencv.h"
#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/dnn.h"
#include "dlib/data_io.h"
*/

/*
/////Para OCR/////
#include "string"
#include "tesseract/baseapi.h"
#include "leptonica/allheaders.h"
//#include "opencv2/opencv.hpp"
#include "opencv4/opencv2/opencv.hpp"
/////Para OCR//////
*/
class LearnOpenCV
{
public:
    LearnOpenCV();
    void detectFaceOpenCVHaar(cv::CascadeClassifier faceCascade, cv::Mat& frameOpenCVHaar, int inHeight, int inWidth);
    void haarCascade();
    void detectFaceOpenCVDNN(cv::dnn::Net net, cv::Mat& frameOpenCVDNN);
    void DNN();
    void AgeAndGenderDetector();
    //void detectFaceDlibHog(dlib::frontal_face_detector hogFaceDetector, cv::Mat &frameDlibHog, int inHeight_2, int inWidth_2);
    void dlibHOG();
    int OCR(QString fileName);
    void Test(QString texto);
};

#endif // LEARNOPENCV_H
