#include "learnopencv.h"

LearnOpenCV::LearnOpenCV()
{

}

using namespace std;
using namespace cv;
using namespace dnn;
//using namespace dlib;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);


#define CAFFE

const std::string caffeConfigFile = "/home/ercmos/Proyectos/TestOCV4/models/deploy.prototxt";
const std::string caffeWeightFile = "/home/ercmos/Proyectos/TestOCV4/models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "/home/ercmos/Proyectos/TestOCV4/models/models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "/home/ercmos/Proyectos/TestOCV4/models/opencv_face_detector_uint8.pb";

void LearnOpenCV::detectFaceOpenCVHaar(CascadeClassifier faceCascade, Mat& frameOpenCVHaar, int inHeight=300, int inWidth=0)
{
    int frameHeight = frameOpenCVHaar.rows;
    int frameWidth = frameOpenCVHaar.cols;
    if (!inWidth)
        inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    Mat frameOpenCVHaarSmall, frameGray;
    resize(frameOpenCVHaar, frameOpenCVHaarSmall, Size(inWidth, inHeight));
    cvtColor(frameOpenCVHaarSmall, frameGray, COLOR_BGR2GRAY);

    std::vector<Rect> faces;
    faceCascade.detectMultiScale(frameGray, faces);

    for ( size_t i = 0; i < faces.size(); i++ )
    {
      int x1 = (int)(faces[i].x * scaleWidth);
      int y1 = (int)(faces[i].y * scaleHeight);
      int x2 = (int)((faces[i].x + faces[i].width) * scaleWidth);
      int y2 = (int)((faces[i].y + faces[i].height) * scaleHeight);
      cv::rectangle(frameOpenCVHaar, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
    }

}

void LearnOpenCV::haarCascade ()
{
    string faceCascadePath;
    CascadeClassifier faceCascade;

    QMessageBox msgBox;
     msgBox.setText("Mensaje");
     //msgBox.exec();

    faceCascadePath = "/home/ercmos/Proyectos/TestOCV4/haarcascade_frontalface_default.xml";
     //faceCascade.load( "/home/ercmos/Proyectos/TestOCV4/haarcascade_frontalcatface.xml" ) ;

      if( !faceCascade.load( faceCascadePath ) ){ printf("--(!)Error loading face cascade\n"); msgBox.exec(); };

      VideoCapture source;
      source.open(0);
      Mat frame;

      double tt_opencvHaar = 0;
      double fpsOpencvHaar = 0;
      while(1)
      {
          source >> frame;
          if(frame.empty())
              break;
          // Show Processed Image with detected faces
          imshow( "Image Detection", frame );
          double t = cv::getTickCount();
          detectFaceOpenCVHaar ( faceCascade, frame );
          tt_opencvHaar = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
          fpsOpencvHaar = 1/tt_opencvHaar;
          putText(frame, format("OpenCV HAAR ; FPS = %.2f",fpsOpencvHaar), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
          imshow( "OpenCV - HAAR Face Detection", frame );
          int c = waitKey(5);
          if(c == 27 || c == 'q' || c == 'Q')
          {
            destroyAllWindows();
            break;
          }
        }
}

void LearnOpenCV::detectFaceOpenCVDNN(cv::dnn::Net net, Mat& frameOpenCVDNN)
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
#ifdef CAFFE
        cv::Mat inputBlob = dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
        cv::Mat inputBlob = dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }

}

void LearnOpenCV::DNN()
{
#ifdef CAFFE
  cv::dnn::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
  cv::dnn::Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif

  VideoCapture source;
  source.open(0);
  Mat frame;

  double tt_opencvDNN = 0;
  double fpsOpencvDNN = 0;
  while(1)
  {
      source >> frame;
      if(frame.empty())
          break;
      // Show Processed Image with detected faces
      imshow( "Image Detection", frame );
      double t = cv::getTickCount();
      detectFaceOpenCVDNN ( net, frame );
      tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
      fpsOpencvDNN = 1/tt_opencvDNN;
      putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
      imshow( "OpenCV - DNN Face Detection", frame );
      cout << "Time Taken = " << tt_opencvDNN << endl;
      int c = waitKey(5);
      if(c == 27 || c == 'q' || c == 'Q')
      {
        destroyAllWindows();
        break;
      }
    }
}
/*
void LearnOpenCV::detectFaceDlibHog(frontal_face_detector hogFaceDetector, Mat &frameDlibHog, int inHeight_2=300, int inWidth_2=0)
{

    int frameHeight = frameDlibHog.rows;
    int frameWidth = frameDlibHog.cols;
    if (!inWidth_2)
        inWidth_2 = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    Mat frameDlibHogSmall;
    resize(frameDlibHog, frameDlibHogSmall, Size(inWidth, inHeight));

    // Convert OpenCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(frameDlibHogSmall);

    // Detect faces in the image
    std::vector<dlib::rectangle> faceRects = hogFaceDetector(dlibIm);

    for ( size_t i = 0; i < faceRects.size(); i++ )
    {
      int x1 = (int)(faceRects[i].left() * scaleWidth);
      int y1 = (int)(faceRects[i].top() * scaleHeight);
      int x2 = (int)(faceRects[i].right() * scaleWidth);
      int y2 = (int)(faceRects[i].bottom() * scaleHeight);
      cv::rectangle(frameDlibHog, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
    }
}

void LearnOpenCV::dlibHOG()
{
    frontal_face_detector hogFaceDetector = get_frontal_face_detector();

      VideoCapture source;
      source.open(0);
      Mat frame;

      double tt_dlibHog = 0;
      double fpsDlibHog = 0;
      while(1)
      {
          source >> frame;
          if(frame.empty())
              break;

          double t = cv::getTickCount();
          detectFaceDlibHog ( hogFaceDetector, frame );
          tt_dlibHog = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
          fpsDlibHog = 1/tt_dlibHog;
          putText(frame, format("DLIB HoG ; FPS = %.2f",fpsDlibHog), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
          imshow( "DLIB - HoG Face Detection", frame );
          int k = waitKey(5);
          if(k == 27)
          {
            destroyAllWindows();
            break;
          }
        }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

tuple<Mat, vector<vector<int>>> getFaceBox(Net net, Mat &frame, double conf_threshold)
{
    Mat frameOpenCVDNN = frame.clone();
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    double inScaleFactor = 1.0;
    Size size = Size(300, 300);
    // std::vector<int> meanVal = {104, 117, 123};
    Scalar meanVal = Scalar(104, 117, 123);

    cv::Mat inputBlob;
    inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    vector<vector<int>> bboxes;

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > conf_threshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            vector<int> box = {x1, y1, x2, y2};
            bboxes.push_back(box);
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }

    return make_tuple(frameOpenCVDNN, bboxes);
}

void LearnOpenCV::AgeAndGenderDetector()
{
string faceProto = "/home/ercmos/Proyectos/TestOCV4/models/opencv_face_detector.pbtxt";
string faceModel = "/home/ercmos/Proyectos/TestOCV4/models/opencv_face_detector_uint8.pb";

string ageProto = "/home/ercmos/Proyectos/TestOCV4/models/age_deploy.prototxt";
string ageModel = "/home/ercmos/Proyectos/TestOCV4/models/age_net.caffemodel";

string genderProto = "/home/ercmos/Proyectos/TestOCV4/models/gender_deploy.prototxt";
string genderModel = "/home/ercmos/Proyectos/TestOCV4/models/gender_net.caffemodel";

Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);

vector<string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
  "(38-43)", "(48-53)", "(60-100)"};

vector<string> genderList = {"Male", "Female"};

// Load Network
Net ageNet = readNet(ageModel, ageProto);
Net genderNet = readNet(genderModel, genderProto);
Net faceNet = readNet(faceModel, faceProto);

VideoCapture cap;
cap.open(0);
int padding = 20;
while(waitKey(1) < 0) {
  // read frame
  Mat frame;
  cap.read(frame);
  if (frame.empty())
  {
      waitKey();
      break;
  }

  vector<vector<int>> bboxes;
  Mat frameFace;
  tie(frameFace, bboxes) = getFaceBox(faceNet, frame, 0.7);

  if(bboxes.size() == 0) {
    cout << "No face detected, checking next frame." << endl;
    continue;
  }
  for (auto it = begin(bboxes); it != end(bboxes); ++it) {
    Rect rec(it->at(0) - padding, it->at(1) - padding, it->at(2) - it->at(0) + 2*padding, it->at(3) - it->at(1) + 2*padding);
    Mat face = frame(rec); // take the ROI of box on the frame

    Mat blob;
    blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
    genderNet.setInput(blob);
    // string gender_preds;
    vector<float> genderPreds = genderNet.forward();
    // printing gender here
    // find max element index
    // distance function does the argmax() work in C++
    int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
    string gender = genderList[max_index_gender];
    cout << "Gender: " << gender << endl;

    /* // Uncomment if you want to iterate through the gender_preds vector
    for(auto it=begin(gender_preds); it != end(gender_preds); ++it) {
      cout << *it << endl;
    }
    */

    ageNet.setInput(blob);
    vector<float> agePreds = ageNet.forward();
    /* // uncomment below code if you want to iterate through the age_preds
     * vector
    cout << "PRINTING AGE_PREDS" << endl;
    for(auto it = age_preds.begin(); it != age_preds.end(); ++it) {
      cout << *it << endl;
    }
    */

    // finding maximum indicd in the age_preds vector
    int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
    string age = ageList[max_indice_age];
    cout << "Age: " << age << endl;
    string label = gender + ", " + age; // label
    cv::putText(frameFace, label, Point(it->at(0), it->at(1) -15), cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
    imshow("Frame", frameFace);
    imwrite("out.jpg",frameFace);
  }
}
}


///////////////////////////OCR///////////////////////////////////
//int LearnOpenCV::OCR(QString fileName)
//{
//        string outText;
//        string imPath = fileName.toStdString();

        // Create Tesseract object
//        tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();

        /*
         Initialize OCR engine to use English (eng) and The LSTM
         OCR engine.

         There are four OCR Engine Mode (oem) available

         OEM_TESSERACT_ONLY             Legacy engine only.
         OEM_LSTM_ONLY                  Neural nets LSTM engine only.
         OEM_TESSERACT_LSTM_COMBINED    Legacy + LSTM engines.
         OEM_DEFAULT                    Default, based on what is available.
        */

//        ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);

        // Set Page segmentation mode to PSM_AUTO (3)
        // Other important psm modes will be discussed in a future post.
//        ocr->SetPageSegMode(tesseract::PSM_AUTO);


        // Open input image using OpenCV
//        Mat im = cv::imread(imPath, IMREAD_COLOR);

        // Set image data
//        ocr->SetImage(im.data, im.cols, im.rows, 3, im.step);

        // Run Tesseract OCR on image
//        outText = string(ocr->GetUTF8Text());

        // print recognized text
//        cout << outText << endl;

        // Destroy used object and release memory
//        ocr->End();

//        return EXIT_SUCCESS;
//}
///////////////////////////OCR///////////////////////////////////



// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold
vector<string> classes;
vector<Scalar> colors;



// For each frame, extract the bounding box and mask for each detected object
void LearnOpenCV::postprocess(Mat& frame, const vector<Mat>& outs)
{
    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; ++i)
    {
        float score = outDetections.at<float>(i, 2);
        if (score > confThreshold)
        {
            // Extract the bounding box
            int classId = static_cast<int>(outDetections.at<float>(i, 1));
            int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
            int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
            int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
            int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

            left = max(0, min(left, frame.cols - 1));
            top = max(0, min(top, frame.rows - 1));
            right = max(0, min(right, frame.cols - 1));
            bottom = max(0, min(bottom, frame.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

            // Extract the mask for the object
            Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i,classId));

            // Draw bounding box, colorize and show the mask on the image
            drawBox(frame, classId, score, box, objectMask);

        }
    }
}

// Draw the predicted bounding box, colorize and show the mask on the image
void LearnOpenCV::drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

    Scalar color = colors[classId%colors.size()];

    // Resize the mask, threshold, color and apply it on the image
    resize(objectMask, objectMask, Size(box.width, box.height));
    Mat mask = (objectMask > maskThreshold);
    Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    // Draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
    coloredRoi.copyTo(frame(box), mask);

}




int LearnOpenCV::Mask_RCNN(QString fileName, QString parser)
{
    // Copyright (C) 2018-2019, BigVision LLC (LearnOpenCV.com), All Rights Reserved.
    // Author : Sunita Nayak
    // Article : https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
    // License: BSD-3-Clause-Attribution (Please read the license file.)

    // Usage example:  ./mask_rcnn.out --video=run.mp4
    //                 ./mask_rcnn.out --image=bird.jpg
//    #include <fstream>
//    #include <sstream>
//    #include <iostream>
//    #include <string.h>

//    #include <opencv2/dnn.hpp>
//    #include <opencv2/imgproc.hpp>
//    #include <opencv2/highgui.hpp>

//    const char* keys =
    "{help h usage ? | | Usage examples: \n\t\t./mask-rcnn.out --image=traffic.jpg \n\t\t./mask-rcnn.out --video=sample.mp4}"
    "{image i        |<none>| input image   }"
    "{video v       |<none>| input video   }"
    ;
//    using namespace cv;
//    using namespace dnn;
//    using namespace std;

    // Draw the predicted bounding box
//    void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask);

    // Postprocess the neural network's output for each frame
//    void postprocess(Mat& frame, const vector<Mat>& outs);

//    int main(int argc, char** argv)
//    {
//        CommandLineParser parser(argc, argv, keys);
//        parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
//        if (parser.has("help"))
//        {
//            parser.printMessage();
//            return 0;
//        }
        // Load names of classes
        string classesFile = "mscoco_labels.names";
        ifstream ifs(classesFile.c_str());
        string line;
        while (getline(ifs, line)) classes.push_back(line);

        // Load the colors
        string colorsFile = "/home/ercmos/Proyectos/TestOCV4-master/models/colors.txt";
        ifstream colorFptr(colorsFile.c_str());
        while (getline(colorFptr, line)) {
            char* pEnd;
            double r, g, b;
            r = strtod (line.c_str(), &pEnd);
            g = strtod (pEnd, NULL);
            b = strtod (pEnd, NULL);
            Scalar color = Scalar(r, g, b, 255.0);
            colors.push_back(Scalar(r, g, b, 255.0));
        }

        // Give the configuration and weight files for the model
        String textGraph = "/home/ercmos/Proyectos/TestOCV4-master/models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
        String modelWeights = "/home/ercmos/Proyectos/TestOCV4-master/models/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

        // Load the network
        Net net = readNetFromTensorflow(modelWeights, textGraph);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        // Open a video file or an image file or a camera stream.
        string str, outputFile;
        VideoCapture cap;
        VideoWriter video;
        Mat frame, blob;

        try {

            outputFile = "mask_rcnn_out_cpp.avi";
//            if (parser.has("image"))
            if (parser=="image")
            {
                // Open the image file
//              str = parser.get<String>("image");
              str = fileName.toStdString();
                //cout << "Image file input : " << str << endl;
                ifstream ifile(str);
                if (!ifile) throw("error");
                cap.open(str);
                str.replace(str.end()-4, str.end(), "_mask_rcnn_out.jpg");
                outputFile = str;
            }
//            else if (parser.has("video"))
            else if (parser=="video")
            {
                // Open the video file
//                str = parser.get<String>("video");
                str = fileName.toStdString();
                ifstream ifile(str);
                if (!ifile) throw("error");
                cap.open(str);
                str.replace(str.end()-4, str.end(), "_mask_rcnn_out.avi");
                outputFile = str;
            }
            // Open the webcam
//            else cap.open(parser.get<int>("device"));
            else cap.open(0);

        }
        catch(...) {
            cout << "Could not open the input image/video stream" << endl;
            return 0;
        }

        // Get the video writer initialized to save the output video
//        if (!parser.has("image")) {
          if (parser!="image") {
            video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
        }

        // Create a window
        static const string kWinName = "Deep learning object detection in OpenCV";
        namedWindow(kWinName, WINDOW_NORMAL);

        // Process frames.
        while (waitKey(1) < 0)
        {
            // get frame from the video
            cap >> frame;

            // Stop the program if reached end of video
            if (frame.empty()) {
                cout << "Done processing !!!" << endl;
                cout << "Output file is stored as " << outputFile << endl;
                waitKey(3000);
                break;
            }
            // Create a 4D blob from a frame.
             blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);
            //blobFromImage(frame, blob);

            //Sets the input to the network
            net.setInput(blob);

            // Runs the forward pass to get output from the output layers
            std::vector<String> outNames(2);
            outNames[0] = "detection_out_final";
            outNames[1] = "detection_masks";
            vector<Mat> outs;
            net.forward(outs, outNames);

            // Extract the bounding box and mask for each of the detected objects
            postprocess(frame, outs);

            // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            string label = format("Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms", t);
            putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

            // Write the frame with the detection boxes
            Mat detectedFrame;
            frame.convertTo(detectedFrame, CV_8U);
//            if (parser.has("image")) imwrite(outputFile, detectedFrame);
            if (parser=="image") imwrite(outputFile, detectedFrame);
            else video.write(detectedFrame);

            imshow(kWinName, frame);

        }

        cap.release();
//        if (!parser.has("image")) video.release();
          if (parser!="image") video.release();

        return 0;
}
