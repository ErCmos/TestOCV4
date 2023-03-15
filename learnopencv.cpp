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
      Mat frame,dst;

      double tt_opencvHaar = 0;
      double fpsOpencvHaar = 0;
      while(1)
      {
          source >> frame;
          if(frame.empty())
              break;
          // Show Processed Image with detected faces
          cv::resize(frame,dst, Size(), .5, .5, INTER_LINEAR);
          imshow( "Image Detection", dst );
          double t = cv::getTickCount();
          detectFaceOpenCVHaar ( faceCascade, dst );
          tt_opencvHaar = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
          fpsOpencvHaar = 1/tt_opencvHaar;
          putText(dst, format("OpenCV HAAR ; FPS = %.2f",fpsOpencvHaar), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
          //cv::resize(frame,dst, Size(), .5, .5, INTER_LINEAR);
          imshow( "OpenCV - HAAR Face Detection", dst );
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
  Mat frame,dst;

  double tt_opencvDNN = 0;
  double fpsOpencvDNN = 0;
  while(1)
  {
      source >> frame;
      if(frame.empty())
          break;
      // Show Processed Image with detected faces
      cv::resize(frame,dst, Size(), .5, .5, INTER_LINEAR);
      imshow( "Image Detection", dst );
      double t = cv::getTickCount();
      detectFaceOpenCVDNN ( net, frame );
      tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
      fpsOpencvDNN = 1/tt_opencvDNN;
      putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
      cv::resize(frame,dst, Size(), .5, .5, INTER_LINEAR);
      imshow( "OpenCV - DNN Face Detection", dst );
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
  cap.read(framocabezudo    Empleados2022c
           ocabezudo    \\e);
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
void LearnOpenCV::Test(QString Texto)
{


    VideoCapture source;
    //source.open(0);
    Mat frame,dst;

    //QMessageBox::information(this,"Título",Texto);
   QMessageBox Box;
   Box.setText(Texto);
   //Box.exec();
   //enum Strings { "Fichero", "Vídeo", "Cámara" }
   //Strings eTexto;

   QStringList eTexto;
   eTexto << "Fichero" << "Vídeo" << "Cámara";

   switch (eTexto.indexOf(Texto))
   {
   case 0:
    Box.setText("Lo elegido es un Fichero");
    Box.exec();
   break;
   case 1:
    Box.setText("Lo elegido es un Vídeo");
    Box.exec();
   break;
   case 2:
    //Box.setText("Lo elegido es una Cámara");
    //Box.exec();
    source.open(0);
   break;
   default:
    Box.exec();
   break;
   }
   while (1)
   {
        source >> frame;

        cv::resize(frame,dst, Size(), .5, .5, INTER_LINEAR);
        //cv::resize(frame,dst,Size(720,576));
        imshow( "Image Detection", dst);
        int c = waitKey(5);
        if(c == 27 || c == 'q' || c == 'Q')
        {
          destroyAllWindows();
          break;
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
