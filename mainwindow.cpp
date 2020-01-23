#include "mainwindow.h"
#include "ui_mainwindow.h"



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

using namespace std;
using namespace cv;

void MainWindow::on_FaceDetector_clicked()
{
   Geeksforgeekers geek;
   geek.inicio();
}

void MainWindow::on_FaceDetector2_clicked()
{
    LearnOpenCV locv;
    locv.haarCascade();
}

void MainWindow::on_FaceDetectorLearnOpenCVDNN_clicked()
{
    LearnOpenCV locv;
    locv.DNN();
}

void MainWindow::on_GenderAndAgeDetector_clicked()
{
    LearnOpenCV locv;
    locv.AgeAndGenderDetector();
}

void MainWindow::on_OCR_clicked()
{/*
    LearnOpenCV locv;
    QString fileName = QFileDialog::getOpenFileName(this,
                tr("Open Video"), "/home/ercmos", tr("Video Files (*.avi *.mpg *.mp4 *.*)"));
    locv.OCR(fileName);
  */
}

void MainWindow::on_Mask_RCNN_clicked()
{
    LearnOpenCV locv;
    QString fileName = QFileDialog::getOpenFileName(this,
                tr("Open Video"), "/home/ercmos", tr("Video Files (*.avi *.mpg *.mp4 *.*)"));
    QString parser=ui->comboBox->currentText();
    locv.Mask_RCNN(fileName,parser);
}
