#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "geeksforgeekers.h"
#include "learnopencv.h"
#include <QMainWindow>
#include <QFileDialog>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:

    void on_FaceDetector_clicked();

    void on_FaceDetector2_clicked();

    void on_FaceDetectorLearnOpenCVDNN_clicked();

    void on_GenderAndAgeDetector_clicked();

    void on_OCR_clicked();

    void on_TestButton_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
