#-------------------------------------------------
#
# Project created by QtCreator 2019-05-24T12:49:40
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestOCV4
TEMPLATE = app

#INCLUDEPATH += /usr/include/opencv
#INCLUDEPATH += /usr/local/include
#INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv4
#LIBS += -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_dnn
#LIBS += -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_features2d -lopencv_xfeatures2d -lopencv_calib3d -lopencv_ml -lopencv_optim
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_objdetect -lopencv_calib3d -lopencv_dnn
#LIBS += -L/usr/lib/x86_64-linux-gnu/libtesseract.so -ltesseract
#LIBS += -L/usr/lib/x86_64-linux-gnu/liblept.so -llept
LIBS += -L/usr/lib/x86_64-linux-gnu/libtesseract.so
LIBS += -L/usr/lib/x86_64-linux-gnu/liblept.so
#LIBS += -L/usr/local/lib -ldlib
#INCLUDEPATH += /usr/local/include/dlib
#LIBS += -L/home/ercmos/Soft/OpenCV/dlib-19.17/build/dlib -ldlib
#PKGCONFIG += dlib-1

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        geeksforgeekers.cpp \
        learnopencv.cpp \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        geeksforgeekers.h \
        learnopencv.h \
        mainwindow.h

FORMS += \
        mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
