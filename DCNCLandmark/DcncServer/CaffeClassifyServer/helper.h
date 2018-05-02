//
//  helper.h
//  CaffeClassifyServer
//
//  Created by betty on 2018/2/14.
//  Copyright © 2018年 RemarkMedia. All rights reserved.
//
#include <stdio.h>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/legacy.hpp"

#ifndef helper_h
#define helper_h

#define PI (3.14159265358979323846)
#define PI_DIV_180 (0.017453292519943296)//π/180
#define DegToRad(x) ((x) * PI_DIV_180)//角度转换为弧度
#define ANGLE(x)   ((x) * 180.0 / PI )  //角度 = 弧度 * 180 / π

const int NUM_REGRESSIONS = 4;
const int NUM_PTS = 8;

struct BBox
{
    float x1;// left
    float y1;// right
    float x2;// top
    float y2;// bottom
    float width = x2 - x1;
    float height = y2 - y1;
    cv::Rect GetRect() const;
    BBox GetSquare() const;
};

struct point2f
{
    float x;
    float y;
};

struct FR_FaceInfo
{
    cv::Rect FaceR;
    point2f FacePts[8];
    cv::Mat FaceImg;
    float pitch;
    float yaw;
    float roll;
};
#endif /* helper_h */
