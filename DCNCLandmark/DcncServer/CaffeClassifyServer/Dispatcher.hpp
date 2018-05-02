//
//  Dispatcher.hpp
//  DCNCLandmark
//
//  Created by betty on 2018/4/26.
//  Copyright © 2018年 betty. All rights reserved.
//

#ifndef Dispatcher_hpp
#define Dispatcher_hpp

#include <stdio.h>
#include "face_alignment.hpp"
#include "DcncAlign.hpp"
#include "Landmaker.hpp"
using namespace std;

// TODO:
class Dispatcher {
private:
    bool reset_track;
    FR::CFaceAlignment mFaceAlig;
    cv::CascadeClassifier face_cascade;
    cv::Rect faceBox;
    DcncAlign mDcnc;
    Landmaker mLandmaker;
    vector<float> base_landmarks;
    cv::Mat trans, trans_inv;
    
public:
    Dispatcher();
    void Init();
    void GetBBox(cv::Mat img, cv::Rect& faceBox);
    cv::Mat CropImg(cv::Mat img);
    void Detect(cv::Mat img, vector<float> &features);
    cv::Vec3d EstimatePose(cv::Mat &shape);
    float EstimateFaceWidth(cv::Vec3d rot, cv::Mat landmarks);
};

#endif /* Dispatcher_hpp */
