//
//  Landmaker.hpp
//  DCNCLandmark
//
//  Created by betty on 2018/4/26.
//  Copyright © 2018年 betty. All rights reserved.
//

#ifndef Landmaker_hpp
#define Landmaker_hpp

#include <stdio.h>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "CaffeClassify.h"
#include "face_alignment.hpp"
#include "helper.h"

using namespace caffe;
using std::string;

#ifndef CAFFE_FLOAT_DOUBLE
#define CAFFE_FLOAT_DOUBLE double
#define CAFFE_MATC3 CV_64FC3
#endif





class Landmaker
{
public:
    explicit Landmaker(int gpu_id,
                       const string &data_path);
    Landmaker();
    
public:
    
    cv::Mat GenerateRoi(cv::Mat img);
    
    void Init();
    cv::Mat CropImageTest(cv::Mat img, point2f FacePts, float scale, cv::Rect & crop_rect);
    FR_FaceInfo Detect(const cv::Mat& image, cv::Rect bbox, std::vector<float> &features);
    cv::Mat CropImage(cv::Mat img, point2f FacePts, float scale, cv::Rect & crop_rect);
    std::string getConfig(string level_id, const string data_path);
   
    point2f Convert(point2f Pts, cv::Rect crop_rect) ;

    
private:
    point2f TwoLevelLE1_1(cv::Mat img);
    point2f TwoLevelLE1_2(cv::Mat img);
    point2f TwoLevelLE2_1(cv::Mat img);
    point2f TwoLevelLE2_2(cv::Mat img);
    point2f TwoLevelLM1_1(cv::Mat img);
    point2f TwoLevelLM1_2(cv::Mat img);
    point2f TwoLevelN1_1(cv::Mat img);
    point2f TwoLevelN1_2(cv::Mat img);
    point2f TwoLevelN2_1(cv::Mat img);
    point2f TwoLevelN2_2(cv::Mat img);
    point2f TwoLevelRE1_1(cv::Mat img);
    point2f TwoLevelRE1_2(cv::Mat img);
    point2f TwoLevelRE2_1(cv::Mat img);
    point2f TwoLevelRE2_2(cv::Mat img);
    point2f TwoLevelRM1_1(cv::Mat img);
    point2f TwoLevelRM1_2(cv::Mat img);
    
private:
    cv::Rect faceBox; // bounding box
    int gpu_id;
    std::string parent_path;
    //FR_FaceInfo current
    FR_FaceInfo current_face;
    FR_FaceInfo pre_face;
    bool reset_track;
    int interval;
    bool isRoted;
    //LKTracker tracker;
    FR::CFaceAlignment mFaceAlig;
    cv::Mat preFrame;
    float x_scale;
    float y_scale;
    //int frame_no;
    cv::CascadeClassifier face_cascade;
    Classifier OneLevel_obj;
    Classifier TwoLevelLE1_1_obj;
    Classifier TwoLevelLE1_2_obj;
    Classifier TwoLevelRE1_1_obj;
    Classifier TwoLevelRE1_2_obj;
    Classifier TwoLevelLE2_1_obj;
    Classifier TwoLevelLE2_2_obj;
    Classifier TwoLevelRE2_1_obj;
    Classifier TwoLevelRE2_2_obj;
    Classifier TwoLevelN1_1_obj;
    Classifier TwoLevelN1_2_obj;
    Classifier TwoLevelN2_1_obj;
    Classifier TwoLevelN2_2_obj;
    Classifier TwoLevelLM_1_obj;
    Classifier TwoLevelLM_2_obj;
    Classifier TwoLevelRM_1_obj;
    Classifier TwoLevelRM_2_obj;
};

#endif /* Landmaker_hpp */
