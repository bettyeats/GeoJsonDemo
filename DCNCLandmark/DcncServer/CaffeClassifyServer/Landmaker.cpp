//
//  Landmaker.cpp
//  DCNCLandmark
//
//  Created by betty on 2018/4/26.
//  Copyright © 2018年 betty. All rights reserved.
//

#include "Landmaker.hpp"
#include <iostream>
#include <fstream>

#include "CaffeClassify.h"

//#define DES_DECODE

#ifdef DES_DECODE
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/coded_stream.h"
#include "cryptopp.h"
#endif

using namespace std;

Landmaker::Landmaker() {
}

Landmaker::Landmaker (int gpu_id, const string& data_path){
    this->gpu_id = gpu_id;
    this->parent_path = data_path;
    x_scale = y_scale = 1;
    interval = 5;
    isRoted = false;
    reset_track = true;
    Init();
}

void Landmaker::Init() {
    
    //LKTracker tracker = LKTracker();
    cv::CascadeClassifier face_cascade;
    string file_path = "/Users/betty/res/haarcascades/haarcascade_frontalface_alt2.xml";
    face_cascade.load(file_path);
    if (face_cascade.empty())
    {
        std::cout << "face detect load failure." << std::endl;
    }
    this->face_cascade = face_cascade;
    string config_path = getConfig("OneLevel", this->parent_path);
    Classifier fClassify(this->gpu_id, config_path);
    this->OneLevel_obj = fClassify;
    
    config_path = getConfig("TwoLevelLE1_1", this->parent_path);
    Classifier TwoLevelLE1_1(this->gpu_id, config_path);
    this->TwoLevelLE1_1_obj = TwoLevelLE1_1;
    
    config_path = getConfig("TwoLevelLE1_2", this->parent_path);
    Classifier TwoLevelLE1_2(this->gpu_id, config_path);
    this->TwoLevelLE1_2_obj = TwoLevelLE1_2;
    
    config_path = getConfig("TwoLevelLE2_1", this->parent_path);
    Classifier TwoLevelLE2_1(this->gpu_id, config_path);
    this->TwoLevelLE2_1_obj = TwoLevelLE2_1;
    
    config_path = getConfig("TwoLevelLE2_2", this->parent_path);
    Classifier TwoLevelLE2_2(this->gpu_id, config_path);
    this->TwoLevelLE2_2_obj = TwoLevelLE2_2;
    
    config_path = getConfig("TwoLevelLM1_1", this->parent_path);
    Classifier TwoLevelLM1_1(this->gpu_id, config_path);
    this->TwoLevelLM_1_obj = TwoLevelLM1_1;
    
    config_path = getConfig("TwoLevelLM1_2", this->parent_path);
    Classifier TwoLevelLM1_2(this->gpu_id, config_path);
    this->TwoLevelLM_2_obj = TwoLevelLM1_2;
    
    config_path = getConfig("TwoLevelN1_1", this->parent_path);
    Classifier TwoLevelN1_1(this->gpu_id, config_path);
    this->TwoLevelN1_1_obj = TwoLevelN1_1;
    
    config_path = getConfig("TwoLevelN1_2", this->parent_path);
    Classifier TwoLevelN1_2(this->gpu_id, config_path);
    this->TwoLevelN1_2_obj = TwoLevelN1_2;
    
    config_path = getConfig("TwoLevelN2_1", this->parent_path);
    Classifier TwoLevelN2_1(this->gpu_id, config_path);
    this->TwoLevelN2_1_obj = TwoLevelN2_1;
    
    config_path = getConfig("TwoLevelN2_2", this->parent_path);
    Classifier TwoLevelN2_2(this->gpu_id, config_path);
    this->TwoLevelN2_2_obj = TwoLevelN2_2;
    
    config_path = getConfig("TwoLevelRE1_1", this->parent_path);
    Classifier TwoLevelRE1_1(this->gpu_id, config_path);
    this->TwoLevelRE1_1_obj = TwoLevelRE1_1;
    
    config_path = getConfig("TwoLevelRE1_2", this->parent_path);
    Classifier TwoLevelRE1_2(this->gpu_id, config_path);
    this->TwoLevelRE1_2_obj = TwoLevelRE1_2;
    
    config_path = getConfig("TwoLevelRE2_1", this->parent_path);
    Classifier TwoLevelRE2_1(this->gpu_id, config_path);
    this->TwoLevelRE2_1_obj = TwoLevelRE2_1;
    
    config_path = getConfig("TwoLevelRE2_2", this->parent_path);
    Classifier TwoLevelRE2_2(this->gpu_id, config_path);
    this->TwoLevelRE2_2_obj = TwoLevelRE2_2;
    
    config_path = getConfig("TwoLevelRM1_1", this->parent_path);
    Classifier TwoLevelRM1_1(this->gpu_id, config_path);
    this->TwoLevelRM_1_obj = TwoLevelRM1_1;
    
    config_path = getConfig("TwoLevelRM1_2", this->parent_path);
    Classifier TwoLevelRM1_2(this->gpu_id, config_path);
    this->TwoLevelRM_2_obj = TwoLevelRM1_2;
}

FR_FaceInfo Landmaker::Detect(const cv::Mat& image, cv::Rect bbox, std::vector<float> &features){
    
    features.clear();
    FR_FaceInfo faces;
    cv::Mat img, croppedImg;
    this->faceBox = bbox;
    
    img = image;
    croppedImg = GenerateRoi(img);
    //std::cout << "after GenerateBoundingBox." << std::endl;
    cv::imwrite("/Users/betty/Documents/betty/examples/duanhai.jpg", croppedImg);
    
    std::vector<float> Pts;
    this->OneLevel_obj.Classify(croppedImg, Pts);
    
    // point LE1
    point2f LE1Pts;
    LE1Pts.x = Pts[4];
    LE1Pts.y = Pts[5];
    cv::Rect crop_le1_1;
    cv::Rect crop_le1_2;
    cv::Mat img_le1_1 = CropImage(img, LE1Pts, 0.16, crop_le1_1);
    cv::Mat img_le1_2 = CropImage(img, LE1Pts, 0.18, crop_le1_2);
    point2f LE1_1Pts = TwoLevelLE1_1(img_le1_1);
    point2f LE1_2Pts = TwoLevelLE1_2(img_le1_2);
    //cv::imwrite("/Users/betty/Documents/betty/examples/3333.jpg", img_le1_1);
    //cv::imwrite("/Users/betty/Documents/betty/examples/0000000.jpg", img_le1_2);
    // TODO: convert position value
    LE1_1Pts = Convert(LE1_1Pts, crop_le1_1);
    LE1_2Pts = Convert(LE1_2Pts, crop_le1_2);
    point2f LE1_Avg;
    LE1_Avg.x = (LE1_1Pts.x + LE1_2Pts.x) / 2;
    LE1_Avg.y = (LE1_1Pts.y + LE1_2Pts.y) / 2;
    
    // point LE2
    point2f LE2Pts;
    LE2Pts.x = Pts[6];
    LE2Pts.y = Pts[7];
    cv::Rect crop_le2_1;
    cv::Rect crop_le2_2;
    cv::Mat img_le2_1 = CropImage(img, LE2Pts, 0.16, crop_le2_1);
    cv::Mat img_le2_2 = CropImage(img, LE2Pts, 0.18, crop_le2_2);
    point2f LE2_1Pts = TwoLevelLE2_1(img_le2_1);
    point2f LE2_2Pts = TwoLevelLE2_2(img_le2_2);
    // TODO: convert position value
    LE2_1Pts = Convert(LE2_1Pts, crop_le2_1);
    LE2_2Pts = Convert(LE2_2Pts, crop_le2_2);
    point2f LE2_Avg;
    LE2_Avg.x = (LE2_1Pts.x + LE2_2Pts.x) / 2;
    LE2_Avg.y = (LE2_1Pts.y + LE2_2Pts.y) / 2;
    
    // point RE1
    point2f RE1Pts;
    RE1Pts.x = Pts[8];
    RE1Pts.y = Pts[9];
    cv::Rect crop_re1_1;
    cv::Rect crop_re1_2;
    cv::Mat img_re1_1 = CropImage(img, RE1Pts, 0.16, crop_re1_1);
    cv::Mat img_re1_2 = CropImage(img, RE1Pts, 0.18, crop_re1_2);
    point2f RE1_1Pts = TwoLevelRE1_1(img_re1_1);
    point2f RE1_2Pts = TwoLevelRE1_2(img_re1_2);
    //cv::imwrite("/Users/betty/Documents/betty/examples/44444.jpg", img_re1_1);
    // TODO: convert position value
    RE1_1Pts = Convert(RE1_1Pts, crop_re1_1);
    RE1_2Pts = Convert(RE1_2Pts, crop_re1_2);
    point2f RE1_Avg;
    RE1_Avg.x = (RE1_1Pts.x + RE1_2Pts.x) / 2;
    RE1_Avg.y = (RE1_1Pts.y + RE1_2Pts.y) / 2;
    
    // point RE2
    point2f RE2Pts;
    RE2Pts.x = Pts[10];
    RE2Pts.y = Pts[11];
    cv::Rect crop_re2_1;
    cv::Rect crop_re2_2;
    cv::Mat img_re2_1 = CropImage(img, RE2Pts, 0.16, crop_re2_1);
    cv::Mat img_re2_2 = CropImage(img, RE2Pts, 0.18, crop_re2_2);
    point2f RE2_1Pts = TwoLevelRE2_1(img_re2_1);
    point2f RE2_2Pts = TwoLevelRE2_2(img_re2_2);
    // TODO: convert position value
    RE2_1Pts = Convert(RE2_1Pts, crop_re2_1);
    RE2_2Pts = Convert(RE2_2Pts, crop_re2_2);
    point2f RE2_Avg;
    RE2_Avg.x = (RE2_1Pts.x + RE2_2Pts.x) / 2;
    RE2_Avg.y = (RE2_1Pts.y + RE2_2Pts.y) / 2;
    
    // point LM1
    point2f LM1Pts;
    LM1Pts.x = Pts[12];
    LM1Pts.y = Pts[13];
    cv::Rect crop_lm1_1;
    cv::Rect crop_lm1_2;
    cv::Mat img_lm1_1 = CropImage(img, LM1Pts, 0.16, crop_lm1_1);
    cv::Mat img_lm1_2 = CropImage(img, LM1Pts, 0.18, crop_lm1_2);
    point2f LM1_1Pts = TwoLevelLM1_1(img_lm1_1);
    point2f LM1_2Pts = TwoLevelLM1_2(img_lm1_2);
    // TODO: convert position value
    LM1_1Pts = Convert(LM1_1Pts, crop_lm1_1);
    LM1_2Pts = Convert(LM1_2Pts, crop_lm1_2);
    point2f LM1_Avg;
    LM1_Avg.x = (LM1_1Pts.x + LM1_2Pts.x) / 2;
    LM1_Avg.y = (LM1_1Pts.y + LM1_2Pts.y) / 2;
    
    // point RM1
    point2f RM1Pts;
    RM1Pts.x = Pts[14];
    RM1Pts.y = Pts[15];
    cv::Rect crop_rm1_1;
    cv::Rect crop_rm1_2;
    cv::Mat img_rm1_1 = CropImage(img, RM1Pts, 0.14, crop_rm1_1);
    cv::Mat img_rm1_2 = CropImage(img, RM1Pts, 0.16, crop_rm1_2);
    //cv::imwrite("/Users/betty/Documents/betty/examples/55555555.jpg", img_rm1_1);
    point2f RM1_1Pts = TwoLevelRM1_1(img_rm1_1);
    point2f RM1_2Pts = TwoLevelRM1_2(img_rm1_2);
    // TODO: convert position value
    RM1_1Pts = Convert(RM1_1Pts, crop_rm1_1);
    RM1_2Pts = Convert(RM1_2Pts, crop_rm1_2);
    point2f RM1_Avg;
    RM1_Avg.x = (RM1_1Pts.x + RM1_2Pts.x) / 2;
    RM1_Avg.y = (RM1_1Pts.y + RM1_2Pts.y) / 2;
    
    // point N1
    point2f N1Pts;
    N1Pts.x = Pts[0];
    N1Pts.y = Pts[1];
    cv::Rect crop_n1_1;
    cv::Rect crop_n1_2;
    cv::Mat img_n1_1 = CropImage(img, N1Pts, 0.16, crop_n1_1);
    cv::Mat img_n1_2 = CropImage(img, N1Pts, 0.18, crop_n1_2);
    point2f N1_1Pts = TwoLevelN1_1(img_n1_1);
    point2f N1_2Pts = TwoLevelN1_2(img_n1_2);
    // TODO: convert position value
    N1_1Pts = Convert(N1_1Pts, crop_n1_1);
    N1_2Pts = Convert(N1_2Pts, crop_n1_2);
    point2f N1_Avg;
    N1_Avg.x = (N1_1Pts.x + N1_2Pts.x) / 2;
    N1_Avg.y = (N1_1Pts.y + N1_2Pts.y) / 2;
    
    // point N2
    point2f N2Pts;
    N2Pts.x = Pts[2];
    N2Pts.y = Pts[3];
    cv::Rect crop_n2_1;
    cv::Rect crop_n2_2;
    cv::Mat img_n2_1 = CropImage(img, N2Pts, 0.16, crop_n2_1);
    cv::Mat img_n2_2 = CropImage(img, N2Pts, 0.18, crop_n2_2);
    point2f N2_1Pts = TwoLevelN2_1(img_n2_1);
    point2f N2_2Pts = TwoLevelN2_2(img_n2_2);
    // TODO: convert position value
    N2_1Pts = Convert(N2_1Pts, crop_n2_1);
    N2_2Pts = Convert(N2_2Pts, crop_n2_2);
    point2f N2_Avg;
    N2_Avg.x = (N2_1Pts.x + N2_2Pts.x) / 2;
    N2_Avg.y = (N2_1Pts.y + N2_2Pts.y) / 2;
    
    faces.FacePts[0] = N1_Avg;
    faces.FacePts[1] = N2_Avg;
    faces.FacePts[2] = LE1_Avg;
    faces.FacePts[3] = LE2_Avg;
    faces.FacePts[4] = RE1_Avg;
    faces.FacePts[5] = RE2_Avg;
    faces.FacePts[6] = LM1_Avg;
    faces.FacePts[7] = RM1_Avg;
    
    features.reserve(16);
    for (int i=0; i < NUM_PTS; i++) {
        float x, y;
        x = faces.FacePts[i].x;
        y = faces.FacePts[i].y;
        features.push_back(x);
        features.push_back(y);
    }
    return faces;
}

point2f Landmaker::TwoLevelLE1_1(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelLE1_1_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}

point2f Landmaker::TwoLevelLE1_2(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelLE1_2_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelLE2_1(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelLE2_1_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelLE2_2(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelLE2_2_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelLM1_1(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelLM_1_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelLM1_2(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelLM_2_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelN1_1(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelN1_1_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelN1_2(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelN1_2_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelN2_1(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelN2_1_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelN2_2(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelN2_2_obj.Classify(img , features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelRE1_1(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelRE1_1_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelRE1_2(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelRE1_2_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelRE2_1(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelRE2_1_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelRE2_2(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelRE2_2_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelRM1_1(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelRM_1_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}
point2f Landmaker::TwoLevelRM1_2(cv::Mat img) {
    std::vector<float> features;
    //TwoLevel.Classify(img, features);
    this->TwoLevelRM_2_obj.Classify(img, features);
    point2f result;
    result.x = features[0];
    result.y = features[1];
    return result;
}

cv::Mat Landmaker::CropImage(cv::Mat img, point2f FacePts, float scale, cv::Rect & crop_rect) {
    // TODO: crop image for level2 model
    //cv::Mat face = cv::resize();
    BBox bbox;
    float x = FacePts.x * this->faceBox.width + this->faceBox.x;
    float y = FacePts.y * this->faceBox.height + this->faceBox.y;
    cv::Rect roiRec;
    roiRec.x = x - this->faceBox.width * scale;
    roiRec.width = x + this->faceBox.width * scale - roiRec.x;
    roiRec.y = y - this->faceBox.height * scale;
    roiRec.height = y + this->faceBox.height * scale - roiRec.y;
    
    int x1 = MAX(roiRec.x,0);
    int y1 = MAX(roiRec.y,0);
    int x2 = MIN(roiRec.x + roiRec.width, img.cols);
    int y2 = MIN(roiRec.y + roiRec.height, img.rows);
    
    if (x1 < x2 && y1 < y2)
    {
        crop_rect.x = MAX( roiRec.x, 0);
        crop_rect.y = MAX( roiRec.y, 0);
        crop_rect.width  = MIN( roiRec.x + roiRec.width,  img.cols ) - crop_rect.x;
        crop_rect.height = MIN( roiRec.y + roiRec.height, img.rows) - crop_rect.y;
    }
    
    cv::Mat cropImg;
    if (roiRec.area() > 0) {
        cropImg = img(crop_rect);
        //std::cout << "after crop." << std::endl;
    }
    else
        cropImg = img;

    return cropImg;
}

point2f Landmaker::Convert(point2f Pts, cv::Rect crop_rect) {
    point2f convertPts;
    convertPts.x = Pts.x * crop_rect.width + crop_rect.x;
    convertPts.y = Pts.y * crop_rect.height + crop_rect.y;
    return convertPts;
}

cv::Mat Landmaker::GenerateRoi(cv::Mat img) {
    
    //cv::imwrite("/Users/betty/Documents/betty/examples/GenerateBoundingBox.jpg",img);
    cv::Mat roiImage;
    int x1 = MAX(this->faceBox.x,0);
    int y1 = MAX(this->faceBox.y,0);
    int x2 = MIN(this->faceBox.x + this->faceBox.width, img.cols);
    int y2 = MIN(this->faceBox.y + this->faceBox.height, img.rows);
    
    if (x1 < x2 && y1 < y2)
    {
        this->faceBox.x = MAX( this->faceBox.x, 0);
        this->faceBox.y = MAX( this->faceBox.y, 0);
        this->faceBox.width  = MIN( this->faceBox.x + this->faceBox.width,  img.cols ) - this->faceBox.x;
        this->faceBox.height = MIN( this->faceBox.y + this->faceBox.height, img.rows) - this->faceBox.y;
    }
    if (this->faceBox.area() > 0) {
        roiImage = img(this->faceBox);
    }
    else
        roiImage = img;
    
    return roiImage;
}

std::string Landmaker::getConfig(string level_id, const string data_path)
{
    string config_file = data_path + "/" + level_id;
    return config_file;
}


