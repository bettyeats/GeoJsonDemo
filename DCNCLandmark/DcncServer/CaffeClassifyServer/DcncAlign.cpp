//
//  DcncAlign.cpp
//  DCNCLandmark
//
//  Created by betty on 2018/4/20.
//  Copyright © 2018年 betty. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "DcncAlign.hpp"
#include "CaffeClassify.h"

//#define DES_DECODE

#ifdef DES_DECODE
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/coded_stream.h"
#include "cryptopp.h"
#endif
#include "DcncAlign.hpp"

using namespace std;
const string DcncAlign::modelnames[10] = {"LE1", "LE2", "RE1", "RE2", "N1", "N2", "LM1", "LM2", "RM1", "RM2"};

DcncAlign::DcncAlign() {
}

DcncAlign::DcncAlign (int gpu_id, const string& data_path){
    this->gpu_id = gpu_id;
    this->parent_path = data_path;
    Init();
}

void DcncAlign::Init() {
    string config_path = getConfig("OneLevelF", this->parent_path);
    Level1.push_back(Classifier(this->gpu_id, config_path));
    std::cout << "after 1_F" << std::endl;
    std::cout << "1_F config " << config_path << std::endl;
    config_path = getConfig("OneLevelEN", this->parent_path);
    std::cout << "1_EN config " << config_path << std::endl;
    Level1.push_back(Classifier(this->gpu_id, config_path));
    std::cout << "after 1_EN" << std::endl;
    std::cout << "1_EN config " << config_path << std::endl;
    config_path = getConfig("OneLevelNM", this->parent_path);
    Level1.push_back(Classifier(this->gpu_id, config_path));
    
    for(int i = 0 ; i < 10 ; i++) {
        string level2_path = string("2_") + modelnames[i];
        string level3_path = string("3_") + modelnames[i];
        std::cout << level2_path << std::endl;
        config_path = getConfig(level2_path, this->parent_path);
        Level2.push_back(Classifier(this->gpu_id, config_path));
        config_path = getConfig(level3_path, this->parent_path);
        Level3.push_back(Classifier(this->gpu_id, config_path));
    }
}

std::vector<float> DcncAlign::Convert(std::vector<float> Pts, cv::Rect crop_rect) {
    
    for (int i=0; i < Pts.size(); i+=2) {
        Pts[i] = Pts[i] * crop_rect.width + crop_rect.x;
        Pts[i+1] = Pts[i+1] * crop_rect.height + crop_rect.y;
    }
    return Pts;
}

point2f DcncAlign::ConvertPt(point2f Pts, cv::Rect crop_rect) {
    
    point2f convertPts;
    convertPts.x = Pts.x * crop_rect.width + crop_rect.x;
    convertPts.y = Pts.y * crop_rect.height + crop_rect.y;
    return convertPts;
}

void DcncAlign::Detect(const cv::Mat& image, cv::Rect bbox, std::vector<float> &features){
    
    features.clear();
    this->faceBox = bbox;
    
    cv::Rect facearea, eyenosearea, nosemoutharea;
    vector<int> facearea_v;
    facearea_v.push_back(faceBox.x);
    facearea_v.push_back(faceBox.x + faceBox.width - 1);
    facearea_v.push_back(faceBox.y);
    facearea_v.push_back(faceBox.y + faceBox.height - 1);
    facearea.x = facearea_v[0];
    facearea.y = facearea_v[2];
    facearea.width = facearea_v[1] - facearea_v[0];
    facearea.height = facearea_v[3] - facearea_v[2];
    
    vector<int> eyenosearea_v = facearea_v;
    eyenosearea_v[3] = eyenosearea_v[2] + (31.0 / 39) * (eyenosearea_v[3] - eyenosearea_v[2]);
    eyenosearea.x = eyenosearea_v[0];
    eyenosearea.y = eyenosearea_v[2];
    eyenosearea.width = eyenosearea_v[1] - eyenosearea_v[0];
    eyenosearea.height = eyenosearea_v[3] - eyenosearea_v[2];
    
    vector<int> nosemoutharea_v = facearea_v;
    nosemoutharea_v[2] = nosemoutharea_v[2] + (8.0 / 39) * (nosemoutharea_v[3] - nosemoutharea_v[2]);
    nosemoutharea.x = nosemoutharea_v[0];
    nosemoutharea.y = nosemoutharea_v[2];
    nosemoutharea.width = nosemoutharea_v[1] - nosemoutharea_v[0];
    nosemoutharea.height = nosemoutharea_v[3] - nosemoutharea_v[2];
    
    vector<float> face_landmark, eyesnose_landmark, nosemouth_landmark;
    Level1[0].Classify(image(facearea), face_landmark);
    Level1[1].Classify(image(eyenosearea), eyesnose_landmark);
    Level1[2].Classify(image(nosemoutharea), nosemouth_landmark);
    assert(face_landmark.size() == 10);
    assert(eyesnose_landmark.size() == 6);
    assert(nosemouth_landmark.size() == 6);
    // TODO: Convert
    face_landmark = Convert(face_landmark, facearea);
    eyesnose_landmark = Convert(eyesnose_landmark, eyenosearea);
    nosemouth_landmark = Convert(nosemouth_landmark, nosemoutharea);
    //LE landmark
    face_landmark[0] = (face_landmark[0] + eyesnose_landmark[0]) / 2;
    face_landmark[1] = (face_landmark[1] + eyesnose_landmark[1]) / 2;
    //RE landmark
    face_landmark[2] = (face_landmark[2] + eyesnose_landmark[2]) / 2;
    face_landmark[3] = (face_landmark[3] + eyesnose_landmark[3]) / 2;
    //N landmark
    face_landmark[4] = (face_landmark[4] + eyesnose_landmark[4] + nosemouth_landmark[0]) / 3;
    face_landmark[5] = (face_landmark[5] + eyesnose_landmark[5] + nosemouth_landmark[1]) / 3;
    //LM landmark
    face_landmark[6] = (face_landmark[6] + nosemouth_landmark[2]) / 2;
    face_landmark[7] = (face_landmark[7] + nosemouth_landmark[3]) / 2;
    //RM landmark
    face_landmark[8] = (face_landmark[8] + nosemouth_landmark[4]) / 2;
    face_landmark[9] = (face_landmark[9] + nosemouth_landmark[5]) / 2;
  
    // Level2&3
    face_landmark = Level(image, face_landmark, Level2, 2);
    face_landmark = Level(image, face_landmark, Level3, 3);
    
    for (int i=0; i < face_landmark.size(); i++) {
        features.push_back(face_landmark[i]);
    }
}

std::vector<float> DcncAlign::Level(cv::Mat img,
                               std::vector<float> landmark,
                               std::vector<Classifier> cnns,
                               int level_no) {
    std::vector<float> padding;
    std::vector<float> features1, features2, result;
    if (level_no == 2) {
        padding.push_back(0.16);
        padding.push_back(0.18);
    } else {
        padding.push_back(0.11);
        padding.push_back(0.12);
    }
   
    assert(img.type() == CV_8UC1);
    assert(landmark.size() == 10);
    point2f Pts, outPts1, outPts2;
    cv::Rect crop_rect1, crop_rect2;
    cv::Mat crop_img1, crop_img2;
    // TODO: Modify
    for (int i = 0; i < 10; i+=2) {
        Pts.x = landmark[i];
        Pts.y = landmark[i+1];
        cv::imwrite("img.jpg", img);
        
        crop_img1 = CropImage(img, Pts, padding[0], crop_rect1);
        crop_img2 = CropImage(img, Pts, padding[1], crop_rect2);
        cv::imwrite("crop_img1.jpg", crop_img1);
        cnns[i].Classify(crop_img1, features1);
        cnns[i+1].Classify(crop_img2, features2);
        outPts1.x = features1[0];
        outPts1.y = features1[1];
        outPts2.x = features2[0];
        outPts2.y = features2[1];
        outPts1 = this->ConvertPt(outPts1, crop_rect1);
        outPts2 = this->ConvertPt(outPts2, crop_rect2);
        result.push_back((outPts1.x + outPts2.x) / 2);
        result.push_back((outPts1.y + outPts2.y) / 2);
    }
    return result;
}

std::string DcncAlign::getConfig(string level_id, const string data_path)
{
    string config_file = data_path + "/" + level_id;
    return config_file;
}

cv::Mat DcncAlign::CropImage(cv::Mat img, point2f FacePts, float scale, cv::Rect & crop_rect) {
    // TODO: crop image for level2 model
    //cv::Mat face = cv::resize();
    BBox bbox;
    //float x = FacePts.x * this->faceBox.width + this->faceBox.x;
    //float y = FacePts.y * this->faceBox.height + this->faceBox.y;
    float x = FacePts.x;
    float y = FacePts.y;
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



