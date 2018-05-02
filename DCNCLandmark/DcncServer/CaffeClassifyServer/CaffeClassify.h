//
//  CaffeClassify.h
//  CaffeClassify
//
//  Created by MrWang on 15/10/15.
//  Copyright (c) 2015年 WangFei. All rights reserved.
//

#ifndef __CaffeClassify__CaffeClassify__
#define __CaffeClassify__CaffeClassify__

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "helper.h"

#define DOUBLE 0
#define UINT32   0

#define MAX_FEALAYER_NUM 3

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
public:
    Classifier();
    Classifier(int gpu_id, const string& data_path);
    
    std::vector<Prediction> Classify(const cv::Mat& img, std::vector<float> &features, int N = 5);
    
private:
    void getConfigures(const string data_path, string &model_file, string &trained_file,
                       string &mean_file, float means[3], string &lable_file, string layers[3]);
               
    void SetMean(const string& mean_file, float means[]);
    
    void SetFeaLayer(const string layers[]);
    
    void SetLabels(const string& label_file);
    
    void Predict(const cv::Mat& img, std::vector<float> &features, std::vector<float> &outputs);
    
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    
    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);
    
private:
#if DOUBLE
    shared_ptr<Net<double> > net_;
    shared_ptr<Blob<double> > feaLayer_[MAX_FEALAYER_NUM];
#elif UINT32
    shared_ptr<Net<unsigned int> > net_;
    shared_ptr<Blob<unsigned int> > feaLayer_[MAX_FEALAYER_NUM];
#else
    shared_ptr<Net<float> > net_;
    shared_ptr<Blob<float> > feaLayer_[MAX_FEALAYER_NUM];
#endif
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    float scale_;
    std::vector<string> labels_;
};

#endif /* defined(__CaffeClassify__CaffeClassify__) */



