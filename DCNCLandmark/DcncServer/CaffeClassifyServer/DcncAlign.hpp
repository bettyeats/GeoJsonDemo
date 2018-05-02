//
//  DcncAlign.hpp
//  DCNCLandmark
//
//  Created by betty on 2018/4/20.
//  Copyright © 2018年 betty. All rights reserved.
//

#ifndef DcncAlign_hpp
#define DcncAlign_hpp

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

typedef enum {
    PIX_FMT_GRAY8,   ///< Y    1        8bpp ( 单通道8bit灰度像素 )
    PIX_FMT_YUV420P, ///< YUV  4:2:0   12bpp ( 3通道, 一个亮度通道, 另两个为U分量和V分量通道, 所有通道都是连续的 )
    PIX_FMT_NV12,    ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道, 另一通道为UV分量交错 )
    PIX_FMT_NV21,    ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道, 另一通道为VU分量交错 )
    PIX_FMT_BGRA8888,///< BGRA 8:8:8:8 32bpp ( 4通道32bit BGRA 像素 )
    PIX_FMT_BGR888,  ///< BGR  8:8:8   24bpp ( 3通道24bit BGR 像素 )
    PIX_FMT_RGBA8888 ///< BGRA 8:8:8:8 32bpp ( 4通道32bit RGBA 像素 )
} pixel_format;

/// image rotate type definition
typedef enum {
    CLOCKWISE_ROTATE_0 = 0,  ///< 图像不需要旋转，图像中的人脸为正脸
    CLOCKWISE_ROTATE_90 = 1, ///< 图像需要顺时针旋转90度，使图像中的人脸为正
    CLOCKWISE_ROTATE_180 = 2,///< 图像需要顺时针旋转180度，使图像中的人脸为正
    CLOCKWISE_ROTATE_270 = 3 ///< 图像需要顺时针旋转270度，使图像中的人脸为正
} rotate_type;

class DcncAlign
{
public:
    explicit DcncAlign(int gpu_id,
                       const string &data_path);
    DcncAlign();
    
public:
    
    void Init();
    void Detect(const cv::Mat& image, cv::Rect bbox, std::vector<float> &features);
    cv::Mat CropImage(cv::Mat img, point2f FacePts, float scale, cv::Rect & crop_rect);
    std::string getConfig(string level_id, const string data_path);
    std::vector<float> Convert(std::vector<float> Pts, cv::Rect crop_rect);
    point2f ConvertPt(point2f Pts, cv::Rect crop_rect);
 
private:
    static const string modelnames[10];
    std::vector<float> Level(cv::Mat img, std::vector<float> landmark, std::vector<Classifier> cnns, int level_no);
    
    cv::Rect faceBox; // bounding box
    int gpu_id;
    std::string parent_path;

    std::vector<Classifier> Level1;
    std::vector<Classifier> Level2;
    std::vector<Classifier> Level3;
    
};

#endif /* DcncAlign_hpp */
