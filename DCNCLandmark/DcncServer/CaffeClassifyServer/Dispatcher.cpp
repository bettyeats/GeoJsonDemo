//
//  Dispatcher.cpp
//  DCNCLandmark
//
//  Created by betty on 2018/4/26.
//  Copyright © 2018年 betty. All rights reserved.
//

#include "Dispatcher.hpp"

Dispatcher::Dispatcher() {
    reset_track = true;
    Init();
}

// TODO：1st frame use opencv and use DcncAlign obj to detect
void Dispatcher::Init() {
   
    string file_path = "/Users/betty/res/haarcascades/haarcascade_frontalface_alt2.xml";
    this->face_cascade.load(file_path);
    if (face_cascade.empty())
        std::cout << "face detect load failure." << std::endl;
    string align_path = "/Users/betty/res/DcncAlign";
    string data_path = "/Users/betty/res/Dcnc";
    mDcnc = DcncAlign(0, align_path);
    mLandmaker = Landmaker(0, data_path);
}

void Dispatcher::GetBBox(cv::Mat img, cv::Rect& faceBox) {
    
    //cv::imwrite("/Users/betty/Documents/betty/examples/bbox.jpg", img);
    if (this->reset_track || faceBox.area() < 1000 || faceBox.area() > 1000000) {
        std::vector<cv::Rect> mFaceRects;
        this->face_cascade.detectMultiScale(img, mFaceRects, 1.3, 2, 0, cv::Size(50, 50));
        if (mFaceRects.size() > 0) {
            std::cout << "Reset EEEEEEEEEE" << std::endl;
            for (int i=0 ; i < mFaceRects.size(); i++) {
                //if (faceBox.area() < mFaceRects[i].area())
                faceBox = mFaceRects[i];
                
            }
            //cv::rectangle(img, faceBox, cv::Scalar(0, 0, 255));
            //cv::imwrite("/Users/betty/Documents/betty/examples/face_cascade.jpg",img);
            this->reset_track = false;
        }
        else {
            faceBox = cv::Rect(0,0,0,0);
        }
    }
}

void Dispatcher::Detect(cv::Mat img, vector<float> &features) {
    
    cv::Mat image, affineImg;
    
    cv::cvtColor(img, image, CV_BGR2GRAY);
    // TODO: 1st frame use opencv & 5p model
    if (reset_track)
    {
        GetBBox(image, faceBox);
        mDcnc.Detect(image, faceBox, base_landmarks);
    }
    else
    {
        // TODO: align & 8p model
        affineImg = CropImg(image);
        mDcnc.Detect(affineImg, faceBox, base_landmarks);
        // TODO:: inver
        invertAffineTransform(trans, trans_inv);
        std::vector<double> Param;
        for(int i = 0; i < trans_inv.rows; i++)
        {
            const double* pData = trans_inv.ptr<double>(i);   //第i+1行的所有元素
            for(int j = 0; j < trans_inv.cols; j++) {
                Param.push_back(pData[j]);
            }
        }
        
        for ( int i=0; i < base_landmarks.size(); i+=2) {
            float x = base_landmarks[i];
            float y = base_landmarks[i+1];
            base_landmarks[i] =  x * Param[0] + y * Param[1] + Param[2];
            base_landmarks[i+1] = x * Param[3] + y * Param[4] + Param[5];
        }
        mLandmaker.Detect(affineImg, faceBox, features);
        
        for ( int i=0; i < features.size(); i+=2) {
            float x = features[i];
            float y = features[i+1];
            features[i] =  x * Param[0] + y * Param[1] + Param[2];
            features[i+1] = x * Param[3] + y * Param[4] + Param[5];
        }
        
        float shape[16];
        int j = 0;
        for (int i=0; i < 16; i+=2) {
            shape[j] = features[i];
            shape[j + 8] = features[i+1];
            j++;
        }
        
        // TODO: calculate angle
        cv::Mat L(1,16,CV_8UC1,shape);
        cv::Vec3d eav = EstimatePose(L);
        float pitch = eav[0];
        float yaw = eav[1];
        float roll = eav[2];
        
        features.push_back(pitch);
        features.push_back(yaw);
        features.push_back(roll);
        // TODO: calculate face width
        float w = EstimateFaceWidth(eav, L);
        features.push_back(w);
    }
}

cv::Mat Dispatcher::CropImg(cv::Mat img) {
    
    cv::Mat affineImg;
    std::vector<cv::Point2d> template_points = {  {47.00508563,  39.97558339 },
        {112.99491437,  39.97558339 },
        {80.  ,       77.24552512 },
        {50.91132736, 109.74190521 },
        {109.08867264, 109.74190521 }};
    
    std::vector<cv::Point2d> target_points;
    mFaceAlig.TransTemplatePoints(template_points, 150, 150, 0, 0, target_points);
    std::vector<cv::Point2d> points(5);
    int j = 0;
    for (int i = 0; i < base_landmarks.size(); i+=2 )
    {
        float x =  base_landmarks[i];
        float y =  base_landmarks[i+1];
        points[j] = cv::Point2d(x, y);
        j++;
    }
    
    trans = mFaceAlig.FindSimilarityTransform(points, target_points, trans_inv);
    cv::warpAffine(img, affineImg, trans, cv::Size(150, 150));
    cv::imwrite("/Users/betty/Documents/betty/examples/affinexxxx.jpg", affineImg);
    cv::imwrite("/Users/betty/Documents/betty/examples/raw.jpg", img);
    faceBox = cv::Rect(0,0,150,150);
    return affineImg;
}
cv::Vec3d Dispatcher::EstimatePose(cv::Mat &shape){
    // 68 points
    //static int HeadPosePointIndexs[] = {36,39,42,45,30,48,54};
    // 8 points
    static int HeadPosePointIndexs[] = {2,3,4,5,1,6,7};
    // for test
    //static int HeadPosePointIndexs[] = {0,1,2,3,4,5,6};
    int *estimateHeadPosePointIndexs = HeadPosePointIndexs;
    static float estimateHeadPose2dArray[] = {
        -0.208764,-0.140359,0.458815,0.106082,0.00859783,-0.0866249,-0.443304,-0.00551231,-0.0697294,
        -0.157724,-0.173532,0.16253,0.0935172,-0.0280447,0.016427,-0.162489,-0.0468956,-0.102772,
        0.126487,-0.164141,0.184245,0.101047,0.0104349,-0.0243688,-0.183127,0.0267416,0.117526,
        0.201744,-0.051405,0.498323,0.0341851,-0.0126043,0.0578142,-0.490372,0.0244975,0.0670094,
        0.0244522,-0.211899,-1.73645,0.0873952,0.00189387,0.0850161,1.72599,0.00521321,0.0315345,
        -0.122839,0.405878,0.28964,-0.23045,0.0212364,-0.0533548,-0.290354,0.0718529,-0.176586,
        0.136662,0.335455,0.142905,-0.191773,-0.00149495,0.00509046,-0.156346,-0.0759126,0.133053,
        -0.0393198,0.307292,0.185202,-0.446933,-0.0789959,0.29604,-0.190589,-0.407886,0.0269739,
        -0.00319206,0.141906,0.143748,-0.194121,-0.0809829,0.0443648,-0.157001,-0.0928255,0.0334674,
        -0.0155408,-0.145267,-0.146458,0.205672,-0.111508,0.0481617,0.142516,-0.0820573,0.0329081,
        -0.0520549,-0.329935,-0.231104,0.451872,-0.140248,0.294419,0.223746,-0.381816,0.0223632,
        0.176198,-0.00558382,0.0509544,0.0258391,0.050704,-1.10825,-0.0198969,1.1124,0.189531,
        -0.0352285,0.163014,0.0842186,-0.24742,0.199899,0.228204,-0.0721214,-0.0561584,-0.157876,
        -0.0308544,-0.131422,-0.0865534,0.205083,0.161144,0.197055,0.0733392,-0.0916629,-0.147355,
        0.527424,-0.0592165,0.0150818,0.0603236,0.640014,-0.0714241,-0.0199933,-0.261328,0.891053};
    cv::Mat estimateHeadPoseMat = cv::Mat(15,9,CV_32FC1,estimateHeadPose2dArray);
    static float estimateHeadPose2dArray2[] = {
        0.139791,27.4028,7.02636,
        -2.48207,9.59384,6.03758,
        1.27402,10.4795,6.20801,
        1.17406,29.1886,1.67768,
        0.306761,-103.832,5.66238,
        4.78663,17.8726,-15.3623,
        -5.20016,9.29488,-11.2495,
        -25.1704,10.8649,-29.4877,
        -5.62572,9.0871,-12.0982,
        -5.19707,-8.25251,13.3965,
        -23.6643,-13.1348,29.4322,
        67.239,0.666896,1.84304,
        -2.83223,4.56333,-15.885,
        -4.74948,-3.79454,12.7986,
        -16.1,1.47175,4.03941 };
    cv::Mat estimateHeadPoseMat2 = cv::Mat(15,3,CV_32FC1,estimateHeadPose2dArray2);
    
    if(shape.empty())
        return NULL;
    static const int samplePdim = 7;
    float miny = 10000000000.0f;
    float maxy = 0.0f;
    float sumx = 0.0f;
    float sumy = 0.0f;
    for(int i=0; i<samplePdim; i++){
        sumx += shape.at<float>(estimateHeadPosePointIndexs[i]);
        float y = shape.at<float>(estimateHeadPosePointIndexs[i]+shape.cols/2);
        sumy += y;
        if(miny > y)
            miny = y;
        if(maxy < y)
            maxy = y;
    }
    float dist = maxy - miny;
    sumx = sumx/samplePdim;
    sumy = sumy/samplePdim;
    static cv::Mat tmp(1, 2*samplePdim+1, CV_32FC1);
    for(int i=0; i<samplePdim; i++){
        tmp.at<float>(i) = (shape.at<float>(estimateHeadPosePointIndexs[i])-sumx)/dist;
        tmp.at<float>(i+samplePdim) = (shape.at<float>(estimateHeadPosePointIndexs[i]+shape.cols/2)-sumy)/dist;
    }
    tmp.at<float>(2*samplePdim) = 1.0f;
    cv::Mat predict = tmp*estimateHeadPoseMat2;
    cv::Vec3d rot;
    rot[0] = predict.at<float>(0); //pitch
    rot[1] = predict.at<float>(1); //yaw
    rot[2] = predict.at<float>(2); //roll
    return rot;
}

float Dispatcher::EstimateFaceWidth(cv::Vec3d rot, cv::Mat landmarks) {
    
    float face_width;
    float error = 0.0f;
    int eyes_indexs[2] = {2,5};
    if (landmarks.empty())
        return error;
    float lx = landmarks.at<float>(eyes_indexs[0]);
    float ly = landmarks.at<float>(eyes_indexs[0] + landmarks.cols/2);
    float rx = landmarks.at<float>(eyes_indexs[1]);
    float ry = landmarks.at<float>(eyes_indexs[1] + landmarks.cols/2);
    float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
    float map = cos(DegToRad(rot[1]));
    face_width = fabs (1.47 * (distance / map));
    return face_width;
}
