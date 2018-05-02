//
//  main.cpp
//  DCNCLandmark
//
//  Created by betty on 2018/4/12.
//  Copyright © 2018年 betty. All rights reserved.
//

#include <iostream>
#include <sys/time.h>
#include "Dispatcher.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void bufferEncode(const cv::Mat img, std::string &buff)
{
    buff.clear();
    
    std::vector<uchar> img_buff;
    cv::imencode(".jpg", img, img_buff);
    
    buff.append((char*)img_buff.data(), img_buff.size());
}

void test() {
    cv::Mat test = cv::imread("img.jpg");
    cv::imwrite("testtest.jpg", test);
    cv::imshow("test", test);
    cv::waitKey(0);
}

int main(int argc, const char * argv[]) {
    // insert code here...
    //test();
    cv::VideoCapture mCamera("/Users/betty/Documents/betty/examples/test.mp4");
    if(!mCamera.isOpened())
    {
        std::cout << "Camera opening failed ..." << std::endl;
        system("pause");
        return 0;
    }
    cv::Mat frame, img_show, img_rz, img_frame, img_back;
    std::string data;
    std::vector<float> features;
    Dispatcher mDispatcher;
  
    // TODO：get image ready
    while (true)
    {
        int count = 3;
        mCamera >> frame;
        //video >> img_rz;
        //cv::resize(frame, img_rz, cv::Size(600, 600*frame.rows/frame.cols), 0, 0, CV_INTER_AREA);
        cv::resize(frame, img_rz, cv::Size(600, 360), 0, 0, CV_INTER_AREA);
        
        // TODO: get landmarks and show
        bufferEncode(img_rz, data);
       
        cv::imwrite("img_rz.jpg", img_rz);
        mDispatcher.Detect(img_rz, features);
      
        if (features.size() == 0)
            continue;
        for (int i = 0; i < 16; i+=2)
        {
            float x = features[i];
            float y = features[i + 1];
            cv::circle(img_rz,cv::Point(x,y), 2, cv::Scalar(0, 255, 0), -1);
        }
            
        float pitch = features[16];
        float yaw = features[17];
        float roll = features[18];
        float width = features[19];
        std::cout << " yaw: " << yaw << ", pitch: " << pitch << ", roll: " << roll << "\n" <<std::endl;
        std::cout << " width: "<< width << "\n" << std::endl;
        
        //cv::imshow("Camera", img_rz);
        cv::imwrite("test.jpg", img_rz);
        cv::waitKey(1);
        if(27 == cv::waitKey(1))
        {
            mCamera.release();
            cv::destroyAllWindows();
            break;
        }
    }
    return 0;
}



