//
//  Server.h
//  AuthenticationServer
//
//  Created by MrWang on 16/5/31.
//  Copyright © 2016年 RemarkMedia. All rights reserved.
//

#ifndef Server_h
#define Server_h

#include <stdio.h>
#include <pthread.h>
#include <opencv2/opencv.hpp>

#include "Service.h"

struct GPU_HANDLE
{
    bool                brun;
    int                 gpu_id;
    pthread_t           thread;
    pthread_mutex_t     mutex;
    pthread_cond_t      cond;
    cv::Mat             image;
    std::vector<double>  val;
    std::vector<std::string>  type;
    std::vector<float>  features;
};

class ServiceHandler : virtual public ServiceIf
{
public:
    
    ServiceHandler(int threadnum);
    
    ~ServiceHandler();
    
    void recognize_Image(RST_RCGZ& _return, const std::string& img);
    
    void get_Feature(std::vector<double> & _return, const std::string& img);
    
private:
    
    GPU_HANDLE *m_halg;
    
    int m_threadnum;
    unsigned long *m_threadid;
    
    int get_ProcessID();
};


#endif /* Server_h */
