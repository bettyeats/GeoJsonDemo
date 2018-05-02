//
//  comm.cpp
//  CaffeClassifyServer
//
//  Created by MrWang on 2017/9/26.
//  Copyright © 2017年 RemarkMedia. All rights reserved.
//

#include "comm.hpp"
#include <sys/time.h>

long getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
