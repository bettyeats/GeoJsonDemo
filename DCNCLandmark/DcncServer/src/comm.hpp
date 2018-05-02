//
//  comm.hpp
//  CaffeClassifyServer
//
//  Created by MrWang on 2017/9/26.
//  Copyright © 2017年 RemarkMedia. All rights reserved.
//

#ifndef comm_hpp
#define comm_hpp

#include <stdio.h>

//#define TIME_TEST

#ifdef TIME_TEST
#define TIME_START() tt_begin = getCurrentTime();
#define TIME_END(name) tt_end = getCurrentTime(); printf("[%s] is taking %ldms\n", name, tt_end - tt_begin); tt_begin = getCurrentTime();
#else
#define TIME_START()
#define TIME_END(name)
#endif


long getCurrentTime();

#endif /* comm_hpp */
