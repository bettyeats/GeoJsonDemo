//
//  gif2jpg.hpp
//  CutImageForTurning
//
//  Created by MrWang on 2017/1/11.
//  Copyright © 2017年 RemarkMedia. All rights reserved.
//

#ifndef gif2jpg_hpp
#define gif2jpg_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

void GIF2RGB(std::string buffer, bool OneFileFlag, std::vector<cv::Mat> &dst_imgs);

#endif /* gif2jpg_hpp */
