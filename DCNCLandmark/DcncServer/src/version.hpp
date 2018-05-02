//
//  version.hpp
//  CaffeClassifyServer
//
//  Created by MrWang on 2017/10/11.
//  Copyright © 2017年 RemarkMedia. All rights reserved.
//

#ifndef version_hpp
#define version_hpp

#include <stdio.h>

/*************************************************************************
# version 01 多线程-色情图片识别
# version 02 删除model文件夹中的默认模型，使用外部映射的方式输入模型，修改通道模型的bug
# version 03 增加黑屏判断-色情图片色别定制
# version 04 删除黑屏判断-同步CPU、GPU版本，增加特征提取接口
# version 05 临时版本，特征值参数长度为0
# version 06 增加layer.txt文件控制特征值提取层
# version 07 增加多GPU支持
# version 08 集成DES模型解密算法
# version 09 通过layer.txt文件可以同时提取多层网络的特征值
# version 10 通过mean.dat文件设置网络mean参数，文件内容可以是*.binaryproto，
             也可以是*.txt的内容，如：127.5\r\n127.5\r\n127.5\r\n
# version 11 整理所有的配置参数，形成config.dat文件，全部内容如下，如有需要等号后都可以为空
             label=lists.txt
             mean=mean.binaryproto
             mean1=128
             mean2=128
             mean3=128
             scale=0.0078125
             layer1=fc5
             layer2=fc6
             layer3=fc7
 *************************************************************************/

char g_version[] = "11";

#endif /* version_hpp */
