//
//  cryptopp.hpp
//  CaffeCompare
//
//  Created by MrWang on 16/11/14.
//  Copyright © 2016年 WangFei. All rights reserved.
//

#ifndef cryptopp_hpp
#define cryptopp_hpp

#include <stdio.h>
#include "des.h"

bool desDecryption(unsigned char *buff_in, unsigned char *buff_out, long size);

#endif /* cryptopp_hpp */
