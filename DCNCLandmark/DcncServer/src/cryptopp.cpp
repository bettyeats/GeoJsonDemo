//
//  cryptopp.cpp
//  CaffeCompare
//
//  Created by MrWang on 16/11/14.
//  Copyright © 2016年 WangFei. All rights reserved.
//

#include "cryptopp.h"

using namespace CryptoPP;

unsigned char DES_KEY[256] = "KanKanApp@RemarkHoldings-China-ChengDu";


bool desDecryption(unsigned char *buff_in, unsigned char *buff_out, long size)
{
    long offset, num = 0;
    DESDecryption decrypt;
    
    decrypt.SetKey(DES_KEY, 8);
    
    num = size/decrypt.BLOCKSIZE;
    for (int i = 0; i < num; i++)
    {
        offset = i*decrypt.BLOCKSIZE;
        decrypt.ProcessBlock(buff_in + offset, buff_out + offset);
    }
    
    return true;
}
