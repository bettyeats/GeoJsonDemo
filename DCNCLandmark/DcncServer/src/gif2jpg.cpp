//
//  gif2jpg.cpp
//  CutImageForTurning
//
//  Created by MrWang on 2017/1/11.
//  Copyright © 2017年 RemarkMedia. All rights reserved.
//

#include <string.h>
#include <stdlib.h>
#include <gif_lib.h>

#include "getarg.h"
#include "gif2jpg.h"

/* Holds the information for the input function.  */
typedef struct gs_gif_input_src
{
    const char      *data;
    unsigned long    length;
    unsigned long    pos;
} gs_gif_input_src;

/* Provides data for the gif library.  */
static int gs_gif_input(GifFileType *file, GifByteType *buffer, int len)
{
    /* according the the libungif sources, this functions has
     to act like fread. */
    int bytesRead;
    gs_gif_input_src *src = (gs_gif_input_src *)file->UserData;
    
    if (src->pos < src->length)
    {
        if ((src->pos + len) > src->length)
        {
            bytesRead = (int)(src->length - src->pos);
        }
        else
        {
            bytesRead = len;
        }
        
        /* We have to copy the data here, looking at
         the libungif source makes this clear.  */
        memcpy(buffer, src->data + src->pos, bytesRead);
        src->pos = src->pos + bytesRead;
    }
    else
    {
        bytesRead = 0;
    }
    
    return bytesRead;
}


/* Initialze a new input source to be used with
 gs_gif_input. The passed structure has to be
 allocated outside this function. */
static void gs_gif_init_input_source(gs_gif_input_src *src, std::string buffer)
{
    src->data   = buffer.data();
    src->length = buffer.length();
    src->pos    = 0;
}

/******************************************************************************
 The real screen dumping routine.
 ******************************************************************************/
void DumpScreen2RGB(cv::Mat &image, int trans_color, int OneFileFlag,
                           ColorMapObject *ColorMap,
                           GifRowType *ScreenBuffer,
                           int ScreenWidth, int ScreenHeight)
{
    int i, j;
    GifRowType GifRow;
    GifColorType *ColorMapEntry;
    FILE *rgbfp[3];
    
    if (!image.empty()) {
        if (!OneFileFlag){
            static char *Postfixes[] = { ".R", ".G", ".B" };
            char OneFileName[80];
            
            for (i = 0; i < 3; i++) {
                strncpy(OneFileName, "", sizeof(OneFileName)-1);
                /* cppcheck-suppress uninitstring */
                strncat(OneFileName, Postfixes[i],
                        sizeof(OneFileName) - 1 - strlen(OneFileName));
                
                if ((rgbfp[i] = fopen(OneFileName, "wb")) == NULL) {
                    GIF_EXIT("Can't open input file name.");
                }
            }
        }
    } else {
        OneFileFlag = true;
        
#ifdef _WIN32
        _setmode(1, O_BINARY);
#endif /* _WIN32 */
        
        rgbfp[0] = stdout;
    }
    
    if (OneFileFlag) {
        unsigned char *BufferP;
        
        for (i = 0; i < ScreenHeight; i++) {
            GifRow = ScreenBuffer[i];
            BufferP = image.data + i*image.cols*image.channels();
            for (j = 0; j < ScreenWidth; j++)
            {
                if (trans_color != -1 && GifRow[j] == trans_color)
                {
                    BufferP += 3;
                    continue;
                }
                
                ColorMapEntry = &ColorMap->Colors[GifRow[j]];
                *(BufferP + 0) = ColorMapEntry->Blue;
                *(BufferP + 1) = ColorMapEntry->Green;
                *(BufferP + 2) = ColorMapEntry->Red;

                BufferP += 3;
            }
        }
        //fclose(rgbfp[0]);
    } else {
        unsigned char *Buffers[3];
        
        if ((Buffers[0] = (unsigned char *) malloc(ScreenWidth)) == NULL ||
            (Buffers[1] = (unsigned char *) malloc(ScreenWidth)) == NULL ||
            (Buffers[2] = (unsigned char *) malloc(ScreenWidth)) == NULL)
            GIF_EXIT("Failed to allocate memory required, aborted.");
        
        for (i = 0; i < ScreenHeight; i++) {
            GifRow = ScreenBuffer[i];
            //GifQprintf("\b\b\b\b%-4d", ScreenHeight - i);
            for (j = 0; j < ScreenWidth; j++) {
                ColorMapEntry = &ColorMap->Colors[GifRow[j]];
                Buffers[0][j] = ColorMapEntry->Red;
                Buffers[1][j] = ColorMapEntry->Green;
                Buffers[2][j] = ColorMapEntry->Blue;
            }
            if (fwrite(Buffers[0], ScreenWidth, 1, rgbfp[0]) != 1 ||
                fwrite(Buffers[1], ScreenWidth, 1, rgbfp[1]) != 1 ||
                fwrite(Buffers[2], ScreenWidth, 1, rgbfp[2]) != 1)
                GIF_EXIT("Write to file(s) failed.");
        }
        
        free((char *) Buffers[0]);
        free((char *) Buffers[1]);
        free((char *) Buffers[2]);
        fclose(rgbfp[0]);
        // cppcheck-suppress useClosedFile
        fclose(rgbfp[1]);
        // cppcheck-suppress useClosedFile
        fclose(rgbfp[2]);
    }
}


void GIF2RGB(std::string buffer, bool OneFileFlag, std::vector<cv::Mat> &dst_imgs)
{
    dst_imgs.clear();
    
    int	i, j, Size, Row, Col, Width, Height, ExtCode, Count, trans_color = -1;
    GifRecordType RecordType;
    GifByteType *Extension;
    GifRowType *ScreenBuffer;
    GifFileType *GifFile;
    int
    InterlacedOffset[] = { 0, 4, 2, 1 }, /* The way Interlaced image should. */
    InterlacedJumps[] = { 8, 8, 4, 2 };    /* be read - offsets and jumps... */
    int ImageNum = 0;
    ColorMapObject *ColorMap;
    
    struct gs_gif_input_src src;
    gs_gif_init_input_source(&src, buffer);
    
    GifFile = DGifOpen(&src, gs_gif_input);
    if (GifFile == NULL)
    {
        DGifCloseFile(GifFile);
        fprintf(stderr, "Failed to DGifSlurp, %d\n", GifLastError());
        return;
    }
    
    
    if (GifFile->SHeight == 0 || GifFile->SWidth == 0)
    {
        DGifCloseFile(GifFile);
        fprintf(stderr, "Image of width or height 0\n");
        return;
    }
    
    /*
     * Allocate the screen as vector of column of rows. Note this
     * screen is device independent - it's the screen defined by the
     * GIF file parameters.
     */
    if ((ScreenBuffer = (GifRowType *) malloc(GifFile->SHeight * sizeof(GifRowType))) == NULL)
    {
        DGifCloseFile(GifFile);
        fprintf(stderr, "Failed to allocate memory required, aborted.\n");
        return;
    }
    
    Size = GifFile->SWidth * sizeof(GifPixelType);/* Size in bytes one row.*/
    if ((ScreenBuffer[0] = (GifRowType) malloc(Size)) == NULL) /* First row. */
    {
        (void)free(ScreenBuffer);
        DGifCloseFile(GifFile);
        fprintf(stderr, "Failed to allocate memory required, aborted.\n");
        return;
    }
    
    for (i = 0; i < GifFile->SWidth; i++)  /* Set its color to BackGround. */
    {
        ScreenBuffer[0][i] = GifFile->SBackGroundColor;
    }
    for (i = 1; i < GifFile->SHeight; i++) {
        /* Allocate the other rows, and set their color to background too: */
        if ((ScreenBuffer[i] = (GifRowType) malloc(Size)) == NULL)
        {
            (void)free(ScreenBuffer);
            DGifCloseFile(GifFile);
            fprintf(stderr, "Failed to allocate memory required, aborted.\n");
            return;
        }
        
        memcpy(ScreenBuffer[i], ScreenBuffer[0], Size);
    }
    
    bool bret = true;
    cv::Mat image(GifFile->SHeight, GifFile->SWidth, CV_8UC3);
    /* Scan the content of the GIF file and load the image(s) in: */
    do {
        if (DGifGetRecordType(GifFile, &RecordType) == GIF_ERROR)
        {
            //PrintGifError(GifLastError());
            bret = false;
            fprintf(stderr, "Failed to DGifGetRecordType, %d\n", GifLastError());
            continue;
        }
        switch (RecordType) {
            case IMAGE_DESC_RECORD_TYPE:
                if (DGifGetImageDesc(GifFile) == GIF_ERROR) {
                    //PrintGifError(GifLastError());
                    bret = false;
                    fprintf(stderr, "Failed to DGifGetImageDesc, %d\n", GifLastError());
                    continue;
                }
                Row = GifFile->Image.Top; /* Image Position relative to Screen. */
                Col = GifFile->Image.Left;
                Width = GifFile->Image.Width;
                Height = GifFile->Image.Height;
                //GifQprintf("\nError: Image %d at (%d, %d) [%dx%d]:     ",
                //           ImageNum, Col, Row, Width, Height);
                if (GifFile->Image.Left + GifFile->Image.Width > GifFile->SWidth ||
                    GifFile->Image.Top + GifFile->Image.Height > GifFile->SHeight) {
                    bret = false;
                    fprintf(stderr, "Image %d is not confined to screen dimension, aborted.\n",ImageNum);
                    continue;
                }
                if (GifFile->Image.Interlace) {
                    /* Need to perform 4 passes on the images: */
                    for (Count = i = 0; i < 4; i++)
                        for (j = Row + InterlacedOffset[i]; j < Row + Height;
                             j += InterlacedJumps[i]) {
                            //GifQprintf("\b\b\b\b%-4d", Count++);
                            if (DGifGetLine(GifFile, &ScreenBuffer[j][Col],
                                            Width) == GIF_ERROR) {
                                //PrintGifError(GifLastError());
                                bret = false;
                                fprintf(stderr, "Failed to DGifGetLine, %d\n", GifLastError());
                                continue;
                            }
                        }
                }
                else {
                    for (i = 0; i < Height; i++) {
                        //GifQprintf("\b\b\b\b%-4d", i);
                        if (DGifGetLine(GifFile, &ScreenBuffer[Row++][Col],
                                        Width) == GIF_ERROR) {
                            bret = false;
                            fprintf(stderr, "Failed to DGifGetLine, %d\n", GifLastError());
                            continue;
                        }
                    }
                }
                
                if (bret)
                {
                    ImageNum++;
                    ColorMap = (GifFile->Image.ColorMap
                                ? GifFile->Image.ColorMap
                                : GifFile->SColorMap);
                    if (ColorMap == NULL)
                    {
                        bret = false;
                        fprintf(stderr, "Gif Image does not have a colormap\n");
                        continue;
                    }
                    
                    DumpScreen2RGB(image, trans_color, OneFileFlag,
                                   ColorMap,
                                   ScreenBuffer,
                                   GifFile->SWidth, GifFile->SHeight);
                    
                    dst_imgs.push_back(image.clone());
                    
                    //cv::imshow("test", image);
                    //cv::waitKey(40);
                }
                break;
            case EXTENSION_RECORD_TYPE:
                /* Skip any extension blocks in file: */
                if (DGifGetExtension(GifFile, &ExtCode, &Extension) == GIF_ERROR)
                {
                    bret = false;
                    fprintf(stderr, "Failed to DGifGetExtension, %d\n", GifLastError());
                    continue;
                }
                
                if (ExtCode == 0xF9 && (Extension[1] & 1) == 1)
                {
                    trans_color = Extension[4];
                }
                
                while (Extension != NULL)
                {
                    if (DGifGetExtensionNext(GifFile, &Extension) == GIF_ERROR)
                    {
                        bret = false;
                        fprintf(stderr, "Failed to DGifGetExtensionNext, %d\n", GifLastError());
                        break;
                    }
                }
                break;
            case TERMINATE_RECORD_TYPE:
                break;
            default:		    /* Should be trapped by DGifGetRecordType. */
                break;
        }
    } while (RecordType != TERMINATE_RECORD_TYPE && bret);
    
    printf("Get %d image\n", ImageNum);
    
    image.release();
    (void)free(ScreenBuffer);
    
    if (DGifCloseFile(GifFile) == GIF_ERROR)
    {
        fprintf(stderr, "Failed to DGifCloseFile, %d ", GifLastError());
    }
}
