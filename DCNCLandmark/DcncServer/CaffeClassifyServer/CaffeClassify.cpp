//
//  CaffeClassify.cpp
//  CaffeClassify
//
//  Created by MrWang on 15/10/15.
//  Copyright (c) 2015年 WangFei. All rights reserved.
//

#include <fstream>

#include "CaffeClassify.h"

//#define DES_DECODE

#ifdef DES_DECODE
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/coded_stream.h"
#include "cryptopp.h"
#endif
Classifier::Classifier()
{
}

Classifier::Classifier(int gpu_id, const string& data_path)
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    printf("Check GPU ID: %d\n", gpu_id);
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
    printf("Set GPU ID: %d\n", gpu_id);
#endif
    
    float means[3] = {127.5, 127.5, 127.5};
    string layers[3] = {"this_is_test", "this_is_test", "this_is_test"};
    string model_file, trained_file, mean_file, lable_file;
    
    getConfigures(data_path, model_file, trained_file, mean_file, means, lable_file, layers);
    std::cout << model_file << std::endl;
    std::cout << trained_file << std::endl;
    /* Load the network. */
#ifdef DES_DECODE
    NetParameter param;
    FILE *f = NULL;
    long f_size;
    char *buffer = NULL, *buff_en = NULL;
    
    param.mutable_state()->set_phase(TEST);
    
    f = fopen(model_file.data(), "rb");
    CHECK(f != NULL) << "Read prototxt file Error.";
    fseek(f, 0, SEEK_END);
    f_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = new char[f_size];
    fread(buffer, 1, f_size, f);
    fclose(f);
    f = NULL;
    
    //buff_en = new char[f_size];
    //desDecryption((uchar *)buffer, (uchar *)buff_en, f_size);
    //memcpy(buffer, buff_en, f_size);
    
    //delete [] buff_en;
    
    buffer[f_size - 1] = '\0';
    google::protobuf::TextFormat::ParseFromString(buffer, &param);
    UpgradeNetAsNeeded("proto file", &param);
    delete [] buffer;
    net_.reset(new Net<float>(param));
    
    f = fopen(trained_file.data(), "rb");
    CHECK(f != NULL) << "Read modle file Error.";
    fseek(f, 0, SEEK_END);
    f_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = new char[f_size];
    fread(buffer, 1, f_size, f);
    fclose(f);
    f = NULL;
    
    buff_en = new char[f_size];
    desDecryption((uchar *)buffer, (uchar *)buff_en, f_size);
    memcpy(buffer, buff_en, f_size);
    delete [] buff_en;
    
    buffer[f_size - 1] = '\0';
    google::protobuf::io::CodedInputStream* coded = new google::protobuf::io::CodedInputStream((uchar*)buffer,
                                                                                               (int)f_size);
    coded->SetTotalBytesLimit(INT_MAX, 536870912);
    param.ParseFromCodedStream(coded);
    UpgradeNetAsNeeded("coded", &param);
    delete [] buffer;
    net_->CopyTrainedLayersFrom(param);
    delete coded;
#else
    #if DOUBLE
    net_.reset(new Net<double>(model_file, TEST));
    #elif UINT32
    net_.reset(new Net<unsigned int>(model_file, TEST));
    #else
    net_.reset(new Net<float>(model_file, TEST));
    #endif
    net_->CopyTrainedLayersFrom(trained_file);
#endif
    
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    
#if DOUBLE
    Blob<double>* input_layer = net_->input_blobs()[0];
#elif UINT32
    Blob<unsigned int>* input_layer = net_->input_blobs()[0];
#else
    Blob<float>* input_layer = net_->input_blobs()[0];
#endif
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    
    // 设置均值
    SetMean(mean_file, means);
    SetLabels(lable_file);
    SetFeaLayer(layers);
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */

//函数用于返回向量v的前N个最大值的索引，也就是返回概率最大的五种物体的标签
//如果你是二分类问题，那么这个N直接选择1
static std::vector<int> Argmax(const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
    
    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    
    return result;
}

/* Return the top N predictions. */
//预测函数，输入一张图片img，希望预测的前N种概率最大的，我们一般取N等于1
//输入预测结果为std::make_pair，每个对包含这个物体的名字，及其相对于的概率
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, std::vector<float> &features, int N)
{
    std::vector<float> output;
    
    Predict(img, features, output);
    
    N = std::min<int>((int)labels_.size(), N);
    
    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i)
    {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }
    
    return predictions;
}

void Classifier::getConfigures(const string data_path, string &model_file, string &trained_file,
                               string &mean_file, float means[3], string &lable_file, string layers[3])
{
    char tmp[256];
    float scale = 1.0;
    string str, config_file = data_path + "/config.dat";
    
    fstream file(config_file.data());
    
    if (file.good())
    {
        while (getline(file, str))
        {
            if (str.find("deploy") == 0)
            {
                sscanf(str.data(), "deploy=%s", tmp);
                model_file = data_path + "/" + tmp;
            }
            else if (str.find("model") == 0)
            {
                sscanf(str.data(), "model=%s", tmp);
                trained_file = data_path + "/" + tmp;
            }
            else if (str.find("mean") == 0)
            {
                if (str.find("mean1") == 0)
                {
                    sscanf(str.data(), "mean1=%f", &means[0]);
                }
                else if (str.find("mean2") == 0)
                {
                    sscanf(str.data(), "mean2=%f", &means[1]);
                }
                else if (str.find("mean3") == 0)
                {
                    sscanf(str.data(), "mean3=%f", &means[2]);
                }
                else
                {
                    sscanf(str.data(), "mean=%s", tmp);
                    mean_file = tmp[0] == '\0' ? "" : data_path + "/" + tmp;
                    mean_file = std::strcmp(tmp, "auto") == 0 ? tmp : mean_file;
                }
            }
            else if (str.find("layer") == 0)
            {
                if (str.find("layer1") == 0)
                {
                    sscanf(str.data(), "layer1=%s", tmp);
                    layers[0] = tmp[0] == '\0' ? "" : tmp;
                }
                else if (str.find("layer2") == 0)
                {
                    sscanf(str.data(), "layer2=%s", tmp);
                    layers[1] = tmp[0] == '\0' ? "" : tmp;
                }
                else if (str.find("layer3") == 0)
                {
                    sscanf(str.data(), "layer3=%s", tmp);
                    layers[2] = tmp[0] == '\0' ? "" : tmp;
                }
            }
            else if (str.find("scale") == 0)
            {
                sscanf(str.data(), "scale=%f", &scale);
            }
            else if (str.find("label") == 0)
            {
                sscanf(str.data(), "label=%s", tmp);
                lable_file = tmp[0] == '\0' ? "" : data_path + "/" + tmp;
            }
            tmp[0] = '\0';
        }
        
        file.close();
    }
    else
    {
        printf("File Open Error: %s\n", strerror(errno));
    }
    
    scale_ = scale < 0 ? 1.0 : scale;
}

/* Load the mean file in binaryproto format. */
//加载均值文件
void Classifier::SetMean(const string& mean_file, float means[])
{
    if (mean_file.empty())
    {
        for (int idx = 0; idx < 3; idx++)
        {
            means[idx] = means[idx] < 0 ? 127.5 : means[idx];
            means[idx] = means[idx] > 255 ? 127.5 : means[idx];
        }
        
#if DOUBLE
        mean_ = cv::Mat(input_geometry_, CV_64FC(num_channels_),
                        cv::Scalar(means[0], means[1], means[2]));
#elif UINT32
        mean_ = cv::Mat(input_geometry_, CV_32SC(num_channels_),
                        cv::Scalar(int(means[0]), int(means[1]), int(means[2])));
#else
        mean_ = cv::Mat(input_geometry_, CV_32FC(num_channels_),
                        cv::Scalar(means[0], means[1], means[2]));
#endif
        
        return;
    }
    else if (mean_file == "auto")
    {
        mean_.release();
        return;
    }
    
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    
    /* Convert from BlobProto to Blob<float> */
#if DOUBLE
    Blob<double> mean_blob;
#elif UINT32
    Blob<unsigned int> mean_blob;
#else
    Blob<float> mean_blob;
#endif
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";
    
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
#if DOUBLE
    double* data = mean_blob.mutable_cpu_data();
#elif UINT32
    unsigned int* data = mean_blob.mutable_cpu_data();
#else
    float* data = mean_blob.mutable_cpu_data();
#endif

    //把三通道的图片分开存储，三张图片按顺序保存到channels中
    for (int i = 0; i < num_channels_; ++i)
    {
        /* Extract an individual channel. */
#if DOUBLE
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_64FC1, data);
#elif UINT32
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32SC1, data);
#else
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
#endif
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }
    
    //重新合成一张图片
    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);
    
    
    //计算每个通道的均值，得到一个三维的向量channel_mean，然后把三维的向量扩展成一张新的均值图片
    //这种图片的每个通道的像素值是相等的，这张均值图片的大小将和网络的输入要求一样
    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

void Classifier::SetFeaLayer(const string layers[])
{
    for (int idx = 0; idx < MAX_FEALAYER_NUM; idx++)
    {
        feaLayer_[idx] = net_->blob_by_name(layers[idx]);
    }
}

void Classifier::SetLabels(const string& label_file)
{
    labels_.clear();
 
    FILE *file = fopen(label_file.data(), "rb");
    if (file != NULL)
    {
        std::ifstream labels(label_file.c_str());
        CHECK(labels) << "Unable to open labels file " << label_file;
        string line;
        while (std::getline(labels, line))
        {
            labels_.push_back(string(line));
        }
        
#if DOUBLE
        Blob<double>* output_layer = net_->output_blobs()[0];
#elif UINT32
        Blob<unsigned int>* output_layer = net_->output_blobs()[0];
#else
        Blob<float>* output_layer = net_->output_blobs()[0];
#endif
        CHECK_EQ(labels_.size(), output_layer->channels())
        << "Number of labels is different from the output layer dimension.";
    }
}

void Classifier::Predict(const cv::Mat& img, std::vector<float> &features, std::vector<float> &outputs)
{
    features.clear();
    outputs.clear();
    
    
#if DOUBLE
    Blob<double>* input_layer = net_->input_blobs()[0];
#elif UINT32
    Blob<unsigned int>* input_layer = net_->input_blobs()[0];
#else
    Blob<float>* input_layer = net_->input_blobs()[0];
#endif
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    
    Preprocess(img, &input_channels);

    net_->Forward();

    
#if DOUBLE
    const double *begin = NULL, *end = NULL;
#elif UINT32
    const unsigned int *begin = NULL, *end = NULL;
#else
    const float *begin = NULL, *end = NULL;
#endif
    for (int i = 0; i < MAX_FEALAYER_NUM; i++)
    {
        if (feaLayer_[i] != NULL)
        {
            begin = feaLayer_[i]->cpu_data();
            end = begin + feaLayer_[i]->channels();
            
            std::vector<float> fea_v = std::vector<float>(begin, end);
            features.insert(features.end(), fea_v.begin(), fea_v.end());
        }
    }
    
    if (labels_.size() > 0)
    {
        /* Copy the output layer to a std::vector */
#if DOUBLE
        Blob<double>* output_layer = net_->output_blobs()[0];
#elif UINT32
        Blob<unsigned int>* output_layer = net_->output_blobs()[0];
#else
        Blob<float>* output_layer = net_->output_blobs()[0];
#endif
        begin = output_layer->cpu_data();
        end = begin + output_layer->channels();
        outputs = std::vector<float>(begin, end);
    }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
/* 这个其实是为了获得net_网络的输入层数据的指针，然后后面我们直接把输入图片数据拷贝到这个指针里面*/
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
#if DOUBLE
    Blob<double>* input_layer = net_->input_blobs()[0];
#elif UINT32
    Blob<unsigned int>* input_layer = net_->input_blobs()[0];
#else
    Blob<float>* input_layer = net_->input_blobs()[0];
#endif
    
    int width = input_layer->width();
    int height = input_layer->height();
#if DOUBLE
    double* input_data = input_layer->mutable_cpu_data();
#elif UINT32
    unsigned int* input_data = input_layer->mutable_cpu_data();
#else
    float* input_data = input_layer->mutable_cpu_data();
#endif
    for (int i = 0; i < input_layer->channels(); ++i)
    {
#if DOUBLE
        cv::Mat channel(height, width, CV_64FC1, input_data);
#elif UINT32
        cv::Mat channel(height, width, CV_32SC1, input_data);
#else
        cv::Mat channel(height, width, CV_32FC1, input_data);
#endif
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

//图片预处理函数，包括图片缩放、归一化、3通道图片分开存储
//对于三通道输入CNN，经过该函数返回的是std::vector<cv::Mat>因为是三通道数据，索引用了vector
void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels)
{
    //cv::imwrite("/Users/betty/Documents/betty/examples/111111.jpg",img);
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    //cv::cvtColor(img, sample, CV_BGR2GRAY);
    //cv::resize(iImg, sample, cv::Size(300, 300));
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    //cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
    
    cv::Mat sample_resized;
    if ((sample.size() != input_geometry_) && (sample.data != NULL))
        cv::resize(sample, sample_resized, input_geometry_, 0, 0, CV_INTER_AREA);
    else
        sample_resized = sample;
    
    cv::Mat sample_float;
    if (num_channels_ == 3)
    {
#if DOUBLE
        sample_resized.convertTo(sample_float, CV_64FC3);
#elif UINT32
        sample_resized.convertTo(sample_float, CV_32SC3);
#else
        sample_resized.convertTo(sample_float, CV_32FC3);
#endif
    }
    else
    {
#if DOUBLE
        sample_resized.convertTo(sample_float, CV_64FC1);
#elif UINT32
        sample_resized.convertTo(sample_float, CV_32SC1);
#else
        sample_resized.convertTo(sample_float, CV_32FC1);
#endif
    }

    if (!mean_.empty())
    {
        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean_, sample_normalized);
        sample_normalized = scale_*sample_normalized;
        
        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        cv::split(sample_normalized, *input_channels);
    }
    else
    {
        cv::Mat sample_normalized, im_mean, im_std;
        cv::meanStdDev(sample_float, im_mean, im_std);
        cv::subtract(sample_float, im_mean, sample_normalized);
        sample_normalized /= im_std;
        
        cv::split(sample_normalized, *input_channels);
    }

#if DOUBLE
    CHECK(reinterpret_cast<double*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
#elif UINT32
    CHECK(reinterpret_cast<unsigned int*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
#else
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
#endif
}


