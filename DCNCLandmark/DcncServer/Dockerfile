#CaffeClassifyServer
# version 01 多线程

# base image
FROM ubuntu:14.04

MAINTAINER wangfei "fei.wang@kankanapp.com.cn"

#ENV DEBIAN_FRONTEND noninteractive

# update source
#COPY ./files/sources.list /etc/apt/sources.list
RUN apt-get update
#RUN apt-get install software-properties-common -y
#RUN add-apt-repository ppa:fkrull/deadsnakes
#RUN apt-get update

#ENV http_proxy http://10.1.64.1:3128

# copy libs
COPY ./files/usr/local/lib /usr/local/lib
COPY ./files/usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY ./files/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
ADD ./files/app/caffe/install/lib /app/caffe/install/lib
COPY ./files/usr/lib /usr/lib

# add bin
ADD ./files/app/CaffeClassifyServer/bin/ /app/CaffeClassifyServer/bin

# run
RUN echo "/usr/lib/x86_64-linux-gnu/mesa">> /etc/ld.so.conf.d/x86_64-linux-gnu.conf
RUN ldconfig

