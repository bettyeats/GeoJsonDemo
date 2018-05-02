# cmake needs this line
cmake_minimum_required(VERSION 2.8)

set(target_name CaffeClassifyServer)

# Define project name
project(${target_name} C CXX)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

add_definitions(-std=c++11 -O3) # -DCPU_ONLY
#add_definitions(-DENV_MACOSX)

include_directories(/usr/local/cuda/include
                    /usr/local/include/caffe
                    /usr/local/include/cryptopp
                    /usr/local/include
                    /usr/include
		    src/Service)

link_directories(/usr/local/lib
                 /usr/lib)

add_executable(${target_name}
	src/main.cpp
	src/Server.cpp
	src/Server.h
	src/Service/Service.cpp
	src/Service/Service.h
	src/Service/ClassifyService_constants.cpp
	src/Service/ClassifyService_constants.h
	src/Service/ClassifyService_types.cpp
	src/Service/ClassifyService_types.h
	src/CaffeClassify.cpp
	src/CaffeClassify.h
        src/version.hpp
        src/comm.cpp
        src/comm.hpp
        src/cryptopp.cpp
        src/cryptopp.h)

# Link your application with libraries
target_link_libraries(${target_name}
	libcaffe.so
	libglog.so
        libpthread.so
	libthrift.so
	libthriftnb.so
        libevent.so
	libboost_system.so
	libopencv_core.so
	libopencv_highgui.so
	libopencv_imgproc.so
        libprotobuf.so
        libcryptopp.so)


