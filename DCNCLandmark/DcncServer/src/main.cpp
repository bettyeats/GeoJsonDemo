//
//  main.cpp
//  CaffeClassifyServer
//
//  Created by MrWang on 2017/1/10.
//  Copyright © 2017年 RemarkMedia. All rights reserved.
//

#include <iostream>

#include "version.hpp"
#include "Server.h"
#include <thrift/config.h>
#include <thrift/server/TNonblockingServer.h>
#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/concurrency/PosixThreadFactory.h>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;
using namespace ::apache::thrift::concurrency;

using boost::shared_ptr;

int main(int argc, const char * argv[])
{
    int port = 6001;
    int thread_num = 2;
    
    if (argc > 2)
    {
        port = atoi(argv[1]);
        thread_num = atoi(argv[2]);
        thread_num = thread_num < 1 ? 1 : thread_num;
        thread_num = thread_num > 10 ? 4 : thread_num;
    }
    
    boost::shared_ptr<ServiceHandler> handler(new ServiceHandler(thread_num));
    boost::shared_ptr<TProcessor> processor(new ServiceProcessor(handler));
    boost::shared_ptr<TProtocolFactory> protocolFactory(new TCompactProtocolFactory());
    
    boost::shared_ptr<ThreadManager> threadManager = ThreadManager::newSimpleThreadManager(thread_num);
    boost::shared_ptr<PosixThreadFactory> threadFactory = boost::shared_ptr<PosixThreadFactory > (new PosixThreadFactory());
    threadManager->threadFactory(threadFactory);
    threadManager->start();
    
    printf("\nStart Image Classify Server %s at %d, thread num is %d ...\n", g_version, port, thread_num);
    
    TNonblockingServer server(processor, protocolFactory, port, threadManager);
    server.serve();
    
    return 0;
}
