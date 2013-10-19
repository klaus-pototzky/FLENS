/*
 *   Copyright (c) 2013, Klaus Pototzky
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PLAYGROUND_FLENS_AUXILIARY_OPENCLENV_TCC
#define PLAYGROUND_FLENS_AUXILIARY_OPENCLENV_TCC

#include <playground/flens/auxiliary/OpenCLEnv.h>
#include <sstream>

namespace flens {

#ifdef HAVE_OPENCL

#ifdef MAIN_FILE 

void
OpenCLEnv::init(){
    if(NCalls==0) {
        
        // Setup OpenCL environment.
        cl_int err;
        err = clGetPlatformIDs(1,
                               &platform,
                               NULL);
        checkStatus(err);
        
        err = clGetDeviceIDs(platform,
#                        ifndef OPENCL_ON_CPU
                             CL_DEVICE_TYPE_GPU, // <- select GPU
#                        else
                             CL_DEVICE_TYPE_CPU,   // <- select CPU
#                        endif
                             1,
                             &device,
                             NULL);
        checkStatus(err);
        
        props[1] = (cl_context_properties)platform;
        ctx = clCreateContext(props,
                              1,
                              &device,
                              NULL,
                              NULL,
                              &err);
        checkStatus(err);
        

        // Create stream with index 0
        queueID   = 0;
        queues.insert(std::make_pair(queueID, cl_command_queue()));
        queues.at(0) = clCreateCommandQueue(ctx,
                                            device,
                                            0,      // in-order execution
                                            &err);
        checkStatus(err);
        
#ifdef  HAVE_CLBLAS
        err = CLBLAS_IMPL(Setup)();
        checkStatus(err);
#endif
#ifdef  HAVE_CLFFT
        CLFFT_IMPL(SetupData) fftSetupData;  
        err = CLFFT_IMPL(Setup)(&fftSetupData);
        checkStatus(err);
#endif
    }
    NCalls++;
}

cl_context &
OpenCLEnv::getContext() {
    if(NCalls==0) {
        std::cerr << "Error: OpenCL not initialized!" << std::endl;
        ASSERT(0);
    }
    return ctx;
}

cl_command_queue &
OpenCLEnv::getQueue() {
    if(NCalls==0) {
        std::cerr << "Error: OpenCL not initialized!" << std::endl;
        ASSERT(0);
    }
    return queues.at(queueID);
}

cl_command_queue *
OpenCLEnv::getQueuePtr() {
    if(NCalls==0) {
        std::cerr << "Error: OpenCL not initialized!" << std::endl;
        ASSERT(0);
    }
    return &(queues.at(queueID));
}

cl_event *
OpenCLEnv::getEventPtr() {

    // Check is unused_event points to an event
    if (unused_event!=0) {
         // Release it
        cl_int status = clReleaseEvent(unused_event);
        checkStatus(status);
    }
    return &unused_event;
}

void OpenCLEnv::release(){
    if(NCalls==0) {
        std::cerr << "Error: OpenCL not initialized!" << std::endl;
        ASSERT(0);
    }
    if(NCalls==1) {
        cl_int status;
#ifdef  HAVE_CLBLAS
        CLBLAS_IMPL(Teardown)();
#endif
#ifdef  HAVE_CLFFT
        CLFFT_IMPL(Teardown)();
#endif

        // Release events
        for (std::map<int, cl_event>::iterator it=events.begin(); it!=events.end(); ++it) {
            status = clReleaseEvent(it->second);
            checkStatus(status);
        }        
        events.clear();
        status = clReleaseEvent(unused_event);
        unused_event = 0;
        checkStatus(status);

        // Release command queues
        for (std::map<int, cl_command_queue>::iterator it=queues.begin(); it!=queues.end(); ++it) {
            status = clReleaseCommandQueue(it->second);
            checkStatus(status);
        }
        queues.clear();

        // Release context
        status = clReleaseContext(ctx);
        checkStatus(status);
    }
    NCalls--;
}


void
OpenCLEnv::destroyStream(int _queueID) 
{
    if(NCalls==0) {
        std::cerr << "Error: OpenCL not initialized!" << std::endl;
        ASSERT(0);
    }
    ASSERT(_queueID!=queueID);
    cl_int status = clReleaseCommandQueue(queues.at(_queueID));
    checkStatus(status);
    queues.erase(_queueID);
}

int
OpenCLEnv::getStreamID()
{
    return queueID;
}


void
OpenCLEnv::setStream(int _queueID)
{
    if(NCalls==0) {
        std::cerr << "Error: OpenCL not initialized!" << std::endl;
        ASSERT(0);
    }
    queueID = _queueID;
    // Create new stream if object is not found
    if (queues.find(queueID) == queues.end()) {
        cl_int status;
        queues.insert(std::make_pair(queueID, cl_command_queue()));
        queues.at(queueID) = clCreateCommandQueue(ctx,
                                                  device,
                                                  0,      // in-order execution
                                                  &status);
        checkStatus(status);
    } 
}

void
OpenCLEnv::syncStream(int _streamID)
{
    if(NCalls==0) {
        std::cerr << "Error: OpenCL not initialized!" << std::endl;
        ASSERT(0);
    }
    cl_int status = clFinish(queues.at(_streamID));
    checkStatus(status);
}


void
OpenCLEnv::enableSyncCopy()
{
    syncCopyEnabled = true; 
}

void
OpenCLEnv::disableSyncCopy()
{
    syncCopyEnabled = false; 
}

bool
OpenCLEnv::isSyncCopyEnabled()
{
    return syncCopyEnabled; 
}

void
OpenCLEnv::eventRecord(int _eventID)
{
    // Event gets created on current stream / queue
    if (events.find(_eventID) == events.end()) {
        // Insert empty event
        events.insert(std::make_pair(_eventID, cl_event()));
    } else {
        // Release previous event
        cl_int status = clReleaseEvent(events.at(_eventID));
        checkStatus(status);
    }
    events.at(_eventID) = unused_event;
    unused_event        = 0;

}

void
OpenCLEnv::eventSynchronize(int _eventID)
{
     ///
     /// cudaEventSynchronize: Host waits until -eventID is completeted
     /// 
     ///
     cl_int status = clWaitForEvents(1, &events.at(_eventID));
     checkStatus(status);
}

std::string
OpenCLEnv::getInfo()
{
    std::stringstream sstream;
    size_t valueSize;
    cl_uint platformCount;

    cl_uint deviceCount;
    cl_uint maxComputeUnits;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), NULL);

    for (cl_uint i = 0; i < platformCount; i++) {


         
        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), NULL);

        // for each device print critical attributes
        for (cl_uint j = 0; j < deviceCount; j++) {

             std::string str;
            // print device name
            clGetDeviceInfo(devices.at(j), CL_DEVICE_NAME, 0, NULL, &valueSize);
            str.resize(valueSize);
            clGetDeviceInfo(devices.at(j), CL_DEVICE_NAME, valueSize, &str.front(), NULL);
            sstream << "OpenCL Device:  " << str << std::endl;
            sstream << "===============================================================" << std::endl;
            
            // print device type
            cl_device_type device_type;
            clGetDeviceInfo(devices.at(j), CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
            if (device_type==CL_DEVICE_TYPE_CPU) {
                sstream << "Device Type:           " << std::setw(40) << "CPU" << std::endl;
            } else if (device_type==CL_DEVICE_TYPE_GPU) {
                sstream << "Device Type:           " << std::setw(40) << "GPU" << std::endl;
            } else if (device_type==CL_DEVICE_TYPE_ACCELERATOR) {
                sstream << "Device Type:           " << std::setw(40) << "Accelerator" << std::endl;
            } else if (device_type==CL_DEVICE_TYPE_DEFAULT) {
                sstream << "Device Type:           " << std::setw(40) << "Default" << std::endl;
            }
            
            bool device_availible;
            clGetDeviceInfo(devices.at(j), CL_DEVICE_AVAILABLE,
                            sizeof(device_availible), &device_availible, NULL);
            sstream << "Device availible:      " << std::setw(40) << std::boolalpha << device_availible << std::endl;

            // print hardware device version
            clGetDeviceInfo(devices.at(j), CL_DEVICE_VERSION, 0, NULL, &valueSize);
            str.resize(valueSize);
            clGetDeviceInfo(devices.at(j), CL_DEVICE_VERSION, valueSize, &str.front(), NULL);
            sstream << "Hardware version:       " << std::setw(40) << str << std::endl;

            // print software driver version
            clGetDeviceInfo(devices.at(j), CL_DRIVER_VERSION, 0, NULL, &valueSize);
            str.resize(valueSize);
            clGetDeviceInfo(devices.at(j), CL_DRIVER_VERSION, valueSize, &str.front(), NULL);
            sstream << "Software version:       " << std::setw(40) << str << std::endl;

            // print c version supported by compiler for device
            clGetDeviceInfo(devices.at(j), CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            str.resize(valueSize);
            clGetDeviceInfo(devices.at(j), CL_DEVICE_OPENCL_C_VERSION, valueSize,  &str.front(), NULL);
            sstream << "OpenCL C version:       " << std::setw(41) << str << std::endl;

            // print parallel compute units
            clGetDeviceInfo(devices.at(j), CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            sstream << "Parallel compute units: " << std::setw(39) << maxComputeUnits << std::endl;
	    
	    cl_ulong mem_size;
	    clGetDeviceInfo(devices.at(j), CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(mem_size), &mem_size, NULL);
            sstream << "Global Memory size: " << std::setw(37) << mem_size / (1024*1024) << " MByte"  << std::endl;

	    clGetDeviceInfo(devices.at(j), CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                            sizeof(mem_size), &mem_size, NULL);
            sstream << "Global Memory Cache size: " << std::setw(31) << mem_size / (1024) << " KByte"  << std::endl;
	    
	    clGetDeviceInfo(devices.at(j), CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(mem_size), &mem_size, NULL);
            sstream << "Max. Memory allocation size: " << std::setw(28) << mem_size / (1024*1024) << " MByte"  << std::endl;
	    
	    
            sstream << std::endl;
        }

    }
    return sstream.str();

}


#endif // MAIN_FILE
#endif // HAVE_OPENCL
    

} // namespace cxxblas

#endif // PLAYGROUND_FLENS_AUXILIARY_OPENCLENV_TCC
