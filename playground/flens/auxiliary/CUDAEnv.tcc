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

#ifndef PLAYGROUND_FLENS_AUXILIARY_CUDAENV_TCC
#define PLAYGROUND_FLENS_AUXILIARY_CUDAENV_TCC

#include <playground/flens/auxiliary/CUDAEnv.h>
#include <utility>
#include <sstream>

namespace flens {

#ifdef HAVE_CUBLAS
#ifdef MAIN_FILE

void
CudaEnv::init(){
    if(NCalls==0) {

        // create Handle
        cublasStatus_t cublas_status = cublasCreate(&handle);
        checkStatus(cublas_status);

        // Create stream with index 0
        streamID   = 0;
        streams.insert(std::make_pair(streamID, cudaStream_t()));
        cudaError_t cuda_status = cudaStreamCreate(&streams.at(0));
        checkStatus(cuda_status);
    }
    NCalls++;
}


void
CudaEnv::release(){
    ASSERT(NCalls>0);
    if (NCalls==1) {
      
        // destroy events
        cudaError_t cuda_status = cudaSuccess;;
        for (std::map<int, cudaEvent_t >::iterator it=events.begin(); it!=events.end(); ++it) {
            cuda_status = cudaEventDestroy(it->second);
            checkStatus(cuda_status);
        }
        events.clear();
        
        // destroy stream
        for (std::map<int, cudaStream_t>::iterator it=streams.begin(); it!=streams.end(); ++it) {
            cuda_status = cudaStreamDestroy(it->second);
            checkStatus(cuda_status);
        }
        streams.clear();

        // destroy handle
        cublasStatus_t cublas_status = cublasDestroy(handle);
        checkStatus(cublas_status);
    }
    NCalls--;
}
    

void
CudaEnv::destroyStream(int _streamID) 
{
  
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }
    
    ASSERT(_streamID!=streamID);
    cudaError_t cuda_status = cudaStreamDestroy(streams.at(_streamID));
    checkStatus(cuda_status);
    streams.erase(_streamID);
}

cublasHandle_t &
CudaEnv::getHandle() {
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }
    
    return handle;
}

cudaStream_t &
CudaEnv::getStream() {
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }
    
    return streams.at(streamID);
}

int
CudaEnv::getStreamID()
{
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }
    
    return streamID;
}


void
CudaEnv::setStream(int _streamID)
{
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }
    
    streamID = _streamID;
    // Create new stream, object not found
    if (streams.find(streamID) == streams.end()) {
        streams.insert(std::make_pair(streamID, cudaStream_t()));
        cudaError_t cuda_status = cudaStreamCreate(&streams.at(streamID));
        checkStatus(cuda_status);
    } 

    // Set stream
    cublasStatus_t cublas_status = cublasSetStream(handle, streams.at(streamID));
    checkStatus(cublas_status);
    
}

void
CudaEnv::syncStream(int _streamID)
{
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }
    
    cudaError_t cuda_status = cudaStreamSynchronize(streams.at(_streamID));
    checkStatus(cuda_status);
}

void
CudaEnv::enableSyncCopy()
{
    syncCopyEnabled = true; 
}

void
CudaEnv::disableSyncCopy()
{
    syncCopyEnabled = false; 
}

bool
CudaEnv::isSyncCopyEnabled()
{
    return syncCopyEnabled; 
}


void
CudaEnv::eventRecord(int _eventID)
{
    ///
    /// Creates event on current stream
    ///
    cudaError_t cuda_status;
    // Insert new event
    if (events.find(_eventID) == events.end()) {
        events.insert(std::make_pair(_eventID, cudaEvent_t ()));
        cuda_status = cudaEventCreate(&events.at(_eventID));
        checkStatus(cuda_status);

    } 
    
    // Create Event
    cuda_status = cudaEventRecord(events.at(_eventID), streams.at(streamID));
    checkStatus(cuda_status);
}

void
CudaEnv::eventSynchronize(int _eventID)
{
     ///
     /// cudaEventSynchronize: Host waits until -eventID is completeted
     /// 
     ///
     cudaError_t cuda_status = cudaEventSynchronize(events.at(_eventID));
     checkStatus(cuda_status);
}

std::string
CudaEnv::getInfo()
{
    std::stringstream sstream;
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);

        sstream << "CUDA Device " << devProp.name << std::endl;
        sstream << "==============================================" << std::endl;
        sstream << "Major revision number:         " << std::setw(15) << devProp.major << std::endl;
        sstream << "Minor revision number:         " << std::setw(15) << devProp.minor << std::endl;
        sstream << "Number of multiprocessors:     " << std::setw(15) << devProp.multiProcessorCount << std::endl;
        sstream << "Clock rate:                    " << std::setw(11) << devProp.clockRate / (1024*1024) << " GHz" << std::endl;
        sstream << "Total global memory:           " << std::setw(9) << devProp.totalGlobalMem / (1024*1024) << " MByte" << std::endl;
        sstream << "Total constant memory:         " << std::setw(9) << devProp.totalConstMem  / (1024) << " KByte"<< std::endl;
        sstream << "Maximum memory:                " << std::setw(9) << devProp.memPitch  / (1024*1024) << " MByte"<< std::endl;
        sstream << "Total shared memory per block: " << std::setw(9) << devProp.sharedMemPerBlock / (1024) << " KByte" << std::endl;
        sstream << "Total registers per block:     " << std::setw(15) << devProp.regsPerBlock << std::endl;
        sstream << "Warp size:                     " << std::setw(15) << devProp.warpSize << std::endl;
        sstream << "Maximum threads per block:     " << std::setw(15) << devProp.maxThreadsPerBlock << std::endl;
        for (int j = 0; j < 3; ++j) {
            sstream << "Maximum dimension of block " << j << ":  " << std::setw(15) << devProp.maxThreadsDim[j] << std::endl;
        }
        for (int j = 0; j < 3; ++j) {
            sstream << "Maximum dimension of grid " << j << ":   "  << std::setw(15) << devProp.maxGridSize[j] << std::endl;
        }
        sstream << "Texture alignment:             " << std::setw(15) << devProp.textureAlignment << std::endl;
        sstream << "Concurrent copy and execution: " << std::setw(15) << std::boolalpha << (devProp.deviceOverlap>0) << std::endl;
        sstream << "Concurrent kernel execution:   " << std::setw(15) << std::boolalpha << (devProp.concurrentKernels>0) << std::endl;
        sstream << "Kernel execution timeout:      " << std::setw(15) << std::boolalpha << (devProp.kernelExecTimeoutEnabled>0) << std::endl;
        sstream << "ECC support enabled:           " << std::setw(15) << std::boolalpha << (devProp.ECCEnabled>0) << std::endl;
    }
    return sstream.str();
}

#endif
#endif //HAVE_CUBLAS
    

} // namespace cxxblas

#endif // PLAYGROUND_FLENS_AUXILIARY_CUDAENV_TCC
