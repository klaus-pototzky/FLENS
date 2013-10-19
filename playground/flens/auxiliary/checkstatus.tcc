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

#ifndef PLAYGROUND_FLENS_AUXILIARY_CHECKSTATUS_TCC
#define PLAYGROUND_FLENS_AUXILIARY_CHECKSTATUS_TCC

#include <playground/flens/auxiliary/checkstatus.h>

namespace flens {

#ifdef HAVE_OPENCL

template<typename CLINT>
typename
RestrictTo<IsSame<CLINT, cl_int>::value,
           void>::Type 
checkStatus(CLINT error) 
{
    if (error==CL_SUCCESS) {
        return;
    }

    switch (error) {
        case CL_DEVICE_NOT_FOUND:
            std::cerr << "Error: Device not found.." << std::endl;
            break;
        case CL_DEVICE_NOT_AVAILABLE:
            std::cerr << "Error: Device not available." << std::endl;
            break;
        case CL_COMPILER_NOT_AVAILABLE:
            std::cerr << "Error: Compiler not available." << std::endl;
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            std::cerr << "Error: Memory object allocation failure." << std::endl;
            break;
        case CL_OUT_OF_RESOURCES:
            std::cerr << "Error: Out of resources." << std::endl;
            break;
        case CL_OUT_OF_HOST_MEMORY:
            std::cerr << "Error: Out of host memory." << std::endl;
            break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            std::cerr << "Error: Profiling information not available." << std::endl;
            break;
        case CL_MEM_COPY_OVERLAP:
            std::cerr << "Error: Memory copy overlap." << std::endl;
            break;
        case CL_IMAGE_FORMAT_MISMATCH:
            std::cerr << "Error: Image format mismatch." << std::endl;
            break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            std::cerr << "Error: Image format not supported." << std::endl;
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            std::cerr << "Error: Program build failure." << std::endl;
            break;
        case CL_MAP_FAILURE:
            std::cerr << "Error: Map failure." << std::endl;
            break;
        case CL_INVALID_VALUE:
            std::cerr << "Error: Invalid value." << std::endl;
            break;
        case CL_INVALID_DEVICE_TYPE:
            std::cerr << "Error: Invalid device type." << std::endl;
            break;
        case CL_INVALID_PLATFORM:
            std::cerr << "Error: Invalid platform." << std::endl;
            break;
        case CL_INVALID_DEVICE:
            std::cerr << "Error: Invalid device." << std::endl;
            break;
        case CL_INVALID_CONTEXT:
            std::cerr << "Error: Invalid context." << std::endl;
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            std::cerr << "Error: Invalid queue properties." << std::endl;
            break;
        case CL_INVALID_COMMAND_QUEUE:
            std::cerr << "Error: Invalid command queue." << std::endl;
            break;
        case CL_INVALID_HOST_PTR:
            std::cerr << "Error: Invalid host pointer." << std::endl;
            break;
        case CL_INVALID_MEM_OBJECT:
            std::cerr << "Error: Invalid memory object." << std::endl;
            break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            std::cerr << "Error: Invalid image format descriptor." << std::endl;
            break;
        case CL_INVALID_IMAGE_SIZE:
            std::cerr << "Error: Invalid image size." << std::endl;
            break;
        case CL_INVALID_SAMPLER:
            std::cerr << "Error: Invalid sampler." << std::endl;
            break;
        case CL_INVALID_BINARY:
            std::cerr << "Error: Invalid binary." << std::endl;
            break;
        case CL_INVALID_BUILD_OPTIONS:
            std::cerr << "Error: Invalid build options." << std::endl;
            break;
        case CL_INVALID_PROGRAM:
            std::cerr << "Error: Invalid program." << std::endl;
            break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
            std::cerr << "Error: Invalid program executable." << std::endl;
            break;
        case CL_INVALID_KERNEL_NAME:
            std::cerr << "Error: Invalid kernel name." << std::endl;
            break;
        case CL_INVALID_KERNEL_DEFINITION:
            std::cerr << "Error: Invalid kernel definition." << std::endl;
            break;
        case CL_INVALID_KERNEL:
            std::cerr << "Error: Invalid kernel." << std::endl;
            break;
        case CL_INVALID_ARG_INDEX:
            std::cerr << "Error: Invalid argument index." << std::endl;
            break;
        case CL_INVALID_ARG_VALUE:
            std::cerr << "Error: Invalid argument value." << std::endl;
            break;
        case CL_INVALID_ARG_SIZE:
            std::cerr << "Error: Invalid argument size." << std::endl;
            break;
        case CL_INVALID_KERNEL_ARGS:
            std::cerr << "Error: Invalid kernel arguments." << std::endl;
            break;
        case CL_INVALID_WORK_DIMENSION:
            std::cerr << "Error: Invalid work dimension." << std::endl;
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            std::cerr << "Error: Invalid work group size." << std::endl;
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            std::cerr << "Error: Invalid work item size." << std::endl;
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            std::cerr << "Error: Invalid global offset." << std::endl;
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            std::cerr << "Error: Invalid event wait list." << std::endl;
            break;
        case CL_INVALID_EVENT:
            std::cerr << "Error: Invalid event." << std::endl;
            break;
        case CL_INVALID_OPERATION:
            std::cerr << "Error: Invalid operation." << std::endl;
            break;
        case CL_INVALID_GL_OBJECT:
            std::cerr << "Error: Invalid OpenGL object." << std::endl;
            break;
        case CL_INVALID_BUFFER_SIZE:
            std::cerr << "Error: Invalid buffer size." << std::endl;
            break;
        case CL_INVALID_MIP_LEVEL:
            std::cerr << "Error: Invalid mip-map level." << std::endl;
            break;    
        case CLBLAS_IMPL(InvalidLeadDimA):
            std::cerr << "Error: CLBLAS_IMPL(InvalidLeadDimA)" << std::endl;
            break;
        case CLBLAS_IMPL(InvalidLeadDimB):
            std::cerr << "Error: CLBLAS_IMPL(InvalidLeadDimB)." << std::endl;
            break;
        case CLBLAS_IMPL(InvalidLeadDimC):
            std::cerr << "Error: CLBLAS_IMPL(InvalidLeadDimC)." << std::endl;
            break;
        case CLBLAS_IMPL(InvalidMatA):
            std::cerr << "Error: CLBLAS_IMPL(InvalidMatA)." << std::endl;
            break;
        case CLBLAS_IMPL(InvalidMatB):
            std::cerr << "Error: CLBLAS_IMPL(InvalidMatB)." << std::endl;
            break;
        case CLBLAS_IMPL(InvalidMatC):
            std::cerr << "Error: CLBLAS_IMPL(InvalidMatC)." << std::endl;
            break;
        case CLBLAS_IMPL(InsufficientMemMatA):
            std::cerr << "Error: CLBLAS_IMPL(InsufficientMemMatA)." << std::endl;
            break;
        case CLBLAS_IMPL(InsufficientMemMatB):
            std::cerr << "Error: CLBLAS_IMPL(InsufficientMemMatB)." << std::endl;
            break;
        case CLBLAS_IMPL(InsufficientMemMatC):
            std::cerr << "Error: CLBLAS_IMPL(InsufficientMemMatC)." << std::endl;
            break;
        default: std::cerr << "Error: Unknown." << std::endl;
    }
    ASSERT(error==CL_SUCCESS);
}


#endif // HAVE_OPENCL


#ifdef HAVE_CUBLAS

template<typename CUBLASSTATUS>
typename
RestrictTo<IsSame<CUBLASSTATUS, cublasStatus_t>::value,
           void>::Type
checkStatus(CUBLASSTATUS status)
{  
    if (status==CUBLAS_STATUS_SUCCESS) {
        return;
    }
    
    if (status==CUBLAS_STATUS_NOT_INITIALIZED) {
        std::cerr << "CUBLAS: Library was not initialized!" << std::endl;
    } else if  (status==CUBLAS_STATUS_INVALID_VALUE) {
        std::cerr << "CUBLAS: Parameter had illegal value!" << std::endl;
    } else if  (status==CUBLAS_STATUS_MAPPING_ERROR) {
        std::cerr << "CUBLAS: Error accessing GPU memory!" << std::endl;
    } else if  (status==CUBLAS_STATUS_ALLOC_FAILED) {
        std::cerr << "CUBLAS: allocation failed!" << std::endl;
    } else if  (status==CUBLAS_STATUS_ARCH_MISMATCH) {
        std::cerr << "CUBLAS: Device does not support double precision!" << std::endl;
    } else if  (status==CUBLAS_STATUS_EXECUTION_FAILED) {
        std::cerr << "CUBLAS: Failed to launch function of the GPU" << std::endl;
    } else if  (status==CUBLAS_STATUS_INTERNAL_ERROR) {
        std::cerr << "CUBLAS: An internal operation failed" << std::endl;
    } else {
        std::cerr << "CUBLAS: Unkown error" << std::endl;
    }
    
    ASSERT(status==CUBLAS_STATUS_SUCCESS);
}
    
template<typename CUDAERROR>
typename
RestrictTo<IsSame<CUDAERROR, cudaError_t>::value,
           void>::Type
checkStatus(CUDAERROR status)
{  
  
    if(status==cudaSuccess) {
        return;
    } else {
        std::cerr << cudaGetErrorString(status) << std::endl;
    }
    ASSERT(status==cudaSuccess);
}
    
#endif //HAVE_CUBLAS
    
    
#ifdef HAVE_CUFFT
    
template<typename CUFFTSTATUS>
typename
RestrictTo<IsSame<CUFFTSTATUS, cufftResult>::value,
           void>::Type
checkStatus(CUFFTSTATUS status)
{
    ASSERT(status==CUFFT_SUCCESS);
}
    
#endif

} // namespace cxxblas

#endif // PLAYGROUND_FLENS_AUXILIARY_CHECKSTATUS_TCC
