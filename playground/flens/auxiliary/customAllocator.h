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

#ifndef PLAYGROUND_FLENS_AUXILIARY_CUSTOMALLOCATOR_H
#define PLAYGROUND_FLENS_AUXILIARY_CUSTOMALLOCATOR_H 1

#include <playground/flens/auxiliary/OpenCLEnv.h>
#include <playground/flens/auxiliary/CUDAEnv.h>
#include <playground/flens/auxiliary/devicePtr.h>
#include <playground/flens/auxiliary/checkstatus.h>

#include <new> 

#ifdef WITH_CUBLAS
#   include <cuda_runtime_api.h>
#endif

namespace flens {
  
  
///
/// Empty Class for Fake-Alloctors
/// 
///
template <typename DataType, StorageType Storage>
class CustomAllocator {

};

#ifdef HAVE_OPENCL

///
/// Fake-Alloctor for Memory using OpenCL
/// 
///

template <typename DataType>
class CustomAllocator<DataType, StorageType::OpenCL> {

public:
    //    typedefs
    typedef DataType                                          value_type;
    typedef device_ptr<DataType, StorageType::OpenCL>         pointer;
    typedef const device_ptr<DataType, StorageType::OpenCL>   const_pointer;
    typedef std::size_t                                       size_type;
    static const StorageType                                  Type = StorageType::OpenCL;

    CustomAllocator() {}
    CustomAllocator(const CustomAllocator&) {}

    pointer allocate(size_type cnt) {
 
        OpenCLEnv::init();

        cl_int status;
        cl_mem ptr = clCreateBuffer(OpenCLEnv::getContext(), 
                                    CL_MEM_READ_WRITE, 
                                    cnt * sizeof (DataType),
                                    NULL, 
                                    &status);
        if (status==CL_MEM_OBJECT_ALLOCATION_FAILURE) {
            std::bad_alloc exception;
            throw exception;
        } else {
            checkStatus(status);
        }

        // TODO: Get deviceID (in case of multiple GPUs)
        int deviceID = 0;
      
        return pointer(ptr, deviceID);
    }

    void deallocate(pointer p, size_type) {
        clReleaseMemObject(p.get());
        OpenCLEnv::release();
    }
};


#endif // HAVE_OPENCL

#ifdef WITH_CUBLAS

///
/// Fake-Alloctor for Memory on the GPU
/// 
///
template <typename DataType>
class CustomAllocator<DataType, StorageType::CUDA> {

public:
    //    typedefs
    typedef DataType                                       value_type;
    typedef device_ptr<DataType, StorageType::CUDA>        pointer;
    typedef const device_ptr<DataType, StorageType::CUDA>  const_pointer;
    typedef std::size_t                                    size_type;
    static const StorageType                               Type = StorageType::CUDA;

    CustomAllocator() {}
    CustomAllocator(const CustomAllocator&) {}

    pointer allocate(size_type cnt) {
 
        CudaEnv::init();

        value_type *ptr;
        cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&ptr), cnt * sizeof (DataType));
        if (status==cudaErrorMemoryAllocation) {
               ptr = NULL;
               std::bad_alloc exception;
               throw exception;
        } else {
              checkStatus(status);
        }
        int deviceID = 0;
        status = cudaGetDevice(&deviceID);
        checkStatus(status);

        return pointer(ptr, deviceID);
    }

    void deallocate(pointer p, size_type) {
            cudaFree(p.get());
        CudaEnv::release();
    }
};


///
/// Alloctor for pinned memory
/// Adapted from: http://devnikhilk.blogspot.de/2011/08/sample-stl-allocator.html
///

template<typename T>
class PinnedAllocator{
public :
    //    typedefs

    typedef T                 value_type;
    typedef value_type*       pointer;
    typedef const value_type* const_pointer;
    typedef value_type&       reference;
    typedef const value_type& const_reference;
    typedef std::size_t       size_type;
    typedef std::ptrdiff_t    difference_type;

public :
    //    convert an allocator<T> to allocator<U>

    template<typename U>
    struct rebind {
        typedef PinnedAllocator<U> other;
    };

public :
    inline explicit PinnedAllocator() {}

    inline          ~PinnedAllocator() {}

    inline explicit PinnedAllocator(PinnedAllocator const&) {}

    template<typename U>
    inline explicit PinnedAllocator(PinnedAllocator<U> const&) {}

    inline pointer       address(reference r)       { return &r; }
    inline const_pointer address(const_reference r) { return &r; }


    inline pointer allocate(size_type cnt,
                            typename std::allocator<void>::const_pointer = 0) {
          pointer new_memory = NULL;
          cudaError_t status = cudaHostAlloc(reinterpret_cast<void**>(&new_memory), cnt * sizeof (T), 0);
          if (status==cudaErrorMemoryAllocation) {
               new_memory = NULL;
               std::bad_alloc exception;
               throw exception;
          } else {
              checkStatus(status);
          }
          return new_memory;

    }

    inline void deallocate(pointer p, size_type n) {
        cudaError_t status = cudaFreeHost(p);
        checkStatus(status);
    }

    //    size
    inline size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    //    construction/destruction

    inline void construct(pointer p, const T& t) {
        new(p) T(t);
    }

    inline void destroy(pointer p) {
        p->~T();
    }

    inline bool operator==(PinnedAllocator const&)   { return true; };
    inline bool operator!=(PinnedAllocator const& a) { return !operator==(a); };
};


#endif // WITH_CUBLAS

} // namespace flens

#endif // PLAYGROUND_FLENS_AUXILIARY_CUSTOMALLOCATOR_H
