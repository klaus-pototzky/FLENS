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

#ifndef PLAYGROUND_CXXBLAS_LEVEL2_TPMV_H
#define PLAYGROUND_CXXBLAS_LEVEL2_TPMV_H 1

#include <cxxblas/typedefs.h>


namespace cxxblas {
  
#ifdef HAVE_CLBLAS

// tpmv [generic without workspace] 
template <typename IndexType, typename T>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const T, flens::StorageType::OpenCL> A, 
         flens::device_ptr<T, flens::StorageType::OpenCL> x, IndexType incX);
    
// stpmv
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const float, flens::StorageType::OpenCL> A, 
         flens::device_ptr<float, flens::StorageType::OpenCL> x, IndexType incX,
         flens::device_ptr<float, flens::StorageType::OpenCL> work);

// dtpmv
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const double, flens::StorageType::OpenCL> A, 
         flens::device_ptr<double, flens::StorageType::OpenCL> x, IndexType incX, 
         flens::device_ptr<double, flens::StorageType::OpenCL> work);

// ctpmv
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A, 
         flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
         flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> work);

// ztpmv
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A, 
         flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
         flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> work);

#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// stpmv
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const float, flens::StorageType::CUDA> A, 
         flens::device_ptr<float, flens::StorageType::CUDA> x, IndexType incX);

// dtpmv
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const double, flens::StorageType::CUDA> A, 
         flens::device_ptr<double, flens::StorageType::CUDA> x, IndexType incX);

// ctpmv
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, 
         flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> x, IndexType incX);

// ztpmv
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    tpmv(StorageOrder order, StorageUpLo upLo,
         Transpose transA, Diag diag,
         IndexType n, 
         const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, 
         flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> x, IndexType incX);

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL2_TPMV_H
