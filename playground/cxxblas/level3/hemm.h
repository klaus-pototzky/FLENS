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

#ifndef PLAYGROUND_CXXBLAS_LEVEL3_HEMM_H
#define PLAYGROUND_CXXBLAS_LEVEL3_HEMM_H 1

#include <cxxblas/typedefs.h>

namespace cxxblas {
  
#ifdef HAVE_CLBLAS

// chemm
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    hemm(StorageOrder order, Side side, StorageUpLo upLo,
         IndexType m, IndexType n,
         const ComplexFloat &alpha,
         const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA,
         const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> B, IndexType ldB,
         const ComplexFloat &beta,
         flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> C, IndexType ldC);

// zhemm
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    hemm(StorageOrder order, Side side, StorageUpLo upLo,
         IndexType m, IndexType n,
         const ComplexDouble &alpha,
         const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA,
         const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> B, IndexType ldB,
         const ComplexDouble &beta,
         flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> C, IndexType ldC);

#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// chemm
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    hemm(StorageOrder order, Side side, StorageUpLo upLo,
         IndexType m, IndexType n,
         const ComplexFloat &alpha,
         const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA,
         const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> B, IndexType ldB,
         const ComplexFloat &beta,
	 flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> C, IndexType ldC);
	 

// zhemm
template <typename IndexType>
    typename If<IndexType>::isBlasCompatibleInteger
    hemm(StorageOrder order, Side side, StorageUpLo upLo,
         IndexType m, IndexType n,
         const ComplexDouble &alpha,
         const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA,
         const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> B, IndexType ldB,
         const ComplexDouble &beta,
	 flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> C, IndexType ldC);

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL3_HEMM_H
