/*
 *   Copyright (c) 2012, Michael Lehn
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

#ifndef PLAYGROUND_CXXMAGMA_INTERFACE_POSV_H
#define PLAYGROUND_CXXMAGMA_INTERFACE_POSV_H 1

#include <complex>

namespace cxxmagma {

template <typename IndexType>
    IndexType
    posv(char       uplo,
         IndexType  n,
         IndexType  nRhs,
         float      *A,
         IndexType  ldA,
         float      *B,
         IndexType  ldB);

template <typename IndexType>
    IndexType
    posv(char       uplo,
         IndexType  n,
         IndexType  nRhs,
         double     *A,
         IndexType  ldA,
         double     *B,
         IndexType  ldB);

template <typename IndexType>
    IndexType
    posv(char                   uplo,
         IndexType              n,
         IndexType              nRhs,
         std::complex<float >   *A,
         IndexType              ldA,
         std::complex<float >   *B,
         IndexType              ldB);

template <typename IndexType>
    IndexType
    posv(char                   uplo,
         IndexType              n,
         IndexType              nRhs,
         std::complex<double>   *A,
         IndexType              ldA,
         std::complex<double>   *B,
         IndexType              ldB);
    
    
#ifdef HAVE_CUBLAS
    
template <typename IndexType>
    IndexType
    posv_gpu(char                                                              uplo,
             IndexType                                                         n,
             IndexType                                                         nRhs,
             flens::device_ptr<float, flens::StorageType::CUDA>                A,
             IndexType                                                         ldA,
             flens::device_ptr<float, flens::StorageType::CUDA>                B,
             IndexType                                                         ldB);

template <typename IndexType>
    IndexType
    posv_gpu(char                                                              uplo,
             IndexType                                                         n,
             IndexType                                                         nRhs, 
             flens::device_ptr<double, flens::StorageType::CUDA>               A,
             IndexType                                                         ldA,
             flens::device_ptr<double, flens::StorageType::CUDA>               B,
             IndexType                                                         ldB);

template <typename IndexType>
    IndexType
    posv_gpu(char                                                              uplo,
             IndexType                                                         n,
             IndexType                                                         nRhs,
             flens::device_ptr<std::complex<float >, flens::StorageType::CUDA> A,
             IndexType                                                         ldA,
             flens::device_ptr<std::complex<float>, flens::StorageType::CUDA>  B,
             IndexType                                                         ldB);

template <typename IndexType>
    IndexType
    posv_gpu(char                                                               uplo,
             IndexType                                                          n,
             IndexType                                                          nRhs,
             flens::device_ptr<std::complex<double>, flens::StorageType::CUDA>  A,
             IndexType                                                          ldA,
             flens::device_ptr<std::complex<double>, flens::StorageType::CUDA>  B,
             IndexType                                                          ldB);
    
#endif

} // namespace cxxmagma

#endif // PLAYGROUND_CXXMAGMA_INTERFACE_POSV_H
