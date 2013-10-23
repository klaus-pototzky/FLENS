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

#ifndef PLAYGROUND_CXXMAGMA_INTERFACE_GETRS_TCC
#define PLAYGROUND_CXXMAGMA_INTERFACE_GETRS_TCC 1

#include <iostream>
#include <playground/cxxmagma/interface/interface.h>

namespace cxxmagma {

template <typename IndexType>
IndexType
getrs_gpu(char                                                                            trans,
          IndexType                                                                       n,
          IndexType                                                                       nRhs,
          const flens::device_ptr<const float, flens::StorageType::CUDA>                  A,
          IndexType                                                                       ldA,
          const IndexType                                                                 *iPiv,
          flens::device_ptr<float, flens::StorageType::CUDA>                              B,
          IndexType                                                                       ldB)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_sgetrs_gpu");
    magma_sgetrs_gpu(trans,
                     n,
                     nRhs,
                     A.get(),
                     ldA,
                     iPiv,
                     B.get(),
                     ldB,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "trans = " << trans << std::endl;
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
getrs_gpu(char                                                                            trans,
          IndexType                                                                       n,
          IndexType                                                                       nRhs,
          const flens::device_ptr<const double, flens::StorageType::CUDA>                 A,
          IndexType                                                                       ldA,
          const IndexType                                                                 *iPiv,
          flens::device_ptr<double, flens::StorageType::CUDA>                             B,
          IndexType                                                                       ldB)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_dgetrs_gpu");
    magma_dgetrs_gpu(trans,
                     n,
                     nRhs,
                     A.get(),
                     ldA,
                     iPiv,
                     B.get(),
                     ldB,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "trans = " << trans << std::endl;
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
getrs_gpu(char                                                                            trans,
          IndexType                                                                       n,
          IndexType                                                                       nRhs,
          const flens::device_ptr<const std::complex<float>, flens::StorageType::CUDA>    A,
          IndexType                                                                       ldA,
          const IndexType                                                                 *iPiv,
          flens::device_ptr<std::complex<float>, flens::StorageType::CUDA>                B,
          IndexType                                                                       ldB)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_cgetrs_gpu");
    magma_cgetrs_gpu(trans,
                     n,
                     nRhs,
                     A.get(),
                     ldA,
                     iPiv,
                     B.get(),
                     ldB,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "trans = " << trans << std::endl;
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
getrs_gpu(char                                                                            trans,
          IndexType                                                                       n,
          IndexType                                                                       nRhs,
          const flens::device_ptr<const std::complex<double>, flens::StorageType::CUDA>   A,
          IndexType                                                                       ldA,
          const IndexType                                                                 *iPiv,
          flens::device_ptr<std::complex<double>, flens::StorageType::CUDA>               B,
          IndexType                                                                       ldB)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_zgetrs_gpu");
    magma_zgetrs_gpu(trans,
                     n,
                     nRhs,
                     A.get(),
                     ldA,
                     iPiv,
                     B.get(),
                     ldB,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "trans = " << trans << std::endl;
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

} // namespace cxxlapack

#endif // PLAYGROUND_CXXMAGMA_INTERFACE_GETRS_TCC 1
