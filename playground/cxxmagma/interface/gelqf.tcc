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

#ifndef PLAYGROUND_CXXMAGMA_INTERFACE_GELQF_TCC
#define PLAYGROUND_CXXMAGMA_INTERFACE_GELQF_TCC 1

#include <iostream>
#include <playground/cxxmagma/interface/interface.h>
#include <cxxlapack/netlib/netlib.h>

namespace cxxmagma {
  
template <typename IndexType, typename DataType>
IndexType
gelqf_nb_query(IndexType m)
{
    if (cxxblas::IsSame<DataType, float>::value) {
        return magma_get_sgelqf_nb(m); 
    } else if (cxxblas::IsSame<DataType, double>::value) {
        return magma_get_dgelqf_nb(m); 
    } else if (cxxblas::IsSame<DataType, std::complex<float> >::value) {
        return magma_get_cgelqf_nb(m); 
    } else if (cxxblas::IsSame<DataType, std::complex<double> >::value) {
        return magma_get_zgelqf_nb(m); 
    }
    ASSERT(0);
    
    return 0;
}

template <typename IndexType>
IndexType
gelqf(IndexType   m,
      IndexType   n,
      float       *A,
      IndexType   ldA,
      float       *tau,
      float       *work,
      IndexType   lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_sgelqf");
    magma_sgelqf(m,
                 n,
                 A,
                 ldA,
                 tau,
                 work,
                 lWork,
                 &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelqf(IndexType   m,
      IndexType   n,
      double      *A,
      IndexType   ldA,
      double      *tau,
      double      *work,
      IndexType   lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_dgelqf");
    magma_dgelqf(m,
                 n,
                 A,
                 ldA,
                 tau,
                 work,
                 lWork,
                 &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelqf(IndexType             m,
      IndexType             n,
      std::complex<float >  *A,
      IndexType             ldA,
      std::complex<float >  *tau,
      std::complex<float >  *work,
      IndexType             lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_cgelqf");
    magma_cgelqf(m,
                 n,
                 A,
                 ldA,
                 tau,
                 work,
                 lWork,
                 &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelqf(IndexType             m,
      IndexType             n,
      std::complex<double>  *A,
      IndexType             ldA,
      std::complex<double>  *tau,
      std::complex<double>  *work,
      IndexType             lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_zgelqf");
    magma_zgelqf(m,
                 n,
                 A,
                 ldA,
                 tau,
                 work,
                 lWork,
                 &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

#ifdef HAVE_CUBLAS

template <typename IndexType>
IndexType
gelqf_gpu(IndexType                                                         m,
          IndexType                                                         n,
          flens::device_ptr<float, flens::StorageType::CUDA>                A,
          IndexType                                                         ldA,
          float                                                             *tau,
          float                                                             *work,
          IndexType                                                         lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_sgelqf_gpu");
    magma_sgelqf_gpu(m,
                     n,
                     A.get(),
                     ldA,
                     tau,
                     work,
                     lWork,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelqf_gpu(IndexType                                                         m,
          IndexType                                                         n,
          flens::device_ptr<double, flens::StorageType::CUDA>               A,
          IndexType                                                         ldA,
          double                                                            *tau,
          double                                                            *work,
          IndexType                                                         lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_dgelqf_gpu");
    magma_dgelqf_gpu(m,
                     n,
                     A.get(),
                     ldA,
                     tau,
                     work,
                     lWork,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelqf_gpu(IndexType                                                         m,
          IndexType                                                         n,
          flens::device_ptr<std::complex<float>, flens::StorageType::CUDA>  A,
          IndexType                                                         ldA,
          std::complex<float>                                               *tau,
          std::complex<float>                                               *work,
          IndexType                                                         lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_cgelqf_gpu");
    magma_cgelqf_gpu(m,
                     n,
                     A.get(),
                     ldA,
                     tau,
                     work,
                     lWork,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

template <typename IndexType>
IndexType
gelqf_gpu(IndexType                                                         m,
          IndexType                                                         n,
          flens::device_ptr<std::complex<double>, flens::StorageType::CUDA> A,
          IndexType                                                         ldA,
          std::complex<double>                                              *tau,
          std::complex<double>                                              *work,
          IndexType                                                         lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_zgelqf_gpu");
    magma_zgelqf_gpu(m,
                     n,
                     A.get(),
                     ldA,
                     tau,
                     work,
                     lWork,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

#endif

} // namespace cxxmagma

#endif // PLAYGROUND_CXXMAGMA_INTERFACE_GELQF_TCC
