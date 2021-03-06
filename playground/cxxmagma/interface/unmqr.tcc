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

#ifndef PLAYGROUND_CXXMAGMA_INTERFACE_UNMQR_TCC
#define PLAYGROUND_CXXMAGMA_INTERFACE_UNMQR_TCC 1

#include <iostream>
#include <playground/cxxmagma/interface/interface.h>
#include <cxxlapack/netlib/netlib.h>

namespace cxxmagma {

template <typename IndexType>
IndexType
unmqr(char                        side,
      char                        trans,
      IndexType                   m,
      IndexType                   n,
      IndexType                   k,
      std::complex<float>         *A,
      IndexType                   ldA,
      const std::complex<float>   *tau,
      std::complex<float>         *C,
      IndexType                   ldC,
      std::complex<float>         *work,
      IndexType                   lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_cunmqr");
    magma_cunmqr(side,
                 trans,
                 m,
                 n,
                 k,
                 A,
                 ldA,
                 tau,
                 C,
                 ldC,
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
unmqr(char                        side,
      char                        trans,
      IndexType                   m,
      IndexType                   n,
      IndexType                   k,
      std::complex<double>        *A,
      IndexType                   ldA,
      const std::complex<double>  *tau,
      std::complex<double>        *C,
      IndexType                   ldC,
      std::complex<double>        *work,
      IndexType                   lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_zunmqr");
    magma_zunmqr(side,
                 trans,
                 m,
                 n,
                 k,
                 A,
                 ldA,
                 tau,
                 C,
                 ldC,
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

#ifdef HAVE_CUBLAS

template <typename IndexType>
IndexType
unmqr_gpu(char                                                                           side,
          char                                                                           trans,
          IndexType                                                                      m,
          IndexType                                                                      n,
          IndexType                                                                      k,
          flens::device_ptr<std::complex<float>, flens::StorageType::CUDA>               A,
          IndexType                                                                      ldA,
          const std::complex<float>                                                      *tau,
          flens::device_ptr<std::complex<float>, flens::StorageType::CUDA>               C,
          IndexType                                                                      ldC,
          std::complex<float>                                                            *work,
          IndexType                                                                      lWork,
          const flens::device_ptr<const std::complex<float>, flens::StorageType::CUDA>   t,
          IndexType                                                                      nb)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_cunmqr");
    magma_cunmqr_gpu(side,
                     trans,
                     m,
                     n,
                     k,
                     A.get(),
                     ldA,
                     tau,
                     C.get(),
                     ldC,
                     work,
                     lWork,
                     t.get(),
                     nb,
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
unmqr_gpu(char                                                                           side,
          char                                                                           trans,
          IndexType                                                                      m,
          IndexType                                                                      n,
          IndexType                                                                      k,
          flens::device_ptr<std::complex<double>, flens::StorageType::CUDA>              A,
          IndexType                                                                      ldA,
          const std::complex<double>                                                     *tau,
          flens::device_ptr<std::complex<double>, flens::StorageType::CUDA>              C,
          IndexType                                                                      ldC,
          std::complex<double>                                                           *work,
          IndexType                                                                      lWork,
          const flens::device_ptr<const std::complex<double>, flens::StorageType::CUDA>  t,
          IndexType                                                                      nb)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_zunmqr");
    magma_zunmqr_gpu(side,
                     trans,
                     m,
                     n,
                     k,
                     A.get(),
                     ldA,
                     tau,
                     C.get(),
                     ldC,
                     work,
                     lWork,
                     t.get(),
                     nb,
                     &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;
}

#endif

} // namespace cxxlapack

#endif // PLAYGROUND_CXXMAGMA_INTERFACE_UNMQR_TCC
