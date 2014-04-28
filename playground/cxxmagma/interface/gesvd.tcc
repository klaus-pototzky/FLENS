/*
 *   Copyright (c) 2012, Michael Lehn, Klaus Pototzky
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

#ifndef PLAYGROUND_CXXMAGMA_INTERFACE_GESVD_TCC
#define PLAYGROUND_CXXMAGMA_INTERFACE_GESVD_TCC 1

#include <iostream>
#include <cxxlapack/interface/interface.h>
#include <cxxlapack/netlib/netlib.h>
#include <playground/cxxmagma/interface/interface.h>

namespace cxxmagma {

template <typename IndexType>
IndexType
gesvd(char                  jobU,
      char                  jobVt,
      IndexType             m,
      IndexType             n,
      float                 *A,
      IndexType             ldA,
      float                 *s,
      float                 *U,
      IndexType             ldU,
      float                 *Vt,
      IndexType             ldVt,
      float                 *work,
      IndexType             lWork)
{
    CXXMAGMA_DEBUG_OUT("magma_dgesvd");

    IndexType info;
    magma_dgesvd(jobU,
                 jobVt,
                 m,
                 n,
                 A,
                 ldA,
                 s,
                 U,
                 ldU,
                 Vt,
                 ldVt,
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
gesvd(char                  jobU,
      char                  jobVt,
      IndexType             m,
      IndexType             n,
      double                *A,
      IndexType             ldA,
      double                *s,
      double                *U,
      IndexType             ldU,
      double                *Vt,
      IndexType             ldVt,
      double                *work,
      IndexType             lWork)
{
    CXXMAGMA_DEBUG_OUT("magma_dgesvd");

    IndexType info;
    magma_dgesvd(jobU,
                 jobVt,
                 m,
                 n,
                 A,
                 ldA,
                 s,
                 U,
                 ldU,
                 Vt,
                 ldVt,
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
gesvd(char                  jobU,
      char                  jobVt,
      IndexType             m,
      IndexType             n,
      std::complex<float >  *A,
      IndexType             ldA,
      float                 *s,
      std::complex<float >  *U,
      IndexType             ldU,
      std::complex<float >  *Vt,
      IndexType             ldVt,
      std::complex<float >  *work,
      IndexType             lWork,
      float                 *rWork)
{
    CXXMAGMA_DEBUG_OUT("magma_cgesvd");

    IndexType info;
    magma_cgesvd(jobU,
                 jobVt,
                 m,
                 n,
                 A,
                 ldA,
                 s,
                 U,
                 ldU,
                 Vt,
                 ldVt,
                 work,
                 lWork,
                 rWork,
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
gesvd(char                  jobU,
      char                  jobVt,
      IndexType             m,
      IndexType             n,
      std::complex<double>  *A,
      IndexType             ldA,
      double                *s,
      std::complex<double>  *U,
      IndexType             ldU,
      std::complex<double>  *Vt,
      IndexType             ldVt,
      std::complex<double>  *work,
      IndexType             lWork,
      double                *rWork)
{
    CXXMAGMA_DEBUG_OUT("magma_zgesvd");

    IndexType info;
    magma_zgesvd(jobU,
                 jobVt,
                 m,
                 n,
                 A,
                 ldA,
                 s,
                 U,
                 ldU,
                 Vt,
                 ldVt,
                 work,
                 lWork,
                 rWork,
                 &info);
#   ifndef NDEBUG
    if (info<0) {
        std::cerr << "info = " << info << std::endl;
    }
#   endif
    ASSERT(info>=0);
    return info;

}

} // namespace cxxmagma

#endif // PLAYGROUND_CXXMAGMA_INTERFACE_GESVD_TCC
