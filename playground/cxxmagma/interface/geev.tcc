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

#ifndef PLAYGROUND_CXXMAGMA_INTERFACE_GEEV_TCC
#define PLAYGROUND_CXXMAGMA_INTERFACE_GEEV_TCC 1

#include <iostream>
#include <playground/cxxmagma/interface/interface.h>
#include <cxxlapack/netlib/netlib.h>

namespace cxxmagma {

template <typename IndexType>
IndexType
geev(char           jobVL,
     char           jobVR,
     IndexType      n,
     float          *A,
     IndexType      ldA,
     float          *wr,
     float          *wi,
     float          *VL,
     IndexType      ldVL,
     float          *VR,
     IndexType      ldVR,
     float          *work,
     IndexType      lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_dgeev");
    magma_dgeev(jobVL,
                jobVR,
                n,
                A,
                ldA,
                wr,
                wi,
                VL,
                ldVL,
                VR,
                ldVR,
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
geev(char           jobVL,
     char           jobVR,
     IndexType      n,
     double         *A,
     IndexType      ldA,
     double         *wr,
     double         *wi,
     double         *VL,
     IndexType      ldVL,
     double         *VR,
     IndexType      ldVR,
     double         *work,
     IndexType      lWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_dgeev");
    magma_dgeev(jobVL,
                jobVR,
                n,
                A,
                ldA,
                wr,
                wi,
                VL,
                ldVL,
                VR,
                ldVR,
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
geev(char                   jobVL,
     char                   jobVR,
     IndexType              n,
     std::complex<float >   *A,
     IndexType              ldA,
     std::complex<float >   *w,
     std::complex<float >   *VL,
     IndexType              ldVL,
     std::complex<float >   *VR,
     IndexType              ldVR,
     std::complex<float >   *work,
     IndexType              lWork,
     float                  *rWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_cgeev");
    magma_cgeev(jobVL,
                jobVR,
                n,
                A,
                ldA,
                w,
                VL,
                ldVL,
                VR,
                ldVR,
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
geev(char                   jobVL,
     char                   jobVR,
     IndexType              n,
     std::complex<double>   *A,
     IndexType              ldA,
     std::complex<double>   *w,
     std::complex<double>   *VL,
     IndexType              ldVL,
     std::complex<double>   *VR,
     IndexType              ldVR,
     std::complex<double>   *work,
     IndexType              lWork,
     double                 *rWork)
{
    IndexType info;
    CXXMAGMA_DEBUG_OUT("magma_zgeev");
    magma_zgeev(jobVL,
                jobVR,
                n,
                A,
                ldA,
                w,
                VL,
                ldVL,
                VR,
                ldVR,
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

} // namespace cxxlapack

#endif // PLAYGROUND_CXXMAGMA_INTERFACE_GEEV_TCC
