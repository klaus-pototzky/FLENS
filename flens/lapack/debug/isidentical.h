/*
 *   Copyright (c) 2011, Michael Lehn
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

#ifndef FLENS_LAPACK_DEBUG_ISIDENTICAL_H
#define FLENS_LAPACK_DEBUG_ISIDENTICAL_H 1

#include <flens/matrixtypes/matrixtypes.h>
#include <flens/vectortypes/vectortypes.h>

namespace flens { namespace lapack {

template <typename X, typename Y>
    bool
    isIdentical(const X &x, const Y &y,
                const char *xName = "x", const char *yName = "y");

template <typename VX, typename VY>
    bool
    isIdentical(const DenseVector<VX> &x, const DenseVector<VY> &y,
                const char *xName = "x", const char *yName = "y");

template <typename MA, typename MB>
    bool
    isIdentical(const GbMatrix<MA> &A, const GbMatrix<MB> &B,
                const char *AName = "A", const char *BName = "B");

template <typename MA, typename MB>
    bool
    isIdentical(const GeMatrix<MA> &A, const GeMatrix<MB> &B,
                const char *AName = "A", const char *BName = "B");

template <typename MA, typename MB>
    bool
    isIdentical(const HeMatrix<MA> &A, const HeMatrix<MB> &B,
                const char *AName = "A", const char *BName = "B");

template <typename MA, typename MB>
    bool
    isIdentical(const TrMatrix<MA> &A, const TrMatrix<MB> &B,
                const char *AName = "A", const char *BName = "B");

template <typename MA, typename MB>
    bool
    isIdentical(const SyMatrix<MA> &A, const SyMatrix<MB> &B,
                const char *AName = "A", const char *BName = "B");

} } // namespace lapack, flens

#endif // FLENS_LAPACK_DEBUG_ISIDENTICAL_H
