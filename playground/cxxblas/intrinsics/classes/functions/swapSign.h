/*
 *   Copyright (c) 2012, Klaus Pototzky
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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_FUNCTIONS_SWAPSIGN_H
#define PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_FUNCTIONS_SWAPSIGN_H 1

#include <playground/cxxblas/intrinsics/includes.h>

#ifdef HAVE_SSE

//--- Change all sign
Intrinsics<float, IntrinsicsLevel::SSE>
inline _swap_sign(const Intrinsics<float, IntrinsicsLevel::SSE> &x);

Intrinsics<double, IntrinsicsLevel::SSE>
inline _swap_sign(const Intrinsics<double, IntrinsicsLevel::SSE> &x);

Intrinsics<std::complex<float>, IntrinsicsLevel::SSE>
inline _swap_sign(const Intrinsics<std::complex<float>, IntrinsicsLevel::SSE> &x);

Intrinsics<std::complex<double>, IntrinsicsLevel::SSE>
inline _swap_sign(const Intrinsics<std::complex<double>, IntrinsicsLevel::SSE> &x);

//--- Change all sign of even entries [0, 2, 4, 6]

Intrinsics<float, IntrinsicsLevel::SSE>
inline _swap_sign_even(const Intrinsics<float, IntrinsicsLevel::SSE> &x);

Intrinsics<double, IntrinsicsLevel::SSE>
inline _swap_sign_even(const Intrinsics<double, IntrinsicsLevel::SSE> &x);

//--- Change all sign of odd entries [1, 3, 5, 7]

Intrinsics<float, IntrinsicsLevel::SSE>
inline _swap_sign_odd(const Intrinsics<float, IntrinsicsLevel::SSE> &x);

Intrinsics<double, IntrinsicsLevel::SSE>
inline _swap_sign_odd(const Intrinsics<double, IntrinsicsLevel::SSE> &x);
#endif // HAVE_SSE



#ifdef HAVE_AVX

//--- Add
Intrinsics<float, IntrinsicsLevel::AVX>
inline _swap_sign(const Intrinsics<float, IntrinsicsLevel::AVX> &x);

Intrinsics<double, IntrinsicsLevel::AVX>
inline _swap_sign(const Intrinsics<double, IntrinsicsLevel::AVX> &x);

Intrinsics<std::complex<float>, IntrinsicsLevel::AVX>
inline _swap_sign(const Intrinsics<std::complex<float>, IntrinsicsLevel::AVX> &x);

Intrinsics<std::complex<double>, IntrinsicsLevel::AVX>
inline _swap_sign(const Intrinsics<std::complex<double>, IntrinsicsLevel::AVX> &x);

//--- Change all sign of even entries [0, 2, 4, 6]

Intrinsics<float, IntrinsicsLevel::AVX>
inline _swap_sign_even(const Intrinsics<float, IntrinsicsLevel::AVX> &x);

Intrinsics<double, IntrinsicsLevel::AVX>
inline _swap_sign_even(const Intrinsics<double, IntrinsicsLevel::AVX> &x);

//--- Change all sign of odd entries [1, 3, 5, 7]

Intrinsics<float, IntrinsicsLevel::AVX>
inline _swap_sign_odd(const Intrinsics<float, IntrinsicsLevel::AVX> &x);

Intrinsics<double, IntrinsicsLevel::AVX>
inline _swap_sign_odd(const Intrinsics<double, IntrinsicsLevel::AVX> &x);

#endif // HAVE_AVX

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_FUNCTIONS_SWAPSIGN_H
