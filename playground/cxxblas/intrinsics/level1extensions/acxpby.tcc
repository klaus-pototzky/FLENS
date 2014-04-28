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

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_ACXPBY_TCC
#define PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_ACXPBY_TCC 1

#include <cxxblas/cxxblas.h>
#include <playground/cxxblas/intrinsics/auxiliary/auxiliary.h>
#include <playground/cxxblas/intrinsics/includes.h>
#include <playground/cxxblas/intrinsics/level1/axpy.h>

namespace cxxblas {

#ifdef USE_INTRINSIC

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsReal<T>::value &&
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
acxpby(IndexType n, const T &alpha, const T *x,
       IndexType incX, const T &beta, T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("acxpby_intrinsics [real, " INTRINSIC_NAME "]");

    cxxblas::axpy(n, alpha, x, incX, beta, y, incY);

}

template <typename IndexType, typename T>
typename flens::RestrictTo<flens::IsComplex<T>::value &&
                           flens::IsIntrinsicsCompatible<T>::value,
                           void>::Type
acxpby(IndexType n, const T &alpha, const T *x,
       IndexType incX, const T &beta, T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("acxpby_intrinsics [complex, " INTRINSIC_NAME "]");

    using std::real;
    using std::imag;

    typedef Intrinsics<T, DEFAULT_INTRINSIC_LEVEL>     IntrinsicType;
    typedef typename IntrinsicType::PrimitiveDataType  PT;
    typedef Intrinsics<PT, DEFAULT_INTRINSIC_LEVEL>    IntrinsicPrimitiveType;


//
//  Catch simple cases
//  1. y = beta*y
//  2. y = y + alpha*x
//  3. y = x
//
    if (alpha==T(0)) {
        cxxblas::scal(n, beta, y, incY);
        return;
    }
    if (beta==T(1)) {
        cxxblas::acxpy(n, alpha, x, incX, y, incY);
    return;
    }
    if (alpha==T(1) && beta==T(0)) {
        cxxblas::ccopy(n, x, incX, y, incY);
    return;
    }

//
//  Case y = beta*y + alpha*conj(x)
//

    if (incX==1 && incY==1) {

        const int numElements = IntrinsicType::numElements;

        IndexType i=0;

        IntrinsicType _x, _y, _tmp;
        IntrinsicPrimitiveType _real_alpha(-real(alpha));
        IntrinsicPrimitiveType _imag_alpha( imag(alpha));
        IntrinsicPrimitiveType _real_beta(  real(beta));
        IntrinsicPrimitiveType _imag_beta(  imag(beta));

        if (beta==T(0)) {
            for (; i+numElements-1<n; i+=numElements) {
                _x.loadu(x+i);
                _y.setZero();
                _y = _intrinsic_addsub(_y, _intrinsic_mul(_real_alpha, _x));
                _x = _intrinsic_swap_real_imag(_x);
                _y = _intrinsic_add(_y, _intrinsic_mul(_imag_alpha, _x));
                _y.storeu(y+i);
            }
        } else {
            for (; i+numElements-1<n; i+=numElements) {
                _x.loadu(x+i);
                _y.loadu(y+i);

                _tmp = _intrinsic_mul(_real_beta, _y);
                _y = _intrinsic_swap_real_imag(_y);
                _y = _intrinsic_mul(_imag_beta, _y);
                _y = _intrinsic_addsub(_tmp, _y);

                _y = _intrinsic_addsub(_y, _intrinsic_mul(_real_alpha, _x));
                _x = _intrinsic_swap_real_imag(_x);
                _y = _intrinsic_add(_y, _intrinsic_mul(_imag_alpha, _x));
                _y.storeu(y+i);
            }
        }

        for (; i<n; ++i) {
            y[i] += alpha*conj(x[i]);
        }

    } else {

        cxxblas::acxpby<IndexType, T, T, T ,T>(n, alpha, x, incX, beta,
                                               y, incY);

    }
}

#endif // USE_INTRINSIC

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_LEVEL1_ACXPY_TCC
