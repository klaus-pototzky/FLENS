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


#ifndef PLAYGROUND_FLENS_MAGMA_GE_SV_TCC
#define PLAYGROUND_FLENS_MAGMA_GE_SV_TCC 1

#include <flens/blas/blas.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic lapack implementation =============================================


//== interface for native magma ===============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- (ge)sv [real and complex variant] -----------------------------------------

template <typename MA, typename VPIV, typename MB>
typename GeMatrix<MA>::IndexType
sv_impl(GeMatrix<MA> &A, DenseVector<VPIV> &piv, GeMatrix<MB> &B)
{
    typedef typename GeMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::gesv<IndexType>(A.numRows(),
                                               B.numCols(),
                                               A.data(),
                                               A.leadingDimension(),
                                               piv.data(),
                                               B.data(),
                                               B.leadingDimension());
    ASSERT(info>=0);
    return info;
}


template <typename MA, typename VPIV, typename MB>
typename GeMatrix<MA>::IndexType
sv_gpu_impl(GeMatrix<MA> &A, DenseVector<VPIV> &piv, GeMatrix<MB> &B)
{
    typedef typename GeMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::gesv_gpu<IndexType>(A.numRows(),
                                                   B.numCols(),
                                                   A.data(),
                                                   A.leadingDimension(),
                                                   piv.data(),
                                                   B.data(),
                                                   B.leadingDimension());
    ASSERT(info>=0);
    return info;
}

} // namespace external

#endif // USE_CXXMAGMA

//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- (ge)sv [real and complex variant] -----------------------------------------

template <typename MA, typename VPIV, typename MB>
    typename RestrictTo<IsHostGeMatrix<MA>::value
                     && IsHostIntegerDenseVector<VPIV>::value
                     && IsHostGeMatrix<MB>::value,
             typename RemoveRef<MA>::Type::IndexType>::Type
    sv(MA &&A, VPIV &&piv, MB &&B)
{
    LAPACK_DEBUG_OUT("(ge)sv [real/complex]");

//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename MatrixA::IndexType     IndexType;

    if (piv.length()==0) {
        piv.resize(A.numRows());
    }
    ASSERT(piv.length()==A.numRows());
//
//  Test the input parameters
//
#   ifndef NDEBUG
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT(A.numRows()==A.numCols());
    ASSERT((piv.inc()>0 && piv.firstIndex()==1)
        || (piv.inc()<0 && piv.firstIndex()==A.numRows()));
    ASSERT(B.firstRow()==1);
    ASSERT(B.firstCol()==1);
    ASSERT(B.numRows()==A.numRows());
#   endif

//
//  Resize output arguments if they are empty
//
    if (piv.length()==0) {
        piv.resize(A.numRows(), 1);
    }
    ASSERT(piv.length()==A.numRows());

//
//  Call implementation
//
    IndexType info = external::sv_impl(A, piv, B);

    return info;
}

//-- (ge)sv [real and complex variant] -----------------------------------------

template <typename MA, typename VPIV, typename MB>
    typename RestrictTo<IsDeviceGeMatrix<MA>::value
                     && IsHostIntegerDenseVector<VPIV>::value
                     && IsDeviceGeMatrix<MB>::value,
             typename RemoveRef<MA>::Type::IndexType>::Type
    sv(MA &&A, VPIV &&piv, MB &&B)
{
    LAPACK_DEBUG_OUT("(ge)sv [real/complex]");

//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename MatrixA::IndexType     IndexType;

    if (piv.length()==0) {
        piv.resize(A.numRows());
    }
    ASSERT(piv.length()==A.numRows());
//
//  Test the input parameters
//
#   ifndef NDEBUG
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT(A.numRows()==A.numCols());
    ASSERT((piv.inc()>0 && piv.firstIndex()==1)
        || (piv.inc()<0 && piv.firstIndex()==A.numRows()));
    ASSERT(B.firstRow()==1);
    ASSERT(B.firstCol()==1);
    ASSERT(B.numRows()==A.numRows());
#   endif

//
//  Resize output arguments if they are empty
//
    if (piv.length()==0) {
        piv.resize(A.numRows(), 1);
    }
    ASSERT(piv.length()==A.numRows());

//
//  Call implementation
//
    IndexType info = external::sv_gpu_impl(A, piv, B);

    return info;
}

//-- (ge)sv [variant if rhs is vector] -----------------------------------------

template <typename MA, typename VPIV, typename VB>
typename RestrictTo<((IsHostGeMatrix<MA>::value
                        && IsHostDenseVector<VB>::value) ||
                     (IsDeviceGeMatrix<MA>::value
                        && IsDeviceDenseVector<VB>::value))
                  && IsHostIntegerDenseVector<VPIV>::value,
          typename RemoveRef<MA>::Type::IndexType>::Type
sv(MA &&A, VPIV &&piv, VB &&b)
{
//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename RemoveRef<VB>::Type    VectorB;

    typedef typename VectorB::IndexType    IndexType;
    typedef typename MatrixA::View         MatrixAView;

    const IndexType    n     = b.length();

    MatrixAView  B(n, 1, b, n);

    return sv(A, piv, B);
}

#endif

} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_GE_SV_TCC
