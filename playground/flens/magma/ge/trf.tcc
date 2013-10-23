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


#ifndef PLAYGROUND_FLENS_MAGMA_GE_TRF_TCC
#define PLAYGROUND_FLENS_MAGMA_GE_TRF_TCC 1

#include <algorithm>
#include <flens/blas/blas.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic magma implementation =============================================


//== interface for native magma ===============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- (ge)trf [real and complex variant] ----------------------------------------

template <typename MA, typename VP>
typename GeMatrix<MA>::IndexType
trf_impl(GeMatrix<MA> &A, DenseVector<VP> &piv)
{
    typedef typename GeMatrix<MA>::IndexType  IndexType;

    return cxxmagma::getrf<IndexType>(A.numRows(), A.numCols(),
                                      A.data(), A.leadingDimension(),
                                      piv.data());
}


//-- (ge)trf [real and complex variant] ----------------------------------------

template <typename MA, typename VP>
typename GeMatrix<MA>::IndexType
trf_gpu_impl(GeMatrix<MA> &A, DenseVector<VP> &piv)
{
    typedef typename GeMatrix<MA>::IndexType  IndexType;

    return cxxmagma::getrf_gpu<IndexType>(A.numRows(), A.numCols(),
                                          A.data(), A.leadingDimension(),
                                          piv.data());
}

} // namespace external

#endif // USE_CXXMAGMA

//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- (ge)trf [real and complex variant] ----------------------------------------

template <typename MA, typename VPIV>
typename RestrictTo<IsHostGeMatrix<MA>::value
                 && IsHostIntegerDenseVector<VPIV>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
trf(MA &&A, VPIV &&piv)
{
    using std::min;

//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename MatrixA::IndexType     IndexType;

//
//  Test the input parameters
//
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT((piv.inc()>0 && piv.firstIndex()==1)
        || (piv.inc()<0 && piv.firstIndex()==A.numRows()));

    const IndexType mn = min(A.numRows(), A.numCols());

    if (piv.length()==0) {
        piv.resize(mn);
    }
    ASSERT(piv.length()==mn);

//
//  Call implementation
//
    IndexType info = external::trf_impl(A, piv);
    
    return info;
}

//-- (ge)trf [real and complex variant] ----------------------------------------

template <typename MA, typename VPIV>
typename RestrictTo<IsDeviceGeMatrix<MA>::value
                 && IsHostIntegerDenseVector<VPIV>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
trf(MA &&A, VPIV &&piv)
{
    using std::min;

//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename MatrixA::IndexType     IndexType;

//
//  Test the input parameters
//
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT((piv.inc()>0 && piv.firstIndex()==1)
        || (piv.inc()<0 && piv.firstIndex()==A.numRows()));

    const IndexType mn = min(A.numRows(), A.numCols());

    if (piv.length()==0) {
        piv.resize(mn);
    }
    ASSERT(piv.length()==mn);

//
//  Call implementation
//
    IndexType info = external::trf_gpu_impl(A, piv);
    
    return info;
}

#endif

} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_GE_TRF_TCC
