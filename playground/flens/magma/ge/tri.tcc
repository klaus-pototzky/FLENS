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


#ifndef PLAYGROUND_FLENS_MAGMA_GE_TRI_TCC
#define PLAYGROUND_FLENS_MAGMA_GE_TRI_TCC 1

#include <algorithm>
#include <flens/blas/blas.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic magma implementation =============================================


//== interface for native magma ===============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- (ge)tri [real and complex variant] ----------------------------------------

template <typename MA, typename VP, typename VWORK>
typename GeMatrix<MA>::IndexType
tri_gpu_impl(GeMatrix<MA>             &A,
            const DenseVector<VP>    &piv,
            DenseVector<VWORK>       &work)
{
    typedef typename GeMatrix<MA>::ElementType  ElementType;
    typedef typename GeMatrix<MA>::IndexType    IndexType;

    IndexType info;

    if (work.length()==0) {
        IndexType nb = cxxmagma::getri_nb_query<IndexType, ElementType>(A.numRows());
        work.resize(nb*A.numRows());
    }

    info = cxxmagma::getri_gpu<IndexType>(A.numRows(),
                                          A.data(),
                                          A.leadingDimension(),
                                          piv.data(),
                                          work.data(),
                                          work.length());
    ASSERT(info>=0);
    return info;
}

} // namespace external

#endif // USE_CXXMAGMA

//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- (ge)tri [real and complex variant] ----------------------------------------

template <typename MA, typename VPIV, typename VWORK>
typename RestrictTo<IsDeviceGeMatrix<MA>::value
                 && IsHostIntegerDenseVector<VPIV>::value
                 && IsDeviceDenseVector<VWORK>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
tri(MA          &&A,
    const VPIV  &piv,
    VWORK       &&work)
{
    using std::max;

//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename MatrixA::IndexType     IndexType;


//
//  Test the input parameters
//
#   ifndef NDEBUG
    
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT(A.numRows()==A.numCols());

    const IndexType n = A.numRows();

    ASSERT(piv.firstIndex()==1);
    ASSERT(piv.length()==n);

    const bool lQuery = (work.length()==0);
    ASSERT(lQuery || work.length()>=max(IndexType(1),n));
#   endif

//
//  Call implementation
//
    const IndexType info = external::tri_gpu_impl(A, piv, work);

    return info;
}


//-- (ge)tri [real/complex variant with temporary workspace] -------------------
template <typename MA, typename VPIV>
typename RestrictTo<IsDeviceGeMatrix<MA>::value
                 && IsHostIntegerDenseVector<VPIV>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
tri(MA          &&A,
    const VPIV  &piv)
{
    typedef typename RemoveRef<MA>::Type::Vector WorkVector;

    WorkVector  work;
    return tri(A, piv, work);
}

#endif // USE_CXXMAGMA

} } // namespace lapack, flens

#endif // PLAYGROUND_FLENS_MAGMA_GE_TRI_TCC
