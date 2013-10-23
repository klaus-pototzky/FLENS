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

#ifndef PLAYGROUND_FLENS_MAGMA_GE_TRS_TCC
#define PLAYGROUND_FLENS_MAGMA_GE_TRS_TCC 1

#include <flens/blas/blas.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic magma implementation =============================================

//== interface for external magma =============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- (ge)trs [real and complex variant] ----------------------------------------

template <typename MA, typename VP, typename MB>
void
trs_gpu_impl(Transpose trans, const GeMatrix<MA> &A, const DenseVector<VP> &piv,
             GeMatrix<MB> &B)
{
    typedef typename GeMatrix<MA>::IndexType  IndexType;

    IndexType info;
    info = cxxmagma::getrs_gpu<IndexType>(lapack::getF77Char(trans),
                                          A.numRows(),
                                          B.numCols(),
                                          A.data(),
                                          A.leadingDimension(),
                                          piv.data(),
                                          B.data(),
                                          B.leadingDimension());
    ASSERT(info==0);
}

} // namespace external

#endif // USE_CXXMAGMA


//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- (ge)trs [real and complex variant] ----------------------------------------

template <typename MA, typename VPIV, typename MB>
typename RestrictTo<IsDeviceGeMatrix<MA>::value
                 && IsHostIntegerDenseVector<VPIV>::value
                 && IsDeviceGeMatrix<MB>::value,
         void>::Type
trs(Transpose trans, const MA &A, const VPIV &piv, MB &&B)
{
    LAPACK_DEBUG_OUT("(ge)trs [real/complex]");

//
//  Test the input parameters
//
#   ifndef NDEBUG
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT(A.numRows()==A.numCols());

    ASSERT(piv.firstIndex()==1);
    ASSERT(piv.length()==A.numRows());

    ASSERT(B.firstRow()==1);
    ASSERT(B.firstCol()==1);
    ASSERT(B.numRows()==A.numRows());
#   endif

//
//  Call implementation
//
    external::trs_gpu_impl(trans, A, piv, B);

}

//-- (ge)trs [variant if rhs is vector] ----------------------------------------

template <typename MA, typename VPIV, typename VB>
typename RestrictTo<IsDeviceGeMatrix<MA>::value
                 && IsHostIntegerDenseVector<VPIV>::value
                 && IsDeviceDenseVector<VB>::value,
         void>::Type
trs(Transpose trans, const MA &A, const VPIV &piv, VB &&b)
{
//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type     MatrixA;
    typedef typename RemoveRef<VB>::Type     VectorB;

//
//  Create matrix view from vector b and call above variant
//
    typedef typename VectorB::ElementType  ElementType;
    typedef typename VectorB::IndexType    IndexType;

    const IndexType    n     = b.length();
    const StorageOrder order = MatrixA::Engine::order;

    GeMatrix<DeviceFullStorageView<ElementType, order> >  B(n, 1, b, n);

    trs(trans, A, piv, B);
}

#endif // USE_CXXMAGMA

} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_GE_TRS_TCC
