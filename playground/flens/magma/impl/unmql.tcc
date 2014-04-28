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


#ifndef PLAYGROUND_FLENS_MAGMA_IMPL_UNMQL_TCC
#define PLAYGROUND_FLENS_MAGMA_IMPL_UNMQL_TCC 1

#include <flens/blas/blas.h>
#include <flens/lapack/lapack.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic magma implementation =============================================

//== interface for native magma ===============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- unmql_wsq [worksize query, CPU] -------------------------------------------

template <typename MA, typename MC>
typename GeMatrix<MC>::IndexType
unmql_wsq_impl(Side              side,
               Transpose         trans,
               GeMatrix<MA>      &A,
               GeMatrix<MC>      &C)
{
    typedef typename GeMatrix<MC>::ElementType  T;
    typedef typename GeMatrix<MC>::IndexType    IndexType;

    T               WORK, DUMMY;
    const IndexType LWORK   = -1;

    cxxmagma::unmql<IndexType>(lapack::getF77Char(side),
                               lapack::getF77Char(trans),
                               C.numRows(),
                               C.numCols(),
                               A.numCols(),
                               A.data(),
                               A.leadingDimension(),
                               &DUMMY,
                               C.data(),
                               C.leadingDimension(),
                               &WORK,
                               LWORK);
    return cxxblas::real(WORK);
}
  

//-- unmql [CPU] --------------------------------------------------------------

template <typename MA, typename VTAU, typename MC, typename VWORK>
void
unmql_impl(Side                       side,
           Transpose                  trans,
           GeMatrix<MA>               &A,
           const DenseVector<VTAU>    &tau,
           GeMatrix<MC>               &C,
           DenseVector<VWORK>         &work)
{
    typedef typename GeMatrix<MC>::IndexType  IndexType;

    if (work.length()==0) {
        work.resize(unmql_wsq_impl(side, trans, A, C));
    }

    cxxmagma::unmql<IndexType>(lapack::getF77Char(side),
                               lapack::getF77Char(trans),
                               C.numRows(),
                               C.numCols(),
                               A.numCols(),
                               A.data(),
                               A.leadingDimension(),
                               tau.data(),
                               C.data(),
                               C.leadingDimension(),
                               work.data(),
                               work.length());
}

} // namespace external

#endif // USE_CXXMAGMA


//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- unmql [CPU variant] -------------------------------------------------------

template <typename MA, typename VTAU, typename MC, typename VWORK>
typename RestrictTo<IsHostRealGeMatrix<MA>::value
                 && IsHostRealDenseVector<VTAU>::value
                 && IsHostRealGeMatrix<MC>::value
                 && IsHostRealDenseVector<VWORK>::value,
         void>::Type
unmql(Side         side,
      Transpose    trans,
      MA           &&A,
      const VTAU   &tau,
      MC           &&C,
      VWORK        &&work)
{

//
//  Test the input parameters
//
#   ifndef NDEBUG

//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MC>::Type    MatrixC;
    typedef typename MatrixC::IndexType     IndexType;
    
    const IndexType m = C.numRows();
    const IndexType n = C.numCols();
    const IndexType k = A.numCols();

    ASSERT(tau.length()==k);

    if (side==Left) {
        ASSERT(A.numRows()==m);
    } else {
        ASSERT(A.numRows()==n);
    }

    if (work.length()>0) {
        if (side==Left) {
            ASSERT(work.length()>=n);
        } else {
            ASSERT(work.length()>=m);
        }
    }
#   endif

//
//  Call implementation
//
    external::unmql_impl(side, trans, A, tau, C, work);

}

//
//  Variant with temporary workspace
//
template <typename MA, typename VTAU, typename MC>
typename RestrictTo<IsHostRealGeMatrix<MA>::value
                 && IsHostRealDenseVector<VTAU>::value
                 && IsHostRealGeMatrix<MC>::value,
         void>::Type
unmql(Side         side,
      Transpose    trans,
      MA           &&A,
      const VTAU   &tau,
      MC           &&C)
{
    typedef typename RemoveRef<MA>::Type::Vector  WorkVector;

    WorkVector  work;
    unmql(side, trans, A, tau, C, work);
}

//
//  Variant for convenience: c is vector
//
template <typename MA, typename VTAU, typename VC, typename VWORK>
typename RestrictTo<IsHostRealGeMatrix<MA>::value
                 && IsHostRealDenseVector<VTAU>::value
                 && IsHostRealDenseVector<VC>::value
                 && IsHostRealDenseVector<VWORK>::value,
         void>::Type
unmql(Side         side,
      Transpose    trans,
      MA           &&A,
      const VTAU   &tau,
      VC           &&c,
      VWORK        &&work)
{
//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename RemoveRef<VC>::Type    VectorC;

    typedef typename VectorC::ElementType  ElementType;
    typedef typename VectorC::IndexType    IndexType;

    const IndexType    n     = c.length();
    const StorageOrder order = MatrixA::Engine::order;

    GeMatrix<FullStorageView<ElementType, order> >  C(n, 1, c, n);

    unmql(side, trans, A, tau, C, work);
}

//
//  Variant for convenience: c is vector and workspace gets created
//                           temporarily.
//
template <typename MA, typename VTAU, typename VC>
typename RestrictTo<IsHostRealGeMatrix<MA>::value
                 && IsHostRealDenseVector<VTAU>::value
                 && IsHostRealDenseVector<VC>::value,
         void>::Type
unmql(Side         side,
      Transpose    trans,
      MA           &&A,
      const VTAU   &tau,
      VC           &&c)
{
//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename RemoveRef<VC>::Type    VectorC;

    typedef typename VectorC::ElementType  ElementType;
    typedef typename VectorC::IndexType    IndexType;

    const IndexType    n     = c.length();
    const StorageOrder order = MatrixA::Engine::order;

    GeMatrix<FullStorageView<ElementType, order> >  C(n, 1, c, n);

    unmql(side, trans, A, tau, C);
}


//-- unmql_wsq [worksize query] ------------------------------------------------

template <typename MA, typename MC>
typename RestrictTo<IsHostRealGeMatrix<MA>::value
                 && IsHostRealGeMatrix<MC>::value,
         typename RemoveRef<MC>::Type::IndexType>::Type
unmql_wsq(Side        side,
          Transpose   trans,
          MA          &&A,
          MC          &&C)
{
    typedef typename RemoveRef<MC>::Type    MatrixC;
    typedef typename MatrixC::IndexType     IndexType;

//
//  Test the input parameters
//
#   ifndef NDEBUG
    const IndexType m = C.numRows();
    const IndexType n = C.numCols();
    const IndexType k = A.numCols();

    if (side==Left) {
        ASSERT(A.numRows()==m);
    } else {
        ASSERT(A.numCols()==n);
    }
#   endif

//
//  Call implementation
//
    const IndexType info = external::unmql_wsq_impl(side, trans, A, C);

    return info;
}

#endif // USE_CXXMAGMA

} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_IMPL_UNMQL_TCC
