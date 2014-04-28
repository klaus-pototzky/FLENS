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


#ifndef PLAYGROUND_FLENS_MAGMA_GE_QLF_TCC
#define PLAYGROUND_FLENS_MAGMA_GE_QLF_TCC 1

#include <flens/blas/blas.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic magma implementation =============================================


//== interface for native magma ===============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- (ge)qlf [real and complex variant an CPU] ---------------------------------

template <typename MA, typename VTAU, typename VWORK>
void
qlf_impl(GeMatrix<MA> &A, DenseVector<VTAU> &tau, DenseVector<VWORK> &work)
{
    using std::max;
    typedef typename GeMatrix<MA>::ElementType      ElementType;
    typedef typename GeMatrix<MA>::IndexType        IndexType;

    if (work.length()==0) {
      
        const IndexType N  = A.numCols();
        const IndexType M  = A.numRows();
        const IndexType NB = cxxmagma::geqlf_nb_query<IndexType, ElementType>(M);
        const IndexType LWORKMIN =  N*NB;

        work.resize(LWORKMIN);
    }

    cxxmagma::geqlf<IndexType>(A.numRows(),
                               A.numCols(),
                               A.data(),
                               A.leadingDimension(),
                               tau.data(),
                               work.data(),
                               work.length());
}

} // namespace external

#endif // USE_CXXMAGMA

//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- (ge)qlf [real/complex variant on CPU] -------------------------------------

template <typename MA, typename VTAU, typename VWORK>
typename RestrictTo<IsHostGeMatrix<MA>::value
                 && IsHostDenseVector<VTAU>::value
                 && IsHostDenseVector<VWORK>::value,
          void>::Type
qlf(MA &&A, VTAU &&tau, VWORK &&work)
{
    LAPACK_DEBUG_OUT("qlf [complex]");
    
    using std::min;
//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type    MatrixA;
    typedef typename MatrixA::IndexType     IndexType;

    const IndexType m = A.numRows();
    const IndexType n = A.numCols();
    const IndexType k = min(m,n);

#   ifndef NDEBUG
//
//  Test the input parameters
//
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT(tau.firstIndex()==1);
    ASSERT(work.firstIndex()==1);

    ASSERT(tau.length()==0 || tau.length()==k);
    ASSERT(work.length()>=n || work.length()==IndexType(0));
#   endif

    if (tau.length()==0) {
        tau.resize(k);
    }


//
//  Call implementation
//
    external::qlf_impl(A, tau, work);

}

//-- (ge)tri [real/complex variant with temporary workspace] -------------------
template <typename MA, typename VTAU>
typename RestrictTo<IsGeMatrix<MA>::value
                 && IsDenseVector<VTAU>::value,
         void>::Type
qlf(MA &&A, VTAU &&tau)
{
    typedef typename RemoveRef<MA>::Type::Vector WorkVector;

    WorkVector  work;
    qlf(A, tau, work);
}

#endif // USE_CXXMAGMA


} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_GE_QLF_TCC
