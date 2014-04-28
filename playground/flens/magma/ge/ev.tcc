/*
 *   Copyright (c) 2014, Klaus Pototzky
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

#ifndef PLAYGROUND_FLENS_MAGMA_GE_EV_TCC
#define PLAYGROUND_FLENS_MAGMA_GE_EV_TCC 1

#include <flens/blas/blas.h>
#include <flens/lapack/lapack.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic magma implementation =============================================

//== interface for native magma ===============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- (ge)ev_wsq [worksize query, real variant] ---------------------------------

template <typename MA>
typename RestrictTo<IsNotComplex<typename MA::ElementType>::value,
         Pair<typename MA::IndexType> >::Type
ev_wsq_impl(bool computeVL, bool computeVR, const GeMatrix<MA> &A)
{
    using std::max;

    typedef typename GeMatrix<MA>::ElementType  T;
    typedef typename GeMatrix<MA>::IndexType    IndexType;

//
//  Compute minimal workspace
//
    IndexType n       = A.numRows();  
    IndexType nb      = cxxmagma::gehrd_nb_query<IndexType, T>(n);
    IndexType minWork = (1+nb)*n;

//
//  Get optimal workspace from external MAGMA
//
    T           DUMMY, WORK;
    IndexType   LDVL = computeVL ? max(A.numRows(), IndexType(1)) : 1,
                LDVR = computeVR ? max(A.numRows(), IndexType(1)) : 1;
    IndexType   LWORK = -1;
    cxxmagma::geev<IndexType>(computeVL ? 'V' : 'N',
                              computeVR ? 'V' : 'N',
                              A.numRows(),
                              &DUMMY,
                              A.leadingDimension(),
                              &DUMMY,
                              &DUMMY,
                              &DUMMY,
                              LDVL,
                              &DUMMY,
                              LDVR,
                              &WORK,
                              LWORK);
    return Pair<IndexType>(minWork, WORK);
}

//-- (ge)ev_wsq [worksize query, complex variant] ------------------------------

template <typename MA>
typename RestrictTo<IsComplex<typename MA::ElementType>::value,
         Pair<typename MA::IndexType> >::Type
ev_wsq_impl(bool computeVL, bool computeVR, const GeMatrix<MA> &A)
{
    using std::max;

    typedef typename GeMatrix<MA>::ElementType          T;
    typedef typename ComplexTrait<T>::PrimitiveType     RT;
    typedef typename GeMatrix<MA>::IndexType            IndexType;

//
//  Compute minimal workspace
//
    IndexType n       = A.numRows();
    IndexType nb      = cxxmagma::gehrd_nb_query<IndexType, T>(n);
    IndexType minWork = (1+nb)*n;

//
//  Get optimal workspace from external MAGMA
//
    T           DUMMY, WORK;
    RT          RDUMMY;
    IndexType   LDVL = computeVL ? max(A.numRows(), IndexType(1)) : 1,
                LDVR = computeVR ? max(A.numRows(), IndexType(1)) : 1;
    IndexType   LWORK = -1;
    cxxmagma::geev<IndexType>(computeVL ? 'V' : 'N',
                              computeVR ? 'V' : 'N',
                              A.numRows(),
                              &DUMMY,
                              A.leadingDimension(),
                              &DUMMY,
                              &DUMMY,
                              LDVL,
                              &DUMMY,
                              LDVR,
                              &WORK,
                              LWORK,
                              &RDUMMY);
    return Pair<IndexType>(minWork, WORK.real());
}

//-- (ge)ev [real variant] -----------------------------------------------------

template <typename MA, typename VWR, typename VWI, typename MVL, typename MVR,
          typename VWORK>
typename GeMatrix<MA>::IndexType
ev_impl(bool                  computeVL,
        bool                  computeVR,
        GeMatrix<MA>          &A,
        DenseVector<VWR>      &wr,
        DenseVector<VWI>      &wi,
        GeMatrix<MVL>         &VL,
        GeMatrix<MVR>         &VR,
        DenseVector<VWORK>    &work)
{
    typedef typename GeMatrix<MA>::IndexType  IndexType;

    if (work.length()==0) {
        const auto ws = ev_wsq_impl(computeVL, computeVR, A);
        work.resize(ws.second, 1);
    }

    IndexType  info;
    info = cxxmagma::geev(computeVL ? 'V' : 'N',
                          computeVR ? 'V' : 'N',
                          A.numRows(),
                          A.data(),
                          A.leadingDimension(),
                          wr.data(),
                          wi.data(),
                          VL.data(),
                          VL.leadingDimension(),
                          VR.data(),
                          VR.leadingDimension(),
                          work.data(),
                          work.length());
    ASSERT(info>=0);
    return info;
}

//-- (ge)ev [complex variant] --------------------------------------------------

template <typename MA, typename VW, typename MVL, typename MVR, typename VWORK,
          typename VRWORK>
typename GeMatrix<MA>::IndexType
ev_impl(bool                  computeVL,
        bool                  computeVR,
        GeMatrix<MA>          &A,
        DenseVector<VW>       &w,
        GeMatrix<MVL>         &VL,
        GeMatrix<MVR>         &VR,
        DenseVector<VWORK>    &work,
        DenseVector<VRWORK>   &rWork)
{
    typedef typename GeMatrix<MA>::IndexType  IndexType;

    if (work.length()==0) {
        const auto ws = ev_wsq_impl(computeVL, computeVR, A);
        work.resize(ws.second, 1);
    }
    if (rWork.length()==0) {
        rWork.resize(2*A.numRows());
    }
    IndexType  info;
    info = cxxmagma::geev(computeVL ? 'V' : 'N',
                          computeVR ? 'V' : 'N',
                          A.numRows(),
                          A.data(),
                          A.leadingDimension(),
                          w.data(),
                          VL.data(),
                          VL.leadingDimension(),
                          VR.data(),
                          VR.leadingDimension(),
                          work.data(),
                          work.length(),
                          rWork.data());
    ASSERT(info>=0);
    return info;
}

} // namespace external

#endif // USE_CXXMAGMA


//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- (ge)ev [real variant] -----------------------------------------------------

template <typename MA, typename VWR, typename VWI, typename MVL, typename MVR,
          typename VWORK>
typename RestrictTo<IsHostRealGeMatrix<MA>::value
                 && IsHostRealDenseVector<VWR>::value
                 && IsHostRealDenseVector<VWI>::value
                 && IsHostRealGeMatrix<MVL>::value
                 && IsHostRealGeMatrix<MVR>::value
                 && IsHostRealDenseVector<VWORK>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
ev(bool     computeVL,
   bool     computeVR,
   MA       &&A,
   VWR      &&wr,
   VWI      &&wi,
   MVL      &&VL,
   MVR      &&VR,
   VWORK    &&work)
{
    LAPACK_DEBUG_OUT("(ge)ev [real]");

//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type     MatrixA;
    typedef typename MatrixA::IndexType      IndexType;

    const IndexType n = A.numRows();

//
//  Test the input parameters
//
#   ifndef NDEBUG
    ASSERT(A.numRows()==A.numCols());
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT(work.firstIndex()==1);


    ASSERT(wr.firstIndex()==1);
    ASSERT(wr.length()==0 || wr.length()==n);

    ASSERT(wi.firstIndex()==1);
    ASSERT(wi.length()==0 || wi.length()==n);

    if (computeVL) {
        ASSERT(VL.numRows()==VL.numCols());
        ASSERT(VL.numRows()==0 || VL.numRows()==n);
        ASSERT(VL.firstRow()==1);
        ASSERT(VL.firstCol()==1);
    }

    if (computeVR) {
        ASSERT(VR.numRows()==VR.numCols());
        ASSERT(VR.numRows()==0 || VR.numRows()==n);
        ASSERT(VR.firstRow()==1);
        ASSERT(VR.firstCol()==1);
    }
#   endif

//
//  Resize output arguments if they are empty and needed
//
    if (wr.length()==0) {
        wr.resize(n, 1);
    }
    if (wi.length()==0) {
        wi.resize(n, 1);
    }
    if (computeVL && VL.numRows()==0) {
        VL.resize(n, n, 1, 1);
    }
    if (computeVR && VR.numRows()==0) {
        VR.resize(n, n, 1, 1);
    }

//
//  Call implementation
//
    IndexType result = external::ev_impl(computeVL, computeVR,
                                         A, wr, wi, VL, VR,
                                         work);
    return result;
}


//-- (ge)ev [complex variant] -----------------------------------------------------

template <typename MA, typename VW, typename MVL, typename MVR, typename VWORK,
          typename VRWORK>
typename RestrictTo<IsHostComplexGeMatrix<MA>::value
                 && IsHostComplexDenseVector<VW>::value
                 && IsHostComplexGeMatrix<MVL>::value
                 && IsHostComplexGeMatrix<MVR>::value
                 && IsHostComplexDenseVector<VWORK>::value
                 && IsHostRealDenseVector<VRWORK>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
ev(bool     computeVL,
   bool     computeVR,
   MA       &&A,
   VW       &&w,
   MVL      &&VL,
   MVR      &&VR,
   VWORK    &&work,
   VRWORK   &&rWork)
{
    LAPACK_DEBUG_OUT("(ge)ev [complex]");

//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type      MatrixA;
    typedef typename MatrixA::IndexType       IndexType;

    const IndexType n = A.numRows();

//
//  Test the input parameters
//
#   ifndef NDEBUG
    ASSERT(A.numRows()==A.numCols());
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
    ASSERT(work.firstIndex()==1);
    ASSERT(rWork.firstIndex()==1);

    ASSERT(w.firstIndex()==1);
    ASSERT(w.length()==0 || w.length()==n);

    if (computeVL) {
        ASSERT(VL.numRows()==VL.numCols());
        ASSERT(VL.numRows()==0 || VL.numRows()==n);
        ASSERT(VL.firstRow()==1);
        ASSERT(VL.firstCol()==1);
    }

    if (computeVR) {
        ASSERT(VR.numRows()==VR.numCols());
        ASSERT(VR.numRows()==0 || VR.numRows()==n);
        ASSERT(VR.firstRow()==1);
        ASSERT(VR.firstCol()==1);
    }
#   endif

//
//  Resize output arguments if they are empty and needed
//
    if (w.length()==0) {
        w.resize(n, 1);
    }
    if (computeVL && VL.numRows()==0) {
        VL.resize(n, n, 1, 1);
    }
    if (computeVR && VR.numRows()==0) {
        VR.resize(n, n, 1, 1);
    }

//
//  Call external implementation
//
    IndexType result = external::ev_impl(computeVL, computeVR, A, w, VL, VR,
                                         work, rWork);
    return result;
}


//-- (ge)ev_wsq [worksize query, real variant] ---------------------------------

template <typename MA>
typename RestrictTo<IsHostRealGeMatrix<MA>::value,
         Pair<typename MA::IndexType> >::Type
ev_wsq(bool computeVL, bool computeVR, const MA &A)
{
    LAPACK_DEBUG_OUT("ev_wsq [real]");

//
//  Test the input parameters
//
#   ifndef NDEBUG
    ASSERT(A.numRows()==A.numCols());
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
#   endif

//
//  Call implementation
//
    const auto ws = external::ev_wsq_impl(computeVL, computeVR, A);

    return ws;
}

//-- (ge)ev_wsq [worksize query, complex variant] ------------------------------

template <typename MA>
typename RestrictTo<IsHostComplexGeMatrix<MA>::value,
         Pair<typename MA::IndexType> >::Type
ev_wsq(bool computeVL, bool computeVR, const MA &A)
{
    LAPACK_DEBUG_OUT("ev_wsq [complex]");

//
//  Test the input parameters
//
#   ifndef NDEBUG
    ASSERT(A.numRows()==A.numCols());
    ASSERT(A.firstRow()==1);
    ASSERT(A.firstCol()==1);
#   endif

//
//  Call implementation
//
    const auto ws = external::ev_wsq_impl(computeVL, computeVR, A);

    return ws;
}

//-- (ge)ev [real variant with temporary workspace] ----------------------------

template <typename MA, typename VWR, typename VWI, typename MVL, typename MVR>
typename RestrictTo<IsHostRealGeMatrix<MA>::value
                 && IsHostRealDenseVector<VWR>::value
                 && IsHostRealDenseVector<VWI>::value
                 && IsHostRealGeMatrix<MVL>::value
                 && IsHostRealGeMatrix<MVR>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
ev(bool     computeVL,
   bool     computeVR,
   MA       &&A,
   VWR      &&wr,
   VWI      &&wi,
   MVL      &&VL,
   MVR      &&VR)
{
//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type::Vector WorkVector;
    WorkVector work;
    return ev(computeVL, computeVR, A, wr, wi, VL, VR, work);
}

//-- (ge)ev [complex variant with temporary workspace] -------------------------

template <typename MA, typename VW, typename MVL, typename MVR>
typename RestrictTo<IsHostComplexGeMatrix<MA>::value
                 && IsHostComplexDenseVector<VW>::value
                 && IsHostComplexGeMatrix<MVL>::value
                 && IsHostComplexGeMatrix<MVR>::value,
          typename RemoveRef<MA>::Type::IndexType>::Type
ev(bool     computeVL,
   bool     computeVR,
   MA       &&A,
   VW       &&w,
   MVL      &&VL,
   MVR      &&VR)
{
//
//  Remove references from rvalue types
//
    typedef typename RemoveRef<MA>::Type::Vector        WorkVector;
    typedef typename RemoveRef<MA>::Type::ElementType   T;
    typedef typename ComplexTrait<T>::PrimitiveType     PT;
    typedef DenseVector<Array<PT> >                     RealWorkVector;

    WorkVector      work;
    RealWorkVector  rwork;

    return ev(computeVL, computeVR, A, w, VL, VR, work, rwork);
}

#endif // USE_CXXMAGMA

} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_GE_EV_TCC
