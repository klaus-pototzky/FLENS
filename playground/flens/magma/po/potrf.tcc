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

#ifndef PLAYGROUND_FLENS_MAGMA_PO_POTRF_TCC
#define PLAYGROUND_FLENS_MAGMA_PO_POTRF_TCC 1

#include <algorithm>
#include <flens/blas/blas.h>
#include <flens/lapack/lapack.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic magma implementation =============================================

//== interface for native magma ===============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- potrf [real variant] ------------------------------------------------------

template <typename MA>
typename SyMatrix<MA>::IndexType
potrf_impl(SyMatrix<MA> &A)
{
    typedef typename SyMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::potrf<IndexType>(lapack::getF77Char(A.upLo()),
                                                A.dim(),
                                                A.data(),
                                                A.leadingDimension());
    ASSERT(info>=0);
    return info;
}


//-- potrf [real variant] ------------------------------------------------------

template <typename MA>
typename SyMatrix<MA>::IndexType
potrf_gpu_impl(SyMatrix<MA> &A)
{
    typedef typename SyMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::potrf_gpu<IndexType>(lapack::getF77Char(A.upLo()),
                                                    A.dim(),
                                                    A.data(),
                                                    A.leadingDimension());
    ASSERT(info>=0);
    return info;
}

//-- potrf [complex variant] ---------------------------------------------------

template <typename MA>
typename HeMatrix<MA>::IndexType
potrf_impl(HeMatrix<MA> &A)
{
    typedef typename HeMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::potrf<IndexType>(lapack::getF77Char(A.upLo()),
                                                A.dim(),
                                                A.data(),
                                                A.leadingDimension());
    ASSERT(info>=0);
    return info;
}

//-- potrf [complex variant] ---------------------------------------------------

template <typename MA>
typename HeMatrix<MA>::IndexType
potrf_gpu_impl(HeMatrix<MA> &A)
{
    typedef typename HeMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::potrf_gpu<IndexType>(lapack::getF77Char(A.upLo()),
                                                    A.dim(),
                                                    A.data(),
                                                    A.leadingDimension());
    ASSERT(info>=0);
    return info;
}


} // namespace external

#endif // USE_CXXMAGMA

//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- potrf [real variant, CPU] ------------------------------------------------------

template <typename MA>
typename RestrictTo<IsHostRealSyMatrix<MA>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
potrf(MA &&A)
{
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

//
//  Call implementation
//
    const IndexType info = external::potrf_impl(A);

    return info;
}


//-- potrf [real variant, GPU] ------------------------------------------------------

template <typename MA>
typename RestrictTo<IsDeviceRealSyMatrix<MA>::value,
         typename RemoveRef<MA>::Type::IndexType>::Type
potrf(MA &&A)
{
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

//
//  Call implementation
//
    const IndexType info = external::potrf_gpu_impl(A);

    return info;
}


//-- potrf [complex variant, CPU] ---------------------------------------------------

template <typename MA>
typename RestrictTo<IsHostHeMatrix<MA>::value,
	  typename RemoveRef<MA>::Type::IndexType>::Type
potrf(MA &&A)
{
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

//
//  Call implementation
//
    const IndexType info = external::potrf_impl(A);

    return info;
}


//-- potrf [complex variant, GPU] ---------------------------------------------------

template <typename MA>
    typename RestrictTo<IsDeviceHeMatrix<MA>::value,
             typename RemoveRef<MA>::Type::IndexType>::Type
    potrf(MA &&A)
{
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

//
//  Call implementation
//
    const IndexType info = external::potrf_gpu_impl(A);

    return info;
}

#endif //USE_CXXMAGMA

} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_PO_POTRF_TCC
