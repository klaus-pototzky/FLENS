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


#ifndef PLAYGROUND_FLENS_MAGMA_PO_POSV_TCC
#define PLAYGROUND_FLENS_MAGMA_PO_POSV_TCC 1

#include <flens/blas/blas.h>
#include <playground/flens/magma/magma.h>

namespace flens { namespace magma {

//== generic lapack implementation =============================================


//== interface for native magma ===============================================

#ifdef USE_CXXMAGMA

namespace external {

//-- (ge)sv [real variant] -----------------------------------------

template <typename MA, typename MB>
typename SyMatrix<MA>::IndexType
posv_impl(SyMatrix<MA> &A, GeMatrix<MB> &B)
{
    typedef typename SyMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::posv<IndexType>(lapack::getF77Char(A.upLo()),
                                               A.numRows(),
                                               B.numCols(),
                                               A.data(),
                                               A.leadingDimension(),
                                               B.data(),
                                               B.leadingDimension());
    ASSERT(info>=0);
    return info;
}

//-- (ge)sv [complex variant] -----------------------------------------

template <typename MA, typename MB>
typename HeMatrix<MA>::IndexType
posv_impl(HeMatrix<MA> &A, GeMatrix<MB> &B)
{
    typedef typename HeMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::posv<IndexType>(lapack::getF77Char(A.upLo()),
                                               A.numRows(),
                                               B.numCols(),
                                               A.data(),
                                               A.leadingDimension(),
                                               B.data(),
                                               B.leadingDimension());
    ASSERT(info>=0);
    return info;
}

//-- (ge)sv [real variant] -----------------------------------------

template <typename MA, typename MB>
typename SyMatrix<MA>::IndexType
posv_gpu_impl(SyMatrix<MA> &A, GeMatrix<MB> &B)
{
    typedef typename SyMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::posv_gpu<IndexType>(lapack::getF77Char(A.upLo()),
                                                   A.numRows(),
                                                   B.numCols(),
                                                   A.data(),
                                                   A.leadingDimension(),
                                                   B.data(),
                                                   B.leadingDimension());
    ASSERT(info>=0);
    return info;
}

//-- (ge)sv [complex variant] -----------------------------------------

template <typename MA, typename MB>
typename HeMatrix<MA>::IndexType
posv_gpu_impl(HeMatrix<MA> &A, GeMatrix<MB> &B)
{
    typedef typename HeMatrix<MA>::IndexType  IndexType;

    IndexType info = cxxmagma::posv_gpu<IndexType>(lapack::getF77Char(A.upLo()),
                                                   A.numRows(),
                                                   B.numCols(),
                                                   A.data(),
                                                   A.leadingDimension(),
                                                   B.data(),
                                                   B.leadingDimension());
    ASSERT(info>=0);
    return info;
}

} // namespace external

#endif // USE_CXXMAGMA

//== public interface ==========================================================

#ifdef USE_CXXMAGMA

//-- (ge)sv [real variant] -----------------------------------------

template <typename MA, typename MB>
typename RestrictTo<IsHostRealSyMatrix<MA>::value
		  && IsHostGeMatrix<MB>::value,
	  typename RemoveRef<MA>::Type::IndexType>::Type
posv(MA &&A, MB &&B)
{
    LAPACK_DEBUG_OUT("(po)sv [real]");

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
    ASSERT(B.firstRow()==1);
    ASSERT(B.firstCol()==1);
    ASSERT(B.numRows()==A.numRows());
#   endif


//
//  Call implementation
//
    IndexType info = external::posv_impl(A, B);

    return info;
}

//-- (ge)sv [complex variant] -----------------------------------------

template <typename MA, typename MB>
typename RestrictTo<IsHostHeMatrix<MA>::value
                  && IsHostGeMatrix<MB>::value,
          typename RemoveRef<MA>::Type::IndexType>::Type
posv(MA &&A, MB &&B)
{
    LAPACK_DEBUG_OUT("(po)sv [complex]");

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
    ASSERT(B.firstRow()==1);
    ASSERT(B.firstCol()==1);
    ASSERT(B.numRows()==A.numRows());
#   endif


//
//  Call implementation
//
    IndexType info = external::posv_impl(A, B);

    return info;
}

//-- (po)sv [real variant] -----------------------------------------

template <typename MA, typename MB>
typename RestrictTo<IsDeviceRealSyMatrix<MA>::value
                  && IsDeviceGeMatrix<MB>::value,
          typename RemoveRef<MA>::Type::IndexType>::Type
posv(MA &&A, MB &&B)
{
    LAPACK_DEBUG_OUT("(po)sv [real]");

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
    ASSERT(B.firstRow()==1);
    ASSERT(B.firstCol()==1);
    ASSERT(B.numRows()==A.numRows());
#   endif

//
//  Call implementation
//
    IndexType info = external::posv_gpu_impl(A, B);

    return info;
}

//-- (ge)sv [real variant] -----------------------------------------

template <typename MA, typename MB>
typename RestrictTo<IsDeviceHeMatrix<MA>::value
                  && IsDeviceGeMatrix<MB>::value,
          typename RemoveRef<MA>::Type::IndexType>::Type
posv(MA &&A, MB &&B)
{
    LAPACK_DEBUG_OUT("(po)sv [complex]");

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
    ASSERT(B.firstRow()==1);
    ASSERT(B.firstCol()==1);
    ASSERT(B.numRows()==A.numRows());
#   endif

//
//  Call implementation
//
    IndexType info = external::posv_gpu_impl(A, B);

    return info;
}

//-- (ge)sv [variant if rhs is vector] -----------------------------------------

template <typename MA, typename VB>
typename RestrictTo<(IsRealSyMatrix<MA>::value ||
                      IsHeMatrix<MA>::value )
                  && IsDeviceDenseVector<VB>::value,
          typename RemoveRef<MA>::Type::IndexType>::Type
posv(MA &&A, VB &&b)
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

    return posv(A, B);
}

#endif

} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_PO_POSV_TCC
