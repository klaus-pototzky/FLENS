/*
 *   Copyright (c) 2012-2013, Michael Lehn, Klaus Pototzky
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

#ifndef FLENS_MATRIXTYPES_TRIANGULAR_IMPL_TRCRSMATRIX_H
#define FLENS_MATRIXTYPES_TRIANGULAR_IMPL_TRCRSMATRIX_H 1

#include <flens/auxiliary/auxiliary.h>
#include <flens/matrixtypes/triangular/triangularmatrix.h>
#include <flens/typedefs.h>

namespace flens {

template <typename CRS>
class TrCRSMatrix
    : public TriangularMatrix<TrCRSMatrix<CRS> >
{
    public:
        typedef CRS                             Engine;
        typedef typename Engine::ElementType    ElementType;
        typedef typename Engine::IndexType      IndexType;

        // -- constructors -----------------------------------------------------
        TrCRSMatrix();

        template <typename RHS>
            TrCRSMatrix(const Matrix<RHS> &rhs);

        // -- operators --------------------------------------------------------
        template <typename RHS>
            void
            operator=(const Matrix<RHS> &rhs);

        // -- methods ----------------------------------------------------------
        IndexType
        dim() const;

        IndexType
        numRows() const;

        IndexType
        numCols() const;

        IndexType
        indexBase() const;

        IndexType
        firstRow() const;

        IndexType
        lastRow() const;

        IndexType
        firstCol() const;

        IndexType
        lastCol() const;

        StorageUpLo
        upLo() const;

        StorageUpLo &
        upLo();

        // -- implementation ---------------------------------------------------
        const Engine &
        engine() const;

        Engine &
        engine();

        // forbidden: This constructor never should get called.  Hence we
        //            don't define an implementation.
        TrCRSMatrix(const TrCRSMatrix &rhs);

    private:

        Engine       _engine;
        StorageUpLo  _upLo;
};

//-- Traits --------------------------------------------------------------------
//
//  IsTrCRSMatrix
//
struct _TrCRSMatrixChecker
{

    struct Two {
        char x;
        char y;
    };

    static Two
    check(_AnyConversion);

    template <typename Any>
        static char
        check(const TrCRSMatrix<Any> &);
};

template <typename T>
struct IsTrCRSMatrix
{
    static T var;
    static const bool value = sizeof(_TrCRSMatrixChecker::check(var))==1;
};

//
//  IsRealTrCRSMatrix
//
template <typename T>
struct IsRealTrCRSMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsTrCRSMatrix<TT>::value
                           && IsNotComplex<typename TT::ElementType>::value;
};

//
//  IsComplexTrCRSMatrix
//
template <typename T>
struct IsComplexTrCRSMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsTrCRSMatrix<TT>::value
                           && IsComplex<typename TT::ElementType>::value;
};

} // namespace flens

#endif // FLENS_MATRIXTYPES_TRIANGULAR_IMPL_TRCRSMATRIX_H
