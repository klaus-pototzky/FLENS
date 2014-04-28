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


#ifndef PLAYGROUND_FLENS_MAGMA_GE_SVD_H
#define PLAYGROUND_FLENS_MAGMA_GE_SVD_H 1

#include <flens/matrixtypes/matrixtypes.h>
#include <flens/vectortypes/vectortypes.h>

namespace flens { namespace magma {

namespace SVD {

    enum Job {
        All          = 'A', // = 'A': All columns are returned;
        Save         = 'S', // = 'S': Only min(m,n) columns are returned;
        Overwrite    = 'O', // = 'O': min(m,n) columns are saved in A
        None         = 'N'  // = 'N': No rows are computed.
    };

}

#ifdef USE_CXXMAGMA

//== (ge)svd ===================================================================
//
//  Real variant
//
template <typename MA, typename VS, typename MU, typename MVT, typename VWORK>
    typename RestrictTo<IsDeviceRealGeMatrix<MA>::value
                     && IsDeviceRealDenseVector<VS>::value
                     && IsDeviceRealGeMatrix<MU>::value
                     && IsDeviceRealGeMatrix<MVT>::value
                     && IsDeviceRealDenseVector<VWORK>::value,
             void>::Type
    svd(SVD::Job    jobU,
        SVD::Job    jobVT,
        MA          &&A,
        VS          &&s,
        MU          &&U,
        MVT         &&VT,
        VWORK       &&work);

//
//  Complex variant
//
template <typename MA, typename VS, typename MU, typename MVT, typename VWORK,
          typename VRWORK>
    typename RestrictTo<IsDeviceComplexGeMatrix<MA>::value
                     && IsDeviceRealDenseVector<VS>::value
                     && IsDeviceComplexGeMatrix<MU>::value
                     && IsDeviceComplexGeMatrix<MVT>::value
                     && IsDeviceComplexDenseVector<VWORK>::value
                     && IsDeviceRealDenseVector<VRWORK>::value,
             void>::Type
    svd(SVD::Job    jobU,
        SVD::Job    jobVT,
        MA          &&A,
        VS          &&s,
        MU          &&U,
        MVT         &&VT,
        VWORK       &&work,
        VRWORK      &&rwork);

//
//  Real variant with temporary workspace
//
template <typename MA, typename VS, typename MU, typename MVT>
    typename RestrictTo<IsDeviceRealGeMatrix<MA>::value
                     && IsDeviceRealDenseVector<VS>::value
                     && IsDeviceRealGeMatrix<MU>::value
                     && IsDeviceRealGeMatrix<MVT>::value,
             void>::Type
    svd(SVD::Job    jobU,
        SVD::Job    jobVT,
        MA          &&A,
        VS          &&s,
        MU          &&U,
        MVT         &&VT);



//
//  Complex variant with temporary workspace
//
template <typename MA, typename VS, typename MU, typename MVT>
    typename RestrictTo<IsDeviceComplexGeMatrix<MA>::value
                     && IsDeviceRealDenseVector<VS>::value
                     && IsDeviceComplexGeMatrix<MU>::value
                     && IsDeviceComplexGeMatrix<MVT>::value,
             void>::Type
    svd(SVD::Job    jobU,
        SVD::Job    jobVT,
        MA          &&A,
        VS          &&s,
        MU          &&U,
        MVT         &&VT);


//== workspace query ===========================================================

template <typename MA, typename VS, typename MU, typename MVT, typename VWORK>
    typename RestrictTo<IsDeviceGeMatrix<MA>::value,
             typename RemoveRef<MA>::Type::IndexType>::Type
    svd_wsq(SVD::Job    jobU,
            SVD::Job    jobVT,
            MA          &&A);

#endif // USE_CXXMAGMA


} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_GE_SVD_H
