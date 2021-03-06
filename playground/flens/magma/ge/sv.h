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


#ifndef PLAYGROUND_FLENS_MAGMA_GE_SV_H
#define PLAYGROUND_FLENS_MAGMA_GE_SV_H 1

#include <flens/lapack/typedefs.h>
#include <flens/matrixtypes/matrixtypes.h>
#include <flens/vectortypes/vectortypes.h>

namespace flens { namespace magma {

#ifdef USE_CXXMAGMA
  
//== (ge)sv ====================================================================
//
//  Real and complex [Host variant]
//
template <typename MA, typename VPIV, typename MB>
    typename RestrictTo<IsHostGeMatrix<MA>::value
                     && IsHostIntegerDenseVector<VPIV>::value
                     && IsHostGeMatrix<MB>::value,
             typename RemoveRef<MA>::Type::IndexType>::Type
    sv(MA &&A, VPIV &&piv, MB &&B);
    
//== (ge)sv ====================================================================
//
//  Real and complex [GPU variant]
//
template <typename MA, typename VPIV, typename MB>
    typename RestrictTo<IsDeviceGeMatrix<MA>::value
                     && IsHostIntegerDenseVector<VPIV>::value
                     && IsDeviceGeMatrix<MB>::value,
             typename RemoveRef<MA>::Type::IndexType>::Type
    sv(MA &&A, VPIV &&piv, MB &&B);

//== (ge)sv variant if rhs is vector ===========================================
//
//  Real and complex [Host and GPU variant]
//
template <typename MA, typename VPIV, typename VB>
    typename RestrictTo<((IsHostGeMatrix<MA>::value
                            && IsHostDenseVector<VB>::value) ||
                         (IsDeviceGeMatrix<MA>::value
                            && IsDeviceDenseVector<VB>::value))
                     && IsHostIntegerDenseVector<VPIV>::value,
             typename RemoveRef<MA>::Type::IndexType>::Type
    sv(MA &&A, VPIV &&piv, VB &&b);
    
#endif // USE_CXXMAGMA

} } // namespace lapack, flens

#endif // PLAYGROUND_FLENS_MAGMA_GE_SV_H
