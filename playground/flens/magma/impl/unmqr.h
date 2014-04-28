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

#ifndef PLAYGROUND_FLENS_MAGMA_IMPL_UNMQR_H
#define PLAYGROUND_FLENS_MAGMA_IMPL_UNMQR_H 1

#include <flens/matrixtypes/matrixtypes.h>
#include <flens/vectortypes/vectortypes.h>

namespace flens { namespace magma {
  
#ifdef USE_CXXMAGMA

//== unmqr [CPU Variants]======================================================

template <typename MA, typename VTAU, typename MC, typename VWORK>
    typename RestrictTo<IsHostComplexGeMatrix<MA>::value
                     && IsHostComplexDenseVector<VTAU>::value
                     && IsHostComplexGeMatrix<MC>::value
                     && IsHostComplexDenseVector<VWORK>::value,
             void>::Type
    unmqr(Side         side,
          Transpose    trans,
          MA           &&A,
          const VTAU   &tau,
          MC           &&C,
          VWORK        &&work);

//
//  Variant with temporary workspace
//
template <typename MA, typename VTAU, typename MC>
    typename RestrictTo<IsHostComplexGeMatrix<MA>::value
                     && IsHostComplexDenseVector<VTAU>::value
                     && IsHostComplexGeMatrix<MC>::value,
             void>::Type
    unmqr(Side         side,
          Transpose    trans,
          MA           &&A,
          const VTAU   &tau,
          MC           &&C);

//
//  Variant for convenience: c is vector
//
template <typename MA, typename VTAU, typename VC, typename VWORK>
    typename RestrictTo<IsHostComplexGeMatrix<MA>::value
                     && IsHostComplexDenseVector<VTAU>::value
                     && IsHostComplexDenseVector<VC>::value
                     && IsHostComplexDenseVector<VWORK>::value,
             void>::Type
    unmqr(Side         side,
          Transpose    trans,
          MA           &&A,
          const VTAU   &tau,
          VC           &&c,
          VWORK        &&work);

//
//  Variant for convenience: c is vector and workspace gets created
//                           temporarily.
//
template <typename MA, typename VTAU, typename VC>
    typename RestrictTo<IsHostComplexGeMatrix<MA>::value
                     && IsHostComplexDenseVector<VTAU>::value
                     && IsHostComplexDenseVector<VC>::value,
             void>::Type
    unmqr(Side         side,
          Transpose    trans,
          MA           &&A,
          const VTAU   &tau,
          VC           &&c);

//== unmqr [GPU Variants]======================================================

template <typename MA, typename VTAU, typename MC, typename VT, typename VWORK>
    typename RestrictTo<IsDeviceComplexGeMatrix<MA>::value
                     && IsHostComplexDenseVector<VTAU>::value
                     && IsDeviceComplexGeMatrix<MC>::value
                     && IsDeviceComplexDenseVector<VT>::value
                     && IsHostComplexDenseVector<VWORK>::value,
             void>::Type
    unmqr(Side         side,
          Transpose    trans,
          MA           &&A,
          const VTAU   &tau,
          MC           &&C,
	  const VT     &t,
          VWORK        &&work);

//
//  Variant with temporary workspace
//
template <typename MA, typename VTAU, typename MC, typename VT>
    typename RestrictTo<IsDeviceComplexGeMatrix<MA>::value
                     && IsHostComplexDenseVector<VTAU>::value
                     && IsDeviceComplexGeMatrix<MC>::value
                     && IsDeviceComplexDenseVector<VT>::value,
             void>::Type
    unmqr(Side         side,
          Transpose    trans,
          MA           &&A,
          const VTAU   &tau,
          MC           &&C,
          const VT     &t);

//
//  Variant for convenience: c is vector
//
template <typename MA, typename VTAU, typename VC, typename VT, typename VWORK>
    typename RestrictTo<IsDeviceComplexGeMatrix<MA>::value
                     && IsHostComplexDenseVector<VTAU>::value
                     && IsDeviceComplexDenseVector<VC>::value
                     && IsDeviceComplexDenseVector<VT>::value
                     && IsHostComplexDenseVector<VWORK>::value,
             void>::Type
    unmqr(Side         side,
          Transpose    trans,
          MA           &&A,
          const VTAU   &tau,
          VC           &&c,
          const VT     &t,
          VWORK        &&work);

//
//  Variant for convenience: c is vector and workspace gets created
//                           temporarily.
//
template <typename MA, typename VTAU, typename VC, typename VT>
    typename RestrictTo<IsDeviceComplexGeMatrix<MA>::value
                     && IsHostComplexDenseVector<VTAU>::value
                     && IsDeviceComplexDenseVector<VC>::value
                     && IsDeviceComplexDenseVector<VT>::value,
             void>::Type
    unmqr(Side         side,
          Transpose    trans,
          MA           &&A,
          const VTAU   &tau,
          VC           &&c,
          const VT     &t);

//== workspace query ===========================================================

template <typename MA, typename MC>
    typename RestrictTo<IsHostComplexGeMatrix<MA>::value
                     && IsHostComplexGeMatrix<MC>::value,
             typename RemoveRef<MC>::Type::IndexType>::Type
    unmqr_wsq(Side        side,
              Transpose   trans,
              MA          &&A,
              MC          &&C);
    
template <typename MA, typename MC>
    typename RestrictTo<IsDeviceComplexGeMatrix<MA>::value
                     && IsDeviceComplexGeMatrix<MC>::value,
             typename RemoveRef<MC>::Type::IndexType>::Type
    unmqr_wsq(Side        side,
              Transpose   trans,
              MA          &&A,
              MC          &&C);
    
#endif // USE_CXXMAGMA
    
} } // namespace magma, flens

#endif // PLAYGROUND_FLENS_MAGMA_IMPL_UNMQR_H
