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

#ifndef PLAYGROUND_FLENS_SOLVER_PCG_H
#define PLAYGROUND_FLENS_SOLVER_PCG_H 1

#include <limits>

#include <flens/lapack/typedefs.h>
#include <flens/matrixtypes/matrixtypes.h>
#include <flens/vectortypes/vectortypes.h>

namespace flens { namespace solver {

template <typename MP, typename MA, typename VX, typename VB>
    typename RestrictTo<IsMatrix<MP>::value
                     && IsSymmetricMatrix<MA>::value
                     && IsDenseVector<VX>::value
                     && IsDenseVector<VB>::value,
             typename RemoveRef<VX>::Type::IndexType>::Type
    pcg(const MP &P, const MA &A, VX &&x, const VB &b,
       typename ComplexTrait<typename RemoveRef<VX>::Type::ElementType>::PrimitiveType tol
                = std::numeric_limits<typename ComplexTrait<typename RemoveRef<VX>::Type::ElementType>::PrimitiveType>::epsilon(),
       typename RemoveRef<VX>::Type::IndexType maxIterations = std::numeric_limits<typename RemoveRef<VX>::Type::IndexType>::max());

} } // namespace solver, flens

#endif // PLAYGROUND_FLENS_SOLVER_PCG_H
