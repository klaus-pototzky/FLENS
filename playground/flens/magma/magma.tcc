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

#ifndef PLAYGROUND_FLENS_MAGMA_MAGMA_TCC
#define PLAYGROUND_FLENS_MAGMA_MAGMA_TCC 1

#ifdef USE_CXXMAGMA
#   include <playground/cxxmagma/cxxmagma.tcc>
#endif

#include <flens/lapack/auxiliary/getf77char.tcc>

#include <playground/flens/magma/ge/ev.tcc>
#include <playground/flens/magma/ge/lqf.tcc>
#include <playground/flens/magma/ge/qlf.tcc>
#include <playground/flens/magma/ge/qrf.tcc>
#include <playground/flens/magma/ge/sv.tcc>
#include <playground/flens/magma/ge/svd.tcc>
#include <playground/flens/magma/ge/trf.tcc>
#include <playground/flens/magma/ge/tri.tcc>
#include <playground/flens/magma/ge/trs.tcc>

#include <playground/flens/magma/impl/ormql.tcc>
#include <playground/flens/magma/impl/ormqr.tcc>
#include <playground/flens/magma/impl/unmql.tcc>
#include <playground/flens/magma/impl/unmqr.tcc>

#include <playground/flens/magma/po/posv.tcc>
#include <playground/flens/magma/po/potrf.tcc>
#include <playground/flens/magma/po/potri.tcc>

#endif // PLAYGROUND_FLENS_MAGMA_MAGMA_TCC
