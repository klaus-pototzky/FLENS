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

#ifndef PLAYGROUND_FLENS_MAGMA_MAGMA_H
#define PLAYGROUND_FLENS_MAGMA_MAGMA_H 1



//
//  Select MAGMA preferences generic vs external implementation.
//

#define MAGMA_SELECT  external


//
//  If an external LAPACK implementation is available include headers
//
#ifdef USE_CXXMAGMA
#   include <playground/cxxmagma/cxxmagma.h>
#endif

#include <cmath>

#include <flens/lapack/auxiliary/getf77char.h>

#include <playground/flens/magma/ge/ev.h>
#include <playground/flens/magma/ge/lqf.h>
#include <playground/flens/magma/ge/qlf.h>
#include <playground/flens/magma/ge/qrf.h>
#include <playground/flens/magma/ge/sv.h>
#include <playground/flens/magma/ge/svd.h>
#include <playground/flens/magma/ge/trf.h>
#include <playground/flens/magma/ge/tri.h>
#include <playground/flens/magma/ge/trs.h>

#include <playground/flens/magma/impl/ormql.h>
#include <playground/flens/magma/impl/ormqr.h>
#include <playground/flens/magma/impl/unmql.h>
#include <playground/flens/magma/impl/unmqr.h>

#include <playground/flens/magma/po/posv.h>
#include <playground/flens/magma/po/potrf.h>
#include <playground/flens/magma/po/potri.h>
#endif // FLENS_LAPACK_LAPACK_H
