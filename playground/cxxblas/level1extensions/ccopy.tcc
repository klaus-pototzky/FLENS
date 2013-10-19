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

#ifndef PLAYGROUND_CXXBLAS_LEVEL1EXTENSIONS_CCOPY_TCC
#define PLAYGROUND_CXXBLAS_LEVEL1EXTENSIONS_CCOPY_TCC 1

#include <cxxblas/auxiliary/auxiliary.h>

namespace cxxblas {

#if defined(HAVE_CLBLAS) || defined(HAVE_CUBLAS)

template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
void
ccopy(IndexType n, const T *x, IndexType incX, 
      flens::device_ptr<T, STORAGETYPE> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("ccopy_generic [real] [host -> device]");
    copy(n, x, incX, y, incY);
}

template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
void
ccopy(IndexType n, const std::complex<T> *x, IndexType incX, 
      flens::device_ptr<std::complex<T>, STORAGETYPE> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("ccopy_generic [complex] [host -> device]");
    
    copy(n, x, incX, y, incY);
    scal(n, T(-1), cast_ptr_to_imag(y), incY);
}

template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
void
ccopy(IndexType n, const flens::device_ptr<T, STORAGETYPE> x, IndexType incX, 
      T *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("ccopy_generic [real] [device -> host]");
    copy(n, x, incX, y, incY);
}

template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
void
ccopy(IndexType n, const flens::device_ptr<std::complex<T>, STORAGETYPE> x, IndexType incX, 
      std::complex<T> *y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("ccopy_generic [complex] [device -> host]");
    
    copy(n, x, incX, y, incY);
    scal(n, T(-1), cast_ptr_to_imag(y), incY);
}

template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
void
ccopy(IndexType n, const flens::device_ptr<T, STORAGETYPE> x, IndexType incX, 
      flens::device_ptr<T, STORAGETYPE> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("ccopy_generic [real] [device -> host]");
    copy(n, x, incX, y, incY);
}

template <typename IndexType, typename T, flens::StorageType STORAGETYPE>
void
ccopy(IndexType n, const flens::device_ptr<std::complex<T>, STORAGETYPE> x, IndexType incX, 
      flens::device_ptr<std::complex<T>, STORAGETYPE> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("ccopy_generic [complex] [device -> host]");
    
    copy(n, x, incX, y, incY);
    scal(n, T(-1), cast_ptr_to_imag(y), incY);
}

#endif
} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL1EXTENSIONS_CCOPY_TCC
