/*
 *   Copyright (c) 2013, Klaus Pototzky
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

#ifndef CXXBLAS_AUXILIARY_CAST_TCC
#define CXXBLAS_AUXILIARY_CAST_TCC 1

#include <cxxblas/auxiliary/cast.h>

namespace cxxblas {

template <typename T>
T*
cast_ptr_to_real(T *x) {
	return x;
}

template <typename T>
const T*
cast_ptr_to_real(const T *x) {
	return x;
}

template <typename T>
T*
cast_ptr_to_real(std::complex<T> *x) {
	T* ptr = reinterpret_cast<T*>(x);
	return ptr;
}

template <typename T>
const T*
cast_ptr_to_real(const std::complex<T> *x) {
	const T* ptr = reinterpret_cast<const T*>(x);
	return ptr;
}

template <typename T>
T*
cast_ptr_to_imag(T *x) {
        ASSERT(0);
	return x;
}

template <typename T>
const T*
cast_ptr_to_imag(const T *x) {
        ASSERT(0);
	return x;
}

template <typename T>
T*
cast_ptr_to_imag(std::complex<T> *x) {
	T* ptr = reinterpret_cast<T*>(x)+1;
	return ptr;
}

template <typename T>
const T*
cast_ptr_to_imag(const std::complex<T> *x) {
	const T* ptr = reinterpret_cast<const T*>(x)+1;
	return ptr;
}


} // namespace cxxblas

#endif // CXXBLAS_AUXILIARY_CAST_TCC
