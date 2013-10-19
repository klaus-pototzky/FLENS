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

#ifndef PLAYGROUND_FLENS_STORAGE_ARRAY_IMAGDEVICEARRAY_H
#define PLAYGROUND_FLENS_STORAGE_ARRAY_IMAGDEVICEARRAY_H 1

#include <complex>
#include <playground/flens/storage/devicearray/devicearray.h>
#include <playground/flens/storage/devicearray/devicearrayview.h>
#include <playground/flens/storage/devicearray/constdevicearrayview.h>

namespace flens {

#ifdef HAVE_DEVICE_STORAGE

template <typename PT, typename I, typename A>
struct ImagArray<DeviceArray<std::complex<PT>, I, A> >
{
    typedef ConstDeviceArrayView<PT, I, A>                  ConstView;
    typedef DeviceArrayView<PT, I, A>                       View;
};

template <typename PT, typename I, typename A>
struct ImagArray<DeviceArrayView<std::complex<PT>, I, A> >
{
    typedef ConstDeviceArrayView<PT, I, A>                  ConstView;
    typedef DeviceArrayView<PT, I, A>                       View;
};

template <typename PT, typename I, typename A>
struct ImagArray<ConstDeviceArrayView<std::complex<PT>, I, A> >
{
    typedef ConstDeviceArrayView<PT, I, A>                  ConstView;
    typedef DeviceArrayView<PT, I, A>                       View;
};

#endif // defined(HAVE_CUBLAS) || defined(HAVE_CLBLAS)

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_ARRAY_IMAGDEVICEARRAY_H
