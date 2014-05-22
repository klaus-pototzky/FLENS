/*
 *   Copyright (c) 2007-2012, Michael Lehn
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

#ifndef PLAYGROUND_FLENS_MATRIXTYPES_TRMATRIXTYPETRAITS_H
#define PLAYGROUND_FLENS_MATRIXTYPES_TRMATRIXTYPETRAITS_H 1

#include <flens/matrixtypes/triangular/impl/trmatrix.h>
#include <flens/storage/fullstorage/isfullstorage.h>
#include <playground/flens/storage/devicefullstorage/isdevicefullstorage.h>

namespace flens {

#ifdef HAVE_DEVICE_STORAGE

//
//  IsHostTrMatrix
//

template <typename T>
struct IsHostTrMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsTrMatrix<TT>::value
                           && IsFullStorage<typename TT::Engine>::value;
};


//
//  IsDeviceTrMatrix
//

template <typename T>
struct IsDeviceTrMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsTrMatrix<TT>::value
                           && IsDeviceFullStorage<typename TT::Engine>::value;
};

//
//  IsHostRealTrMatrix
//

template <typename T>
struct IsHostRealTrMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsTrMatrix<TT>::value
                           && IsFullStorage<typename TT::Engine>::value
                           && IsNotComplex<typename TT::ElementType>::value;
};

//
//  IsRealTrMatrix
//

template <typename T>
struct IsDeviceRealTrMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsTrMatrix<TT>::value
                           && IsDeviceFullStorage<typename TT::Engine>::value
                           && IsNotComplex<typename TT::ElementType>::value;
};



//
//  IsHostComplexTrMatrix
//

template <typename T>
struct IsHostComplexTrMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsTrMatrix<TT>::value
                           && IsFullStorage<typename TT::Engine>::value
                           && IsComplex<typename TT::ElementType>::value;
};

//
//  IsDeviceComplexTrMatrix
//

template <typename T>
struct IsDeviceComplexTrMatrix
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsTrMatrix<TT>::value
                           && IsDeviceFullStorage<typename TT::Engine>::value
                           && IsComplex<typename TT::ElementType>::value;
};

#endif // HAVE_DEVICE_STORAGE

} // namespace flens

#endif // PLAYGROUND_FLENS_MATRIXTYPES_TRMATRIXTYPETRAITS_H