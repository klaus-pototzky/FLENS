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

#ifndef PLAYGROUND_FLENS_VECTORTYPES_DENSEVECTORTYPETRAITS_H
#define PLAYGROUND_FLENS_VECTORTYPES_DENSEVECTORTYPETRAITS_H 1

#include <flens/vectortypes/impl/densevector.h>
#include <flens/storage/array/isarraystorage.h>
#include <playground/flens/storage/devicearray/isdevicearraystorage.h>

namespace flens {

#ifdef HAVE_DEVICE_STORAGE
  
//
//  IsHostDenseVector
//

template <typename T>
struct IsHostDenseVector
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsDenseVector<TT>::value
                           && IsArrayStorage<typename TT::Engine>::value;
};


//
//  IsDeviceDenseVector
//

template <typename T>
struct IsDeviceDenseVector
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsDenseVector<TT>::value
                           && IsDeviceArrayStorage<typename TT::Engine>::value;
};

//
//  IsHostIntegerDenseVector
//

template <typename T>
struct IsHostIntegerDenseVector
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsDenseVector<TT>::value
                           && IsArrayStorage<typename TT::Engine>::value
                           && IsInteger<typename TT::ElementType>::value;
};

//
//  IsDeviceIntegerDenseVector
//

template <typename T>
struct IsDeviceIntegerDenseVector
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsDenseVector<TT>::value
                           && IsDeviceArrayStorage<typename TT::Engine>::value
                           && IsInteger<typename TT::ElementType>::value;
};

//
//  IsHostRealDenseVector
//

template <typename T>
struct IsHostRealDenseVector
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsDenseVector<TT>::value
                           && IsArrayStorage<typename TT::Engine>::value
                           && IsNotComplex<typename TT::ElementType>::value;
};

//
//  IsRealDenseVector
//

template <typename T>
struct IsDeviceRealDenseVector
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsDenseVector<TT>::value
                           && IsDeviceArrayStorage<typename TT::Engine>::value
                           && IsNotComplex<typename TT::ElementType>::value;
};



//
//  IsHostComplexDenseVector
//

template <typename T>
struct IsHostComplexDenseVector
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsDenseVector<TT>::value
                           && IsArrayStorage<typename TT::Engine>::value
                           && IsComplex<typename TT::ElementType>::value;
};

//
//  IsDeviceComplexDenseVector
//

template <typename T>
struct IsDeviceComplexDenseVector
{
    typedef typename std::remove_reference<T>::type  TT;

    static const bool value = IsDenseVector<TT>::value
                           && IsDeviceArrayStorage<typename TT::Engine>::value
                           && IsComplex<typename TT::ElementType>::value;
};

#endif // HAVE_DEVICE_STORAGE

} // namespace flens

#endif // PLAYGROUND_FLENS_VECTORTYPES_DENSEVECTORTYPETRAITS_H
