/*
 *   Copyright (c) 2007-2013, Michael Lehn, Klaus Pototzky
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

#ifndef PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAY_TCC
#define PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAY_TCC 1

#include <cassert>
#include <cxxblas/level1/copy.h>
#include <flens/auxiliary/auxiliary.h>
#include <playground/flens/storage/devicearray/devicearray.h>
#include <playground/flens/storage/devicearray/devicearrayview.h>
#include <playground/flens/storage/devicearray/constdevicearrayview.h>

namespace flens {
  
#ifdef HAVE_DEVICE_STORAGE

template <typename T, typename I, typename A>
DeviceArray<T, I, A>::DeviceArray()
    : _data(), _length(0), _firstIndex(1)
{
}

template <typename T, typename I, typename A>
DeviceArray<T, I, A>::DeviceArray(IndexType length, IndexType firstIndex,
                      const ElementType &value, const Allocator &allocator)
    : _data(), _allocator(allocator), _length(length), _firstIndex(firstIndex)
{
    ASSERT(_length>=0);

    _allocate(value);
}

template <typename T, typename I, typename A>
DeviceArray<T, I, A>::DeviceArray(const DeviceArray &rhs)
    : _data(), _allocator(rhs.allocator()),
      _length(rhs.length()), _firstIndex(rhs.firstIndex())
{
    ASSERT(_length>=0);

    if (length()>0) {
        _allocate();
        cxxblas::copy(length(), rhs.data(), rhs.stride(), data(), stride());
    }
}

template <typename T, typename I, typename A>
template <typename RHS>
DeviceArray<T, I, A>::DeviceArray(const RHS &rhs)
    : _data(), _allocator(rhs.allocator()),
      _length(rhs.length()), _firstIndex(rhs.firstIndex())
{
    if (length()>0) {
        _allocate();
        cxxblas::copy(length(), rhs.data(), rhs.stride(), data(), stride());
    }
}

template <typename T, typename I, typename A>
DeviceArray<T, I, A>::~DeviceArray()
{
    _release();
}

//-- operators -----------------------------------------------------------------

template <typename T, typename I, typename A>
const typename DeviceArray<T, I, A>::ElementType &
DeviceArray<T, I, A>::operator()(IndexType index) const
{
    ASSERT(0);
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return _data.get()[index-_firstIndex];
}

template <typename T, typename I, typename A>
typename DeviceArray<T, I, A>::ElementType &
DeviceArray<T, I, A>::operator()(IndexType index)
{
    ASSERT(0);
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return _data.get()[index-_firstIndex];
}

template <typename T, typename I, typename A>
typename DeviceArray<T, I, A>::IndexType
DeviceArray<T, I, A>::firstIndex() const
{
    return _firstIndex;
}

template <typename T, typename I, typename A>
typename DeviceArray<T, I, A>::IndexType
DeviceArray<T, I, A>::lastIndex() const
{
    return _firstIndex+_length-IndexType(1);
}

template <typename T, typename I, typename A>
typename DeviceArray<T, I, A>::IndexType
DeviceArray<T, I, A>::length() const
{
    return _length;
}

template <typename T, typename I, typename A>
typename DeviceArray<T, I, A>::IndexType
DeviceArray<T, I, A>::stride() const
{
    return IndexType(1);
}


template <typename T, typename I, typename A>
const typename DeviceArray<T, I, A>::ConstPointerType
DeviceArray<T, I, A>::data() const
{
    return _data;
}

template <typename T, typename I, typename A>
typename DeviceArray<T, I, A>::PointerType
DeviceArray<T, I, A>::data()
{
    return _data;
}

template <typename T, typename I, typename A>
const typename DeviceArray<T, I, A>::ConstPointerType
DeviceArray<T, I, A>::data(IndexType index) const
{
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return _data.shift(index-_firstIndex);
}

template <typename T, typename I, typename A>
typename DeviceArray<T, I, A>::PointerType
DeviceArray<T, I, A>::data(IndexType index)
{
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return _data.shift(index-_firstIndex);
}

template <typename T, typename I, typename A>
const typename DeviceArray<T, I, A>::Allocator &
DeviceArray<T, I, A>::allocator() const
{
    return _allocator;
}

template <typename T, typename I, typename A>
bool
DeviceArray<T, I, A>::resize(IndexType length, IndexType firstIndex,
                             const ElementType &value)
{
    if (length!=_length) {
        _release();
        _length = length;
        _firstIndex = firstIndex;
        _allocate(value);
        return true;
    }
    changeIndexBase(firstIndex);
    return false;
}

template <typename T, typename I, typename A>
template <typename ARRAY>
bool
DeviceArray<T, I, A>::resize(const ARRAY &rhs, const ElementType &value)
{
    return resize(rhs.length(), rhs.firstIndex(), value);
}

template <typename T, typename I, typename A>
bool
DeviceArray<T, I, A>::fill(const ElementType &value)
{
    flens::fill_n(_data, length(), value);
    return true;
}

template <typename T, typename I, typename A>
void
DeviceArray<T, I, A>::changeIndexBase(IndexType firstIndex)
{
    _firstIndex = firstIndex;
}

template <typename T, typename I, typename A>
const typename DeviceArray<T, I, A>::ConstView
DeviceArray<T, I, A>::view(IndexType from, IndexType to,
                           IndexType stride, IndexType firstViewIndex) const
{
    const IndexType length = (to-from)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    const PointerType  newdata = (length!=0) ? data(from) : PointerType(NULL, 0);

    if (length!=0) {
        ASSERT(firstIndex()<=from);
        ASSERT(lastIndex()>=to);
        ASSERT(from<=to);
    }
    ASSERT(stride>=1);
#   else
    const PointerType  newdata = data(from);
#   endif

    return ConstView(length, newdata, stride, firstViewIndex, allocator());
}

template <typename T, typename I, typename A>
typename DeviceArray<T, I, A>::View
DeviceArray<T, I, A>::view(IndexType from, IndexType to,
                           IndexType stride, IndexType firstViewIndex)
{
    const IndexType     length = (to-from)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    PointerType         newdata = (length!=0) ? data(from) : PointerType(NULL, 0);

    if (length!=0) {
        ASSERT(firstIndex()<=from);
        ASSERT(lastIndex()>=to);
        ASSERT(from<=to);
    }
    ASSERT(stride>=1);
#   else
    PointerType         newdata = data(from);
#   endif

    return View(length, newdata, stride, firstViewIndex, allocator());
}

//-- private methods -----------------------------------------------------------

template <typename T, typename I, typename A>
void
DeviceArray<T, I, A>::_raw_allocate()
{
    ASSERT(!_data.get());
    ASSERT(length()>=0);

    if (length()>0) {
        _data = _allocator.allocate(_length);
        ASSERT(_data.get());
    }
}

template <typename T, typename I, typename A>
void
DeviceArray<T, I, A>::_allocate(const ElementType &value)
{
    _raw_allocate();
    flens::fill_n(_data, length(), value);
}

template <typename T, typename I, typename A>
void
DeviceArray<T, I, A>::_release()
{
    if (_data.get()) {
        ASSERT(length()>0);
        _allocator.deallocate(_data, _length);
        _data = PointerType(NULL, 0);
    }
    ASSERT(_data.get()==0);
}

#endif // defined(HAVE_CUBLAS) || defined(HAVE_CLBLAS)

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAY_TCC
