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

#ifndef PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAYVIEW_TCC
#define PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAYVIEW_TCC 1

#include <cassert>
#include <cxxblas/level1/copy.h>
#include <flens/auxiliary/auxiliary.h>
#include <playground/flens/storage/devicearray/devicearray.h>
#include <playground/flens/storage/devicearray/devicearrayview.h>
#include <playground/flens/storage/devicearray/constdevicearrayview.h>

namespace flens {

#ifdef HAVE_DEVICE_STORAGE

template <typename T, typename I, typename A>
DeviceArrayView<T, I, A>::DeviceArrayView(IndexType length, PointerType data,
                                          IndexType stride, IndexType firstIndex,
                                          const Allocator &allocator)
    : _data(data),
      _allocator(allocator),
      _length(length),
      _stride(stride),
      _firstIndex(firstIndex)
{
    ASSERT(_length>=0);
    ASSERT(_stride>0);
}

template <typename T, typename I, typename A>
DeviceArrayView<T, I, A>::DeviceArrayView(const DeviceArrayView &rhs)
    : _data(rhs._data),
      _allocator(rhs.allocator()),
      _length(rhs.length()),
      _stride(rhs.stride()),
      _firstIndex(rhs.firstIndex())
{
    ASSERT(_stride>0);
}

template <typename T, typename I, typename A>
template <typename RHS>
DeviceArrayView<T, I, A>::DeviceArrayView(RHS &rhs)
    : _data(rhs.data()),
      _allocator(rhs.allocator()),
      _length(rhs.length()),
      _stride(rhs.stride()),
      _firstIndex(rhs.firstIndex())
{
    ASSERT(_stride>0);
}

template <typename T, typename I, typename A>
DeviceArrayView<T, I, A>::~DeviceArrayView()
{
}

//-- operators -----------------------------------------------------------------

template <typename T, typename I, typename A>
const typename DeviceArrayView<T, I, A>::ElementType &
DeviceArrayView<T, I, A>::operator()(IndexType index) const
{
    ASSERT(0);
#   ifndef NDEBUG
    if (lastIndex()>=firstIndex()) {
        ASSERT(index>=firstIndex());
        ASSERT(index<=lastIndex());
    }
#   endif

    return _data[_stride*(index-_firstIndex)];
}

template <typename T, typename I, typename A>
typename DeviceArrayView<T, I, A>::ElementType &
DeviceArrayView<T, I, A>::operator()(IndexType index)
{
    ASSERT(0);
#   ifndef NDEBUG
    if (lastIndex()>=firstIndex()) {
        ASSERT(index>=firstIndex());
        ASSERT(index<=lastIndex());
    }
#   endif

    return _data[_stride*(index-_firstIndex)];
}

template <typename T, typename I, typename A>
typename DeviceArrayView<T, I, A>::IndexType
DeviceArrayView<T, I, A>::firstIndex() const
{
    return _firstIndex;
}

template <typename T, typename I, typename A>
typename DeviceArrayView<T, I, A>::IndexType
DeviceArrayView<T, I, A>::lastIndex() const
{
    return _firstIndex+_length-1;
}

template <typename T, typename I, typename A>
typename DeviceArrayView<T, I, A>::IndexType
DeviceArrayView<T, I, A>::length() const
{
    return _length;
}

template <typename T, typename I, typename A>
typename DeviceArrayView<T, I, A>::IndexType
DeviceArrayView<T, I, A>::stride() const
{
    return _stride;
}

template <typename T, typename I, typename A>
const typename DeviceArrayView<T, I, A>::ConstPointerType
DeviceArrayView<T, I, A>::data() const
{
    return _data;
}

template <typename T, typename I, typename A>
typename DeviceArrayView<T, I, A>::PointerType
DeviceArrayView<T, I, A>::data()
{
    return _data;
}

template <typename T, typename I, typename A>
const typename DeviceArrayView<T, I, A>::ConstPointerType
DeviceArrayView<T, I, A>::data(IndexType index) const
{
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return _data.shift(_stride*(index-_firstIndex));
}

template <typename T, typename I, typename A>
typename DeviceArrayView<T, I, A>::PointerType
DeviceArrayView<T, I, A>::data(IndexType index)
{
    ASSERT(index>=firstIndex());
    ASSERT(index<=lastIndex());
    return _data.shift(_stride*(index-_firstIndex));
}

template <typename T, typename I, typename A>
const typename DeviceArrayView<T, I, A>::Allocator &
DeviceArrayView<T, I, A>::allocator() const
{
    return _allocator;
}

template <typename T, typename I, typename A>
bool
DeviceArrayView<T, I, A>::resize(IndexType length,
                           IndexType firstIndex,
                           const ElementType &)
{
    ASSERT(length==_length);
    changeIndexBase(firstIndex);
    return false;
}

template <typename T, typename I, typename A>
template <typename ARRAY>
bool
DeviceArrayView<T, I, A>::resize(const ARRAY &rhs, const ElementType &value)
{
    return resize(rhs.length(), rhs.firstIndex(), value);
}

template <typename T, typename I, typename A>
bool
DeviceArrayView<T, I, A>::fill(const ElementType &value)
{
  
    fill_stride(length(), value, data(), stride());
    
    return true;
}

template <typename T, typename I, typename A>
void
DeviceArrayView<T, I, A>::changeIndexBase(IndexType firstIndex)
{
    _firstIndex = firstIndex;
}

template <typename T, typename I, typename A>
const typename DeviceArrayView<T, I, A>::ConstView
DeviceArrayView<T, I, A>::view(IndexType from, IndexType to,
                               IndexType stride, IndexType firstViewIndex) const
{
    const IndexType length = (to-from)/stride+1;

#   ifndef NDEBUG
    const PointerType         new_data = (length!=0) ? data(from) : PointerType(NULL, 0);
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        return ConstView(length,                // length
                         new_data,              // data
                         stride*_stride,        // stride
                         firstViewIndex,        // firstIndex in view
                         allocator());          // allocator
    }
#   else
    const PointerType         new_data = data(from);
#   endif

    ASSERT(firstIndex()<=from);
    ASSERT(lastIndex()>=to);
    ASSERT(from<=to);
    ASSERT(stride>0);
    return ConstView(length,                // length
                     new_data,              // data
                     stride*_stride,        // stride
                     firstViewIndex,        // firstIndex in view
                     allocator());          // allocator
}

template <typename T, typename I, typename A>
DeviceArrayView<T, I, A>
DeviceArrayView<T, I, A>::view(IndexType from, IndexType to,
                               IndexType stride, IndexType firstViewIndex)
{
    const IndexType length = (to-from)/stride+1;

#   ifndef NDEBUG
    PointerType         new_data = (length!=0) ? data(from) : PointerType(NULL, 0);
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        return DeviceArrayView(length,          // length
                               new_data,              // data
                               stride*_stride,        // stride
                               firstViewIndex,        // firstIndex in view
                               allocator());          // allocator
    } else {
        ASSERT(firstIndex()<=from);
        ASSERT(lastIndex()>=to);
        ASSERT(from<=to);
    }
#   else
    PointerType         new_data = data(from);
#   endif

    ASSERT(stride>0);
    return DeviceArrayView(length,          // length
                           new_data,              // data
                           stride*_stride,        // stride
                           firstViewIndex,        // firstIndex in view
                           allocator());          // allocator
}

#endif // defined(HAVE_CUBLAS) || defined(HAVE_CLBLAS)

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAYVIEW_TCC
