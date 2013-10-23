/*
 *   Copyright (c) 2012, Michael Lehn, Klaus Pototzky
view *
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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEPACKEDSTORAGE_DEVICEPACKEDSTORAGE_TCC
#define PLAYGROUND_FLENS_STORAGE_DEVICEPACKEDSTORAGE_DEVICEPACKEDSTORAGE_TCC

#include <flens/auxiliary/auxiliary.h>
#include <playground/flens/storage/devicepackedstorage/devicepackedstorage.h>
#include <flens/typedefs.h>

namespace flens {

#ifdef HAVE_DEVICE_STORAGE

//-- Constructors --------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
DevicePackedStorage<T, Order, I, A>::DevicePackedStorage(IndexType dim,
                                                         IndexType indexBase,
                                                         const ElementType &value,
                                                         const Allocator &allocator)
    : _data(), _allocator(allocator),
      _dim(dim),
      _indexBase(indexBase)
{
    ASSERT(_dim>=0);

    _allocate(value);
}

template <typename T, StorageOrder Order, typename I, typename A>
DevicePackedStorage<T, Order, I, A>::DevicePackedStorage(const DevicePackedStorage &rhs)
    : _data(), _allocator(rhs.allocator()),
      _dim(rhs.dim()),
      _indexBase(rhs.indexBase())
{
    _allocate(ElementType());

    cxxblas::copy(rhs.dim(), rhs.data(), 1, data(), 1);
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename RHS>
DevicePackedStorage<T, Order, I, A>::DevicePackedStorage(const RHS &rhs)
    : _data(), _allocator(rhs.allocator()),
      _dim(rhs.dim()),
      _indexBase(rhs.indexBase())
{
    _allocate(ElementType());

    cxxblas::copy(rhs.dim(), rhs.data(), 1, data(), 1);

}

template <typename T, StorageOrder Order, typename I, typename A>
DevicePackedStorage<T, Order, I, A>::~DevicePackedStorage()
{
    _release();
}

//-- operators -----------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
const typename DevicePackedStorage<T, Order, I, A>::ElementType &
DevicePackedStorage<T, Order, I, A>::operator()(StorageUpLo  upLo,
                                                IndexType    row,
                                                IndexType    col) const
{
    ASSERT(row>=indexBase());
    ASSERT(col>=indexBase());
    ASSERT(row<=indexBase()+dim()-1);
    ASSERT(col<=indexBase()+dim()-1);

    if (upLo==Lower) {
        ASSERT(row>=col);
    } else {
        ASSERT(col>=row);
    }

    const IndexType i = row - _indexBase;
    const IndexType j = col - _indexBase;
    const IndexType n = _dim;

    if ((order==RowMajor) && (upLo==Upper)) {
        return _data[j+i*(2*n-i-1)/2];
    }
    if ((order==RowMajor) && (upLo==Lower)) {
        return _data[j+i*(i+1)/2];
    }
    if ((order==ColMajor) && (upLo==Upper)) {
        return _data[i+j*(j+1)/2];
    }
    return _data[i+j*(2*n-j-1)/2];
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DevicePackedStorage<T, Order, I, A>::ElementType &
DevicePackedStorage<T, Order, I, A>::operator()(StorageUpLo  upLo,
                                                IndexType    row,
                                                IndexType    col)
{
    ASSERT(row>=indexBase());
    ASSERT(col>=indexBase());
    ASSERT(row<=indexBase()+dim()-1);
    ASSERT(col<=indexBase()+dim()-1);

    if (upLo==Lower) {
        ASSERT(row>=col);
    } else {
        ASSERT(col>=row);
    }

    const IndexType i = row - _indexBase;
    const IndexType j = col - _indexBase;
    const IndexType n = _dim;

    if ((order==RowMajor) && (upLo==Upper)) {
        return _data[j+i*(2*n-i-1)/2];
    }
    if ((order==RowMajor) && (upLo==Lower)) {
        return _data[j+i*(i+1)/2];
    }
    if ((order==ColMajor) && (upLo==Upper)) {
        return _data[i+j*(j+1)/2];
    }
    return _data[i+j*(2*n-j-1)/2];
}

//-- Methods -------------------------------------------------------------------
template <typename T, StorageOrder Order, typename I, typename A>
typename DevicePackedStorage<T, Order, I, A>::IndexType
DevicePackedStorage<T, Order, I, A>::indexBase() const
{
    return _indexBase;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DevicePackedStorage<T, Order, I, A>::IndexType
DevicePackedStorage<T, Order, I, A>::numNonZeros() const
{
    return (_dim+1)*_dim/IndexType(2);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DevicePackedStorage<T, Order, I, A>::IndexType
DevicePackedStorage<T, Order, I, A>::dim() const
{
    return _dim;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DevicePackedStorage<T, Order, I, A>::ConstPointerType
DevicePackedStorage<T, Order, I, A>::data() const
{
    return _data;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DevicePackedStorage<T, Order, I, A>::PointerType
DevicePackedStorage<T, Order, I, A>::data()
{
    return _data;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DevicePackedStorage<T, Order, I, A>::ConstPointerType
DevicePackedStorage<T, Order, I, A>::data(StorageUpLo  upLo, IndexType row, IndexType col) const
{
    ASSERT(row>=indexBase());
    ASSERT(col>=indexBase());
    ASSERT(row<=indexBase()+dim()-1);
    ASSERT(col<=indexBase()+dim()-1);

    if (upLo==Lower) {
        ASSERT(row>=col);
    } else {
        ASSERT(col>=row);
    }

    const IndexType i = row - _indexBase;
    const IndexType j = col - _indexBase;
    const IndexType n = _dim;

    if ((order==RowMajor) && (upLo==Upper)) {
        return _data.shift(j+i*(2*n-i-1)/2);
    }
    if ((order==RowMajor) && (upLo==Lower)) {
        return _data.shift(j+i*(i+1)/2);
    }
    if ((order==ColMajor) && (upLo==Upper)) {
        return _data.shift(i+j*(j+1)/2);
    }
    return _data.shift(i+j*(2*n-j-1)/2);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DevicePackedStorage<T, Order, I, A>::PointerType
DevicePackedStorage<T, Order, I, A>::data(StorageUpLo  upLo, IndexType row, IndexType col)
{
    ASSERT(row>=indexBase());
    ASSERT(col>=indexBase());
    ASSERT(row<=indexBase()+dim()-1);
    ASSERT(col<=indexBase()+dim()-1);

    if (upLo==Lower) {
        ASSERT(row>=col);
    } else {
        ASSERT(col>=row);
    }

    const IndexType i = row - _indexBase;
    const IndexType j = col - _indexBase;
    const IndexType n = _dim;

    if ((order==RowMajor) && (upLo==Upper)) {
        return _data.shift(j+i*(2*n-i-1)/2);
    }
    if ((order==RowMajor) && (upLo==Lower)) {
        return _data.shift(j+i*(i+1)/2);
    }
    if ((order==ColMajor) && (upLo==Upper)) {
        return _data.shift(i+j*(j+1)/2);
    }
    return _data.shift(i+j*(2*n-j-1)/2);
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DevicePackedStorage<T, Order, I, A>::Allocator &
DevicePackedStorage<T, Order, I, A>::allocator() const
{
    return _allocator;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DevicePackedStorage<T, Order, I, A>::resize(IndexType dim,
                                            IndexType indexBase,
                                            const ElementType &value)
{
    if (_dim!=dim) {
        _release();
        _dim = dim;
        _indexBase = indexBase;
        _allocate(value);
        return true;
    }
    changeIndexBase(indexBase);
    return false;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DevicePackedStorage<T, Order, I, A>::fill(const ElementType &value)
{
    ASSERT(_data.get());
    flens::fill_n(data(), ((dim()+1)*dim())/2, value);
    return true;
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DevicePackedStorage<T, Order, I, A>::changeIndexBase(IndexType indexBase)
{
    _indexBase = indexBase;
}

//-- Private Methods -----------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
void
DevicePackedStorage<T, Order, I, A>::_raw_allocate()
{
    ASSERT(!_data.get());
    ASSERT(_dim>0);

    _data = _allocator.allocate(numNonZeros());
    ASSERT(_data.get());

#ifndef NDEBUG
    PointerType p = _data;
#endif

    changeIndexBase(_indexBase);

#ifndef NDEBUG
    ASSERT(p==data());
#endif
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DevicePackedStorage<T, Order, I, A>::_allocate(const ElementType &value)
{
    const IndexType numElements = numNonZeros();

    if (numElements==0) {
        return;
    }

    _raw_allocate();
    fill(value);
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DevicePackedStorage<T, Order, I, A>::_release()
{
    if (_data.get()) {
        const IndexType numElements = numNonZeros();
        _allocator.deallocate(data(), numElements);
        _data = PointerType(NULL, 0);
    }
    ASSERT(_data.get()==0);
}

#endif

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_DEVICEPACKEDSTORAGE_DEVICEPACKEDSTORAGE_TCC
