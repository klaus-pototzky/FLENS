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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEPACKEDSTORAGE_DEVICEPACKEDSTORAGEVIEW_H
#define PLAYGROUND_FLENS_STORAGE_DEVICEPACKEDSTORAGE_DEVICEPACKEDSTORAGEVIEW_H

#include <cxxblas/typedefs.h>
#include <flens/typedefs.h>
#include <flens/storage/indexoptions.h>

namespace flens {

#ifdef HAVE_DEVICE_STORAGE

template <typename T, typename I, typename A>
    class DeviceArray;

template <typename T, typename I, typename A>
    class DeviceArrayView;

template <typename T, typename I, typename A>
    class ConstDeviceArrayView;

template <typename T, StorageOrder Order, typename I, typename A>
    class DevicePackedStorage;

template <typename T, StorageOrder Order, typename I, typename A>
    class ConstDevicePackedStorageView;

template <typename T,
          StorageOrder Order = ColMajor,
          typename I = IndexOptions<>,
          typename A = CustomAllocator<T, StorageType::DEFAULT_DEVICE_STORAGE_TYPE> >
class DevicePackedStorageView
{
    public:
        typedef T                                      ElementType;
        typedef device_ptr<T, A::Type>                 PointerType;
        typedef const device_ptr<const T, A::Type>     ConstPointerType;
        typedef typename I::IndexType                  IndexType;
        typedef A                                      Allocator;

        static const StorageOrder       order = Order;
        static const IndexType          defaultIndexBase = I::defaultIndexBase;

        typedef ConstDevicePackedStorageView<T, Order, I, A>  ConstView;
        typedef DevicePackedStorageView                       View;
        typedef DevicePackedStorage<T, Order, I, A>           NoView;

        typedef flens::ConstArrayView<T, I, A>          ConstArrayView;
        typedef flens::ArrayView<T, I, A>               ArrayView;
        typedef flens::Array<T, I, A>                   Array;

        DevicePackedStorageView(IndexType dim,
                                PointerType data,
                                IndexType indexBase = I::defaultIndexBase,
                                const Allocator &allocator = Allocator());

        template <typename ARRAY>
            DevicePackedStorageView(IndexType dim,
                              ARRAY &array,
                              IndexType indexBase = I::defaultIndexBase,
                              const Allocator &allocator = Allocator());

        DevicePackedStorageView(const DevicePackedStorageView &rhs);

        template <typename RHS>
            DevicePackedStorageView(RHS &rhs);

        ~DevicePackedStorageView();

        //-- operators ---------------------------------------------------------

        const ElementType &
        operator()(StorageUpLo upLo, IndexType row, IndexType col) const;

        ElementType &
        operator()(StorageUpLo upLo, IndexType row, IndexType col);

        //-- methods -----------------------------------------------------------

        IndexType
        indexBase() const;

        IndexType
        numNonZeros() const;

        IndexType
        dim() const;

        const ConstPointerType
        data() const;

        PointerType
        data();
	
        const ConstPointerType
        data(StorageUpLo upLo, IndexType row, IndexType col) const;

        PointerType
        data(StorageUpLo upLo, IndexType row, IndexType col);
	
        const Allocator &
        allocator() const;

        bool
        resize(IndexType dim,
               IndexType indexBase = I::defaultIndexBase,
               const ElementType &value = ElementType());

        bool
        fill(const ElementType &value = ElementType(0));

        void
        changeIndexBase(IndexType indexBase);

    private:
        PointerType  _data;
        Allocator    _allocator;
        IndexType    _dim;
        IndexType    _indexBase;
};

#endif

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_DEVICEPACKEDSTORAGE_DEVICEPACKEDSTORAGEVIEW_H
