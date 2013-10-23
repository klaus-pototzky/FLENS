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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEPACKEDSTORAGE_CONSTDEVICEPACKEDSTORAGEVIEW_H
#define PLAYGROUND_FLENS_STORAGE_DEVICEPACKEDSTORAGE_CONSTDEVICEPACKEDSTORAGEVIEW_H 1

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
    class DevicePackedStorageView;

template <typename T,
          StorageOrder Order = ColMajor,
          typename I = IndexOptions<>,
          typename A = CustomAllocator<T, StorageType::DEFAULT_DEVICE_STORAGE_TYPE> >
class ConstDevicePackedStorageView
{
    public:
        typedef T                                      ElementType;
        typedef device_ptr<T, A::Type>                 PointerType;
        typedef const device_ptr<const T, A::Type>     ConstPointerType;
        typedef typename I::IndexType                  IndexType;
        typedef A                                      Allocator;

        static const StorageOrder       order = Order;
        static const IndexType          defaultIndexBase = I::defaultIndexBase;

        typedef ConstDevicePackedStorageView               ConstView;
        typedef DevicePackedStorageView<T, Order, I, A>    View;
        typedef DevicePackedStorage<T, Order, I, A>        NoView;

        typedef flens::ConstDeviceArrayView<T, I, A>       ConstArrayView;
        typedef flens::DeviceArrayView<T, I, A>            ArrayView;
        typedef flens::DeviceArray<T, I, A>                Array;

        ConstDevicePackedStorageView(IndexType dim,
                                     const ConstPointerType data,
                                     IndexType indexBase = I::defaultIndexBase,
                                     const Allocator &allocator = Allocator());

        template <typename ARRAY>
            ConstDevicePackedStorageView(IndexType dim,
                                         ARRAY &array,
                                         IndexType indexBase = I::defaultIndexBase,
                                         const Allocator &allocator = Allocator());

        ConstDevicePackedStorageView(const ConstDevicePackedStorageView &rhs);

        template <typename RHS>
            ConstDevicePackedStorageView(const RHS &rhs);

        ~ConstDevicePackedStorageView();

        //-- operators ---------------------------------------------------------

        const ElementType &
        operator()(StorageUpLo upLo, IndexType row, IndexType col) const;

        //-- methods -----------------------------------------------------------

        IndexType
        indexBase() const;

        IndexType
        numNonZeros() const;

        IndexType
        dim() const;

        const ConstPointerType
        data() const;
	
        const ConstPointerType
        data(StorageUpLo upLo, IndexType row, IndexType col) const;

        const Allocator &
        allocator() const;

        void
        changeIndexBase(IndexType indexBase);


    private:
        const ConstPointerType _data;
        Allocator              _allocator;
        IndexType              _dim;
        IndexType              _indexBase;
};

#endif

} // namespace flens

#endif // FLENS_STORAGE_PACKEDSTORAGE_CONSTPACKEDSTORAGEVIEW_H
