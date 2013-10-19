/*
 *   Copyright (c) 2007, 2013, Michael Lehn, Klaus Pototzky
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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGE_H
#define PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGE_H 1

#include <cxxblas/typedefs.h>
#include <flens/auxiliary/auxiliary.h>
#include <flens/storage/indexoptions.h>
#include <flens/typedefs.h>
#include <playground/flens/auxiliary/devicePtr.h>
#include <playground/flens/auxiliary/customAllocator.h>

namespace flens {
  
#ifdef HAVE_DEVICE_STORAGE

template <typename T, typename I, typename A>
    class DeviceArray;

template <typename T, typename I, typename A>
    class DeviceArrayView;

template <typename T, typename I, typename A>
    class ConstDeviceArrayView;

template <typename T, StorageOrder Order, typename I, typename A>
    class DeviceFullStorageView;

template <typename T, StorageOrder Order, typename I, typename A>
    class ConstDeviceFullStorageView;

template <typename T,
          StorageOrder Order = ColMajor,
          typename I = IndexOptions<>,
          typename A = CustomAllocator<T, StorageType::DEFAULT_DEVICE_STORAGE_TYPE> >
class DeviceFullStorage
{
    public:
        typedef T                                     ElementType;
        typedef device_ptr<T, A::Type>                PointerType;
        typedef const device_ptr<T, A::Type>          ConstPointerType;
        typedef typename I::IndexType                 IndexType;
        typedef A                                     Allocator;

        static const StorageOrder                     order = Order;
        static const IndexType                        defaultIndexBase
                                                          = I::defaultIndexBase;

        typedef ConstDeviceFullStorageView<T, Order, I, A>  ConstView;
        typedef DeviceFullStorageView<T, Order, I, A>       View;
        typedef DeviceFullStorage                           NoView;

        typedef flens::ConstDeviceArrayView<T, I, A>        ConstArrayView;
        typedef flens::DeviceArrayView<T, I, A>             ArrayView;
        typedef flens::DeviceArray<T, I, A>                 Array;

        DeviceFullStorage();

        DeviceFullStorage(IndexType numRows, IndexType numCols,
                          IndexType firstRow = I::defaultIndexBase,
                          IndexType firstCol = I::defaultIndexBase,
                          const ElementType &value = ElementType(),
                          const Allocator &allocator = Allocator());

        DeviceFullStorage(const DeviceFullStorage &rhs);

        template <typename RHS>
            DeviceFullStorage(const RHS &rhs);

        ~DeviceFullStorage();

        //-- operators ---------------------------------------------------------

        const ElementType &
        operator()(IndexType row, IndexType col) const;

        ElementType &
        operator()(IndexType row, IndexType col);

        //-- methods -----------------------------------------------------------

        IndexType
        firstRow() const;

        IndexType
        lastRow() const;

        IndexType
        firstCol() const;

        IndexType
        lastCol() const;

        IndexType
        numRows() const;

        IndexType
        numCols() const;

        IndexType
        leadingDimension() const;

        IndexType
        strideRow() const;

        IndexType
        strideCol() const;

        const ConstPointerType
        data() const;

        PointerType
        data();

        const ConstPointerType
        data(IndexType row, IndexType col) const;

        PointerType
        data(IndexType row, IndexType col);

        const Allocator &
        allocator() const;

        bool
        resize(IndexType numRows, IndexType numCols,
               IndexType firstRow = I::defaultIndexBase,
               IndexType firstCol = I::defaultIndexBase,
               const ElementType &value = ElementType());

        bool
        resize(const Range<IndexType> &rows,
               const Range<IndexType> &cols,
               const ElementType &value = ElementType());

        template <typename FS>
            bool
            resize(const FS &rhs, const ElementType &value = ElementType());

        bool
        fill(const ElementType &value = ElementType(0));

        bool
        fill(StorageUpLo upLo, const ElementType &value = ElementType(0));

        void
        changeIndexBase(IndexType firstRow, IndexType firstCol);

        // view of fullstorage scheme as an array
        const ConstArrayView
        arrayView(IndexType firstViewIndex = I::defaultIndexBase) const;

        ArrayView
        arrayView(IndexType firstViewIndex = I::defaultIndexBase);

        // view of rectangular part
        const ConstView
        view(IndexType fromRow, IndexType fromCol,
             IndexType toRow, IndexType toCol,
             IndexType strideRow = IndexType(1),
             IndexType strideCol = IndexType(1),
             IndexType firstViewRow = I::defaultIndexBase,
             IndexType firstViewCol = I::defaultIndexBase) const;
    
        View
        view(IndexType fromRow, IndexType fromCol,
             IndexType toRow, IndexType toCol,
             IndexType strideRow = IndexType(1),
             IndexType strideCol = IndexType(1),
             IndexType firstViewRow = I::defaultIndexBase,
             IndexType firstViewCol = I::defaultIndexBase);

        // view of single row
        const ConstArrayView
        viewRow(IndexType row,
                IndexType firstViewIndex = I::defaultIndexBase) const;

        ArrayView
        viewRow(IndexType row,
                IndexType firstViewIndex = I::defaultIndexBase);

        const ConstArrayView
        viewRow(IndexType row,
                IndexType firstCol, IndexType lastCol,
                IndexType stride,
                IndexType firstViewIndex = I::defaultIndexBase) const;

        ArrayView
        viewRow(IndexType row,
                IndexType firstCol, IndexType lastCol,
                IndexType stride,
                IndexType firstViewIndex = I::defaultIndexBase);

        // view of single column
        const ConstArrayView
        viewCol(IndexType col,
                IndexType firstViewIndex = I::defaultIndexBase) const;

        ArrayView
        viewCol(IndexType col,
                IndexType firstViewIndex = I::defaultIndexBase);

        const ConstArrayView
        viewCol(IndexType firstRow, IndexType lastRow,
                IndexType stride, IndexType col,
                IndexType firstViewIndex = I::defaultIndexBase) const;

        ArrayView
        viewCol(IndexType firstRow, IndexType lastRow,
                IndexType stride, IndexType col,
                IndexType firstViewIndex = I::defaultIndexBase);

        // view of d-th diagonal
        const ConstArrayView
        viewDiag(IndexType d,
                 IndexType firstViewIndex = I::defaultIndexBase) const;

        ArrayView
        viewDiag(IndexType d,
                 IndexType firstViewIndex = I::defaultIndexBase);

        // view of d-th anti-diagonal
        const ConstArrayView
        viewAntiDiag(IndexType d,
                     IndexType firstViewIndex = I::defaultIndexBase) const;

        ArrayView
        viewAntiDiag(IndexType d,
                     IndexType firstViewIndex = I::defaultIndexBase);

    private:

        void
        _setIndexBase(IndexType firstRow, IndexType firstCol);

        void
        _raw_allocate();

        void
        _allocate(const ElementType &value = ElementType());

        void
        _release();

        PointerType  _data;
        Allocator    _allocator;
        IndexType    _numRows, _numCols;
        IndexType    _firstRow, _firstCol;
};

#endif // HAVE_DEVICE_STORAGE

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGE_H
