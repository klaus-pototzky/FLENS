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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_CONSTDEVICEFULLSTORAGEVIEW_TCC
#define PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_CONSTDEVICEFULLSTORAGEVIEW_TCC 1

#include <flens/storage/fullstorage/constfullstorageview.h>
#include <flens/typedefs.h>

namespace flens {
  
#ifdef HAVE_DEVICE_STORAGE

template <typename T, StorageOrder Order, typename I, typename A>
ConstDeviceFullStorageView<T, Order, I, A>::ConstDeviceFullStorageView(
                                        IndexType numRows, IndexType numCols,
                                        const PointerType data,
                                        IndexType leadingDimension,
                                        IndexType firstRow,
                                        IndexType firstCol,
                                        const Allocator &allocator)
    : _data(data), _allocator(allocator), _numRows(numRows), _numCols(numCols),
      _leadingDimension(leadingDimension), _firstRow(0), _firstCol(0)
{
    ASSERT(_numRows>=0);
    ASSERT(_numCols>=0);
    changeIndexBase(firstRow, firstCol);
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename ARRAY>
ConstDeviceFullStorageView<T, Order, I, A>::ConstDeviceFullStorageView(
                                       IndexType numRows, IndexType numCols,
                                       ARRAY &array, IndexType leadingDimension,
                                       IndexType firstRow, IndexType firstCol,
                                       const Allocator &allocator)
    : _data(array.data()),
      _allocator(allocator),
      _numRows(numRows), _numCols(numCols),
      _leadingDimension(leadingDimension),
      _firstRow(0), _firstCol(0)
{
    ASSERT(_numRows>=0);
    ASSERT(_numCols>=0);
    ASSERT(numRows*numCols<=array.length());

    changeIndexBase(firstRow, firstCol);
}

template <typename T, StorageOrder Order, typename I, typename A>
ConstDeviceFullStorageView<T, Order, I, A>::ConstDeviceFullStorageView(
                                                const ConstDeviceFullStorageView &rhs)
    : _data(rhs._data),
      _allocator(rhs._allocator),
      _numRows(rhs._numRows), _numCols(rhs._numCols),
      _leadingDimension(rhs._leadingDimension),
      _firstRow(rhs._firstRow), _firstCol(rhs._firstCol)
{
    ASSERT(order==rhs.order);
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename RHS>
ConstDeviceFullStorageView<T, Order, I, A>::ConstDeviceFullStorageView(const RHS &rhs)
    : _data(rhs.data()),
      _allocator(rhs.allocator()),
      _numRows(rhs.numRows()), _numCols(rhs.numCols()),
      _leadingDimension(rhs.leadingDimension()),
      _firstRow(0), _firstCol(0)
{
    ASSERT(order==rhs.order);
    changeIndexBase(rhs.firstRow(), rhs.firstCol());
}

template <typename T, StorageOrder Order, typename I, typename A>
ConstDeviceFullStorageView<T, Order, I, A>::~ConstDeviceFullStorageView()
{
}

//-- operators -----------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ElementType &
ConstDeviceFullStorageView<T, Order, I, A>::operator()(IndexType row,
                                                       IndexType col) const
{
    ASSERT(0);
}

//-- methods -------------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::firstRow() const
{
    return _firstRow;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::firstCol() const
{
    return _firstCol;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::lastRow() const
{
    return _firstRow+_numRows-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::lastCol() const
{
    return _firstCol+_numCols-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::numRows() const
{
    return _numRows;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::numCols() const
{
    return _numCols;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::leadingDimension() const
{
    return _leadingDimension;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::strideRow() const
{
    return (Order==ColMajor) ? 1
                             : leadingDimension();
}

template <typename T, StorageOrder Order, typename I, typename A>
typename ConstDeviceFullStorageView<T, Order, I, A>::IndexType
ConstDeviceFullStorageView<T, Order, I, A>::strideCol() const
{
    return (Order==ColMajor) ? leadingDimension()
                             : 1;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstPointerType
ConstDeviceFullStorageView<T, Order, I, A>::data() const
{
    if (Order==ColMajor) {
        return _data;
    }
    return _data;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstPointerType
ConstDeviceFullStorageView<T, Order, I, A>::data(IndexType row, IndexType col) const
{
#   ifndef NDEBUG
    if (numRows()>0 && numCols()>0) {
        ASSERT(row>=_firstRow);
        ASSERT(row<_firstRow+_numRows);
        ASSERT(col>=_firstCol);
        ASSERT(col<_firstCol+_numCols);
    } else {
        ASSERT(row==_firstRow);
        ASSERT(col==_firstCol);
        ASSERT(_data.get());
    }
#   endif

    IndexType _row = row - _firstRow;
    IndexType _col = col - _firstCol;
    
    if (Order==ColMajor) {
        return _data.shift(_col*_leadingDimension+_row);
    }
    return _data.shift(_row*_leadingDimension+_col);
}


template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::Allocator &
ConstDeviceFullStorageView<T, Order, I, A>::allocator() const
{
    return _allocator;
}

template <typename T, StorageOrder Order, typename I, typename A>
void
ConstDeviceFullStorageView<T, Order, I, A>::changeIndexBase(IndexType firstRow,
                                                            IndexType firstCol)
{
    _firstRow = firstRow;
    _firstCol = firstCol;
}

/// view of fullstorage scheme as an array
template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstArrayView
ConstDeviceFullStorageView<T, Order, I, A>::arrayView(IndexType firstViewIndex) const
{
#   ifndef NDEBUG
    if (order==ColMajor) {
        ASSERT(numRows()==leadingDimension());
    } else {
        ASSERT(numCols()==leadingDimension());
    }
#   endif

    return ConstArrayView(numCols()*numRows(),
                          data(),
                          IndexType(1),
                          firstViewIndex,
                          allocator());
}

// view of rectangular part
template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstView
ConstDeviceFullStorageView<T, Order, I, A>::view(IndexType fromRow, IndexType fromCol,
                                                 IndexType toRow, IndexType toCol,
                                                 IndexType strideRow, IndexType strideCol,
                                                 IndexType firstViewRow,
                                                 IndexType firstViewCol) const
{
    const IndexType numRows = (toRow-fromRow)/strideRow+1;
    const IndexType numCols = (toCol-fromCol)/strideCol+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if ((numRows==0) || (numCols==0)) {
        return ConstView(numRows, numCols, PointerType(NULL, 0), leadingDimension(),
                         firstViewRow, firstViewCol, allocator());
    }
    
#   endif

    ASSERT(fromRow>=firstRow());
    ASSERT(fromRow<=toRow);
    ASSERT(toRow<=lastRow());

    ASSERT(fromCol>=firstCol());
    ASSERT(fromCol<=toCol);
    ASSERT(toCol<=lastCol());
    
    ASSERT(order==ColMajor || strideCol==IndexType(1) );
    ASSERT(order==RowMajor || strideRow==IndexType(1) );
    
    return ConstView(numRows,                                 // # rows
                     numCols,                                 // # cols
                     data(fromRow, fromCol),                  // data
                     leadingDimension()*strideRow*strideCol,  // leading dimension
                     firstViewRow,                            // firstRow
                     firstViewCol,                            // firstCol
                     allocator());                            // allocator
}

// view of single row
template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstArrayView
ConstDeviceFullStorageView<T, Order, I, A>::viewRow(IndexType row,
                                                    IndexType firstViewIndex) const
{
#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (numCols()==0) {
        row = firstRow();
    }
#   endif

    ASSERT(row>=firstRow());
    ASSERT(row<=lastRow());

    return ConstArrayView(numCols(),
                          data(row, _firstCol),
                          strideCol(),
                          firstViewIndex,
                          allocator());
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstArrayView
ConstDeviceFullStorageView<T, Order, I, A>::viewRow(IndexType row,
                                                    IndexType firstCol,
                                                    IndexType lastCol,
                                                    IndexType stride,
                                                    IndexType firstViewIndex) const
{
    const IndexType length = (lastCol-firstCol)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        row =       this->firstRow();
        firstCol =  this->firstCol();
    }
#   endif

    ASSERT(row>=firstRow());
    ASSERT(row<=lastRow());

    return ConstArrayView(length,
                          data(row, firstCol),
                          strideCol()*stride,
                          firstViewIndex,
                          allocator());
}

// view of single columns
template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstArrayView
ConstDeviceFullStorageView<T, Order, I, A>::viewCol(IndexType col,
                                                    IndexType firstViewIndex) const
{
#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (numRows()==0) {
        col = firstCol();
    }
#   endif

    ASSERT(col>=firstCol());
    ASSERT(col<=lastCol());

    return ConstArrayView(numRows(),
                          data(_firstRow, col),
                          strideRow(),
                          firstViewIndex,
                          allocator());
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstArrayView
ConstDeviceFullStorageView<T, Order, I, A>::viewCol(IndexType firstRow,
                                                    IndexType lastRow,
                                                    IndexType stride,
                                                    IndexType col,
                                                    IndexType firstViewIndex) const
{
    const IndexType length = (lastRow-firstRow)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        firstRow =  this->firstRow();
        col =       this->firstCol();
    }
#   endif

    ASSERT(col>=firstCol());
    ASSERT(col<=lastCol());

    return ConstArrayView(length,
                          data(firstRow, col),
                          strideRow()*stride,
                          firstViewIndex,
                          allocator());
}

// view of d-th diagonal
template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstArrayView
ConstDeviceFullStorageView<T, Order, I, A>::viewDiag(IndexType d,
                                                     IndexType firstViewIndex) const
{
    IndexType _row = (d>0) ? 0 : -d;
    IndexType _col = (d>0) ? d :  0;

    IndexType row = firstRow() + _row;
    IndexType col = firstCol() + _col;

    return ConstArrayView(std::min(numRows()-_row, numCols()-_col),
                          data(row,col),
                          leadingDimension()+1,
                          firstViewIndex,
                          allocator());
}

// view of d-th anti-diagonal
template <typename T, StorageOrder Order, typename I, typename A>
const typename ConstDeviceFullStorageView<T, Order, I, A>::ConstArrayView
ConstDeviceFullStorageView<T, Order, I, A>::viewAntiDiag(IndexType d,
                                                         IndexType firstViewIndex) const
{
    IndexType _row = (d>0) ? 0 : -d;
    IndexType _col = (d>0) ? d :  0;

    IndexType row = firstRow() + _row;
    IndexType col = firstCol() + _col;

    return ConstArrayView(std::min(numRows()-_row, numCols()-_col),
                          data(row,lastCol()-col+1),
                          -leadingDimension()+1,
                          firstViewIndex,
                          allocator());
}

#endif // HAVE_DEVICE_STORAGE

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_CONSTDEVICEFULLSTORAGEVIEW_TCC
