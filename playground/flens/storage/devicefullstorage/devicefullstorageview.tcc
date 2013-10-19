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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGEVIEW_TCC
#define PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGEVIEW_TCC 1

#include <flens/auxiliary/auxiliary.h>
#include <playground/flens/storage/devicefullstorage/devicefullstorageview.h>
#include <flens/storage/fullstorage/trapezoidalfill.h>
#include <flens/typedefs.h>

namespace flens {

#ifdef HAVE_DEVICE_STORAGE
  
template <typename T, StorageOrder Order, typename I, typename A>
DeviceFullStorageView<T, Order, I, A>::DeviceFullStorageView(IndexType numRows,
                                                             IndexType numCols,
                                                             PointerType data,
                                                             IndexType leadingDimension,
                                                             IndexType firstRow,
                                                             IndexType firstCol,
                                                             const Allocator &allocator)
    : _data(data),
      _allocator(allocator),
      _numRows(numRows), _numCols(numCols),
      _leadingDimension(leadingDimension),
      _firstRow(0), _firstCol(0)
{
    ASSERT(_numRows>=0);
    ASSERT(_numCols>=0);

    changeIndexBase(firstRow, firstCol);
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename ARRAY>
DeviceFullStorageView<T, Order, I, A>::DeviceFullStorageView(IndexType numRows,
                                                             IndexType numCols,
                                                             ARRAY &array,
                                                             IndexType leadingDimension,
                                                             IndexType firstRow,
                                                             IndexType firstCol,
                                                             const Allocator &allocator)
    : _data(array.data()), 
      _allocator(allocator),
      _numRows(numRows), _numCols(numCols),
      _leadingDimension(leadingDimension),
      _firstRow(0), _firstCol(0)
{
    ASSERT(numRows*numCols<=array.length());
    ASSERT(_numRows>=0);
    ASSERT(_numCols>=0);

    changeIndexBase(firstRow, firstCol);
}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceFullStorageView<T, Order, I, A>::DeviceFullStorageView(const DeviceFullStorageView &rhs)
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
DeviceFullStorageView<T, Order, I, A>::DeviceFullStorageView(RHS &rhs)
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
DeviceFullStorageView<T, Order, I, A>::~DeviceFullStorageView()
{
}

//-- operators -----------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ElementType &
DeviceFullStorageView<T, Order, I, A>::operator()(IndexType row, IndexType col) const
{
#   ifndef NDEBUG
    if (numRows()>0 && numCols()>0) {
        ASSERT(row>=_firstRow);
        ASSERT(row<_firstRow+_numRows);
        ASSERT(col>=_firstCol);
        ASSERT(col<_firstCol+_numCols);
        ASSERT(_data);
    } else {
        ASSERT(row==_firstRow);
        ASSERT(col==_firstCol);
    }
#   endif

    if (Order==ColMajor) {
        return _data.shift(col*_leadingDimension+row);
    }
    return _data.shift(row*_leadingDimension+col);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::ElementType &
DeviceFullStorageView<T, Order, I, A>::operator()(IndexType row, IndexType col)
{
#   ifndef NDEBUG
    if (numRows()>0 && numCols()>0) {
        ASSERT(row>=_firstRow);
        ASSERT(row<_firstRow+_numRows);
        ASSERT(col>=_firstCol);
        ASSERT(col<_firstCol+_numCols);
        ASSERT(_data.get());
    } else {
        ASSERT(row==_firstRow);
        ASSERT(col==_firstCol);
    }
#   endif

    if (Order==ColMajor) {
        return _data.shift(col*_leadingDimension+row);
    }
    return _data.shift(row*_leadingDimension+col);
}

//-- Methods -------------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::firstRow() const
{
    return _firstRow;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::lastRow() const
{
    return _firstRow+_numRows-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::firstCol() const
{
    return _firstCol;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::lastCol() const
{
    return _firstCol+_numCols-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::numRows() const
{
    return _numRows;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::numCols() const
{
    return _numCols;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::leadingDimension() const
{
    return _leadingDimension;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::strideRow() const
{
    return (Order==ColMajor) ? 1
                             : leadingDimension();
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::IndexType
DeviceFullStorageView<T, Order, I, A>::strideCol() const
{
    return (Order==ColMajor) ? leadingDimension()
                             : 1;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstPointerType
DeviceFullStorageView<T, Order, I, A>::data() const
{
    if (Order==ColMajor) {
        return _data.shift(_firstCol*_leadingDimension+_firstRow);
    }
    return _data.shift(_firstRow*_leadingDimension+_firstCol);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::PointerType
DeviceFullStorageView<T, Order, I, A>::data()
{
    if (Order==ColMajor) {
        return _data.shift(_firstCol*_leadingDimension+_firstRow);
    }
    return _data.shift(_firstRow*_leadingDimension+_firstCol);
   
}


template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstPointerType
DeviceFullStorageView<T, Order, I, A>::data(IndexType row, IndexType col) const
{
#   ifndef NDEBUG
    if (numRows()>0 && numCols()>0) {
        ASSERT(row>=_firstRow);
        ASSERT(row<_firstRow+_numRows);
        ASSERT(col>=_firstCol);
        ASSERT(col<_firstCol+_numCols);
        ASSERT(_data.get());
    } else {
        ASSERT(row==_firstRow);
        ASSERT(col==_firstCol);
    }
#   endif

    if (Order==ColMajor) {
        return _data.shift(col*_leadingDimension+row);
    }
    return _data.shift(row*_leadingDimension+col);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::PointerType
DeviceFullStorageView<T, Order, I, A>::data(IndexType row, IndexType col)
{
 
#   ifndef NDEBUG
    if (numRows()>0 && numCols()>0) {
        ASSERT(row>=_firstRow);
        ASSERT(row<_firstRow+_numRows);
        ASSERT(col>=_firstCol);
        ASSERT(col<_firstCol+_numCols);
        ASSERT(_data.get());
    } else {
        ASSERT(row==_firstRow);
        ASSERT(col==_firstCol);
    }
#   endif

    if (Order==ColMajor) {
        return _data.shift(col*_leadingDimension+row);
    }
    return _data.shift(row*_leadingDimension+col);
    
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::Allocator &
DeviceFullStorageView<T, Order, I, A>::allocator() const
{
    return _allocator;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceFullStorageView<T, Order, I, A>::resize(IndexType _numRows,
                                        IndexType _numCols,
                                        IndexType firstRow,
                                        IndexType firstCol,
                                        const ElementType &)
{
    ASSERT(_numRows==numRows());
    ASSERT(_numCols==numCols());
    changeIndexBase(firstRow, firstCol);
    return false;
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename FS>
bool
DeviceFullStorageView<T, Order, I, A>::resize(const FS &rhs, const ElementType &value)
{
    return resize(rhs.numRows(), rhs.numCols(),
                  rhs.firstRow(), rhs.firstCol(),
                  value);
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceFullStorageView<T, Order, I, A>::fill(const ElementType &value)
{
    if (Order==RowMajor) {
        PointerType p = data();
        for (IndexType i=0; i<numRows(); ++i) {
            flens::fill_n(p, numCols(), value);
            p = p.shift(leadingDimension());
        }
        return true;
    }
    if (Order==ColMajor) {
        PointerType p = data();
        for (IndexType j=0; j<numCols(); ++j) {
            flens::fill_n(p, numRows(), value);
            p = p.shift(leadingDimension());
        }
        return true;
    }
    ASSERT(0);
    return false;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceFullStorageView<T, Order, I, A>::fill(StorageUpLo  upLo,
                                            const ElementType &value)
{
    trapezoidalFill(order, upLo, value,
                    numRows(), numCols(),
                    data(), leadingDimension());
    return true;
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceFullStorageView<T, Order, I, A>::changeIndexBase(IndexType firstRow,
                                                       IndexType firstCol)
{
#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if ((numRows()==0) || (numCols()==0)) {
        _firstRow = firstRow;
        _firstCol = firstCol;
        return;
    }
#   endif

    if (_data.get()) {
        if (Order==RowMajor) {
            _data = data().shift(- (firstRow*leadingDimension() + firstCol));
	}
        if (Order==ColMajor) {
            _data = data().shift(- (firstCol*leadingDimension() + firstRow));
        }
    }
    _firstRow = firstRow;
    _firstCol = firstCol;
}

// view of fullstorage scheme as an array
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstArrayView
DeviceFullStorageView<T, Order, I, A>::arrayView(IndexType firstViewIndex) const
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

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::ArrayView
DeviceFullStorageView<T, Order, I, A>::arrayView(IndexType firstViewIndex)
{
#   ifndef NDEBUG
    if (order==ColMajor) {
        ASSERT(numRows()==leadingDimension());
    } else {
        ASSERT(numCols()==leadingDimension());
    }
#   endif

    return ArrayView(numCols()*numRows(),
                     data(),
                     IndexType(1),
                     firstViewIndex,
                     allocator());
}


// view of rectangular part
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstView
DeviceFullStorageView<T, Order, I, A>::view(IndexType fromRow, IndexType fromCol,
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

// view of rectangular part
template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::View
DeviceFullStorageView<T, Order, I, A>::view(IndexType fromRow, IndexType fromCol,
                                            IndexType toRow, IndexType toCol,
                                            IndexType strideRow, IndexType strideCol,
                                            IndexType firstViewRow,
                                            IndexType firstViewCol) 
{
    const IndexType numRows = (toRow-fromRow)/strideRow+1;
    const IndexType numCols = (toCol-fromCol)/strideCol+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if ((numRows==0) || (numCols==0)) {
        return      View(numRows, numCols, PointerType(NULL, 0), leadingDimension(),
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
    
    return      View(numRows,                                 // # rows
                     numCols,                                 // # cols
                     data(fromRow, fromCol),                  // data
                     leadingDimension()*strideRow*strideCol,  // leading dimension
                     firstViewRow,                            // firstRow
                     firstViewCol,                            // firstCol
                     allocator());                            // allocator
}

// view of single row
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstArrayView
DeviceFullStorageView<T, Order, I, A>::viewRow(IndexType row,
                                               IndexType firstViewIndex) const
{
#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (numCols()==0) {
        return ConstArrayView(numCols(), PointerType(NULL, 0), strideCol(),
                              firstViewIndex, allocator());
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
typename DeviceFullStorageView<T, Order, I, A>::ArrayView
DeviceFullStorageView<T, Order, I, A>::viewRow(IndexType row,
                                               IndexType firstViewIndex)
{
#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (numCols()==0) {
        return ArrayView(numCols(), PointerType(NULL, 0), strideCol(),
                         firstViewIndex, allocator());
    }
#   endif

    ASSERT(row>=firstRow());
    ASSERT(row<=lastRow());

    return ArrayView(numCols(),
                     data(row, _firstCol),
                     strideCol(),
                     firstViewIndex,
                     allocator());
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstArrayView
DeviceFullStorageView<T, Order, I, A>::viewRow(IndexType row,
                                               IndexType firstCol, IndexType lastCol,
                                               IndexType stride, IndexType firstViewIndex) const
{
    const IndexType length = (lastCol-firstCol)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        return ConstArrayView(length, PointerType(NULL, 0), strideCol()*stride,
                              firstViewIndex, allocator());
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

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::ArrayView
DeviceFullStorageView<T, Order, I, A>::viewRow(IndexType row,
                                               IndexType firstCol, IndexType lastCol,
                                               IndexType stride, IndexType firstViewIndex)
{
    const IndexType length = (lastCol-firstCol)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        return ArrayView(length, PointerType(NULL, 0), strideCol()*stride,
                         firstViewIndex, allocator());
    }
#   endif

    ASSERT(row>=firstRow());
    ASSERT(row<=lastRow());

    return ArrayView(length,
                     data(row, firstCol),
                     strideCol()*stride,
                     firstViewIndex,
                     allocator());
}

// view of single column
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstArrayView
DeviceFullStorageView<T, Order, I, A>::viewCol(IndexType col,
                                               IndexType firstViewIndex) const
{
#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (numRows()==0) {
        return ConstArrayView(numRows(), PointerType(NULL, 0), strideRow(),
                              firstViewIndex, allocator());
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
typename DeviceFullStorageView<T, Order, I, A>::ArrayView
DeviceFullStorageView<T, Order, I, A>::viewCol(IndexType col,
                                               IndexType firstViewIndex)
{
#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (numRows()==0) {
        return ArrayView(numRows(), PointerType(NULL, 0), strideRow(),
                         firstViewIndex, allocator());
    }
#   endif

    ASSERT(col>=firstCol());
    ASSERT(col<=lastCol());

    return ArrayView(numRows(),
                     data(_firstRow, col),
                     strideRow(),
                     firstViewIndex,
                     allocator());
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstArrayView
DeviceFullStorageView<T, Order, I, A>::viewCol(IndexType firstRow, IndexType lastRow,
                                               IndexType stride, IndexType col,
                                               IndexType firstViewIndex) const
{
    const IndexType length = (lastRow-firstRow)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        return ConstArrayView(length, PointerType(NULL, 0), strideRow()*stride,
                              firstViewIndex, allocator());
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

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::ArrayView
DeviceFullStorageView<T, Order, I, A>::viewCol(IndexType firstRow, IndexType lastRow,
                                               IndexType stride, IndexType col,
                                               IndexType firstViewIndex)
{
    const IndexType length = (lastRow-firstRow)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        return ArrayView(length, PointerType(NULL, 0), strideRow()*stride,
                         firstViewIndex, allocator());
    }
#   endif

    ASSERT(col>=firstCol());
    ASSERT(col<=lastCol());

    return ArrayView(length,
                     data(firstRow, col),
                     strideRow()*stride,
                     firstViewIndex,
                     allocator());
}

// view of d-th diagonal
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstArrayView
DeviceFullStorageView<T, Order, I, A>::viewDiag(IndexType d,
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

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::ArrayView
DeviceFullStorageView<T, Order, I, A>::viewDiag(IndexType d,
                                                IndexType firstViewIndex)
{  
    IndexType _row = (d>0) ? 0 : -d;
    IndexType _col = (d>0) ? d :  0;

    IndexType row = firstRow() + _row;
    IndexType col = firstCol() + _col;

    return ArrayView(std::min(numRows()-_row,numCols()-_col),
                     data(row,col),
                     leadingDimension()+1,
                     firstViewIndex,
                     allocator());
}

// view of d-th anti-diagonal
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorageView<T, Order, I, A>::ConstArrayView
DeviceFullStorageView<T, Order, I, A>::viewAntiDiag(IndexType d,
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

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorageView<T, Order, I, A>::ArrayView
DeviceFullStorageView<T, Order, I, A>::viewAntiDiag(IndexType d,
                                                    IndexType firstViewIndex)
{
    IndexType _row = (d>0) ? 0 : -d;
    IndexType _col = (d>0) ? d :  0;

    IndexType row = firstRow() + _row;
    IndexType col = firstCol() + _col;

    return ArrayView(std::min(numRows()-_row,numCols()-_col),
                     data(row,lastCol()-col+1),
                     -leadingDimension()+1,
                     firstViewIndex,
                     allocator());
}

#endif // HAVE_DEVICE_STORAGE

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGEVIEW_TCC
