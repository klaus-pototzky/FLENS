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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGE_TCC
#define PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGE_TCC 1

#include <cxxblas/level1extensions/gecopy.h>
#include <flens/auxiliary/auxiliary.h>
#include <playground/flens/storage/devicefullstorage/devicefullstorage.h>
#include <flens/storage/fullstorage/trapezoidalfill.h>
#include <flens/typedefs.h>

namespace flens {
  
#ifdef HAVE_DEVICE_STORAGE

//= Constructors

template <typename T, StorageOrder Order, typename I, typename A>
DeviceFullStorage<T, Order, I, A>::DeviceFullStorage()
    :  _data(),
       _numRows(0), _numCols(0),
       _firstRow(I::defaultIndexBase), _firstCol(I::defaultIndexBase)
{
}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceFullStorage<T, Order, I, A>::DeviceFullStorage(IndexType numRows, IndexType numCols,
                                                     IndexType firstRow, IndexType firstCol,
                                                     const ElementType &value,
                                                     const Allocator &allocator)
    : _data(), _allocator(allocator),
      _numRows(numRows), _numCols(numCols),
      _firstRow(firstRow), _firstCol(firstCol)
{
    ASSERT(_numRows>=0);
    ASSERT(_numCols>=0);

    _allocate(value);
}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceFullStorage<T, Order, I, A>::DeviceFullStorage(const DeviceFullStorage &rhs)
    : _data(), _allocator(rhs.allocator()),
      _numRows(rhs.numRows()), _numCols(rhs.numCols()),
      _firstRow(rhs.firstRow()), _firstCol(rhs.firstCol())
{
    _allocate(ElementType());
    Transpose trans = (Order==rhs.order) ? NoTrans : Trans;
    cxxblas::gecopy(Order,
                    trans, _numRows, _numCols,
                    rhs.data(), rhs.leadingDimension(),
                    data(), leadingDimension());
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename RHS>
DeviceFullStorage<T, Order, I, A>::DeviceFullStorage(const RHS &rhs)
    : _data(), _allocator(rhs.allocator()),
      _numRows(rhs.numRows()), _numCols(rhs.numCols()),
      _firstRow(rhs.firstRow()), _firstCol(rhs.firstCol())
{
    _allocate(ElementType());
    Transpose trans = (Order==rhs.order) ? NoTrans : Trans;
    cxxblas::gecopy(Order,
                    trans, _numRows, _numCols,
                    rhs.data(), rhs.leadingDimension(),
                    data(), leadingDimension());
}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceFullStorage<T, Order, I, A>::~DeviceFullStorage()
{
    _release();
}

//-- operators -----------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorage<T, Order, I, A>::ElementType &
DeviceFullStorage<T, Order, I, A>::operator()(IndexType row, IndexType col) const
{
    ASSERT(0);
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
        return _data.get()[col*_numRows+row];
    }
    return _data.get()[row*_numCols+col];
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::ElementType &
DeviceFullStorage<T, Order, I, A>::operator()(IndexType row, IndexType col)
{
    ASSERT(0);
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
        return _data.get()[col*_numRows+row];
    }
    return _data.get()[row*_numCols+col];
}

//-- Methods -------------------------------------------------------------------
template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::firstRow() const
{
    return _firstRow;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::lastRow() const
{
    return _firstRow+_numRows-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::firstCol() const
{
    return _firstCol;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::lastCol() const
{
    return _firstCol+_numCols-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::numRows() const
{
    return _numRows;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::numCols() const
{
    return _numCols;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::leadingDimension() const
{
    return (Order==ColMajor) ? std::max(_numRows, IndexType(1))
                             : std::max(_numCols, IndexType(1));
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::strideRow() const
{
    return (Order==ColMajor) ? 1
                             : leadingDimension();
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::IndexType
DeviceFullStorage<T, Order, I, A>::strideCol() const
{
    return (Order==ColMajor) ? leadingDimension()
                             : 1;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorage<T, Order, I, A>::ConstPointerType
DeviceFullStorage<T, Order, I, A>::data() const
{
    if (Order==ColMajor) {
        return _data.shift(_firstCol*_numRows+_firstRow);
    }
    return _data.shift(_firstRow*_numCols+_firstCol);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::PointerType
DeviceFullStorage<T, Order, I, A>::data()
{
    if (Order==ColMajor) {
        return _data.shift(_firstCol*_numRows+_firstRow);
    }
    return _data.shift(_firstRow*_numCols+_firstCol);
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorage<T, Order, I, A>::ConstPointerType
DeviceFullStorage<T, Order, I, A>::data(IndexType row, IndexType col) const
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
        return _data.shift(col*_numRows+row);
    }
    return _data.shift(row*_numCols+col);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceFullStorage<T, Order, I, A>::PointerType
DeviceFullStorage<T, Order, I, A>::data(IndexType row, IndexType col)
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
        return _data.shift(col*_numRows+row);
    }
    return _data.shift(row*_numCols+col);
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorage<T, Order, I, A>::Allocator &
DeviceFullStorage<T, Order, I, A>::allocator() const
{
    return _allocator;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceFullStorage<T, Order, I, A>::resize(IndexType numRows, IndexType numCols,
                                          IndexType firstRow, IndexType firstCol,
                                          const ElementType &value)
{
    if ((_numRows!=numRows) || (_numCols!=numCols)) {
        _release();
        _numRows = numRows;
        _numCols = numCols;
        _firstRow = firstRow;
        _firstCol = firstCol;
        _allocate(value);
        return true;
    }
    changeIndexBase(firstRow, firstCol);
    return false;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceFullStorage<T, Order, I, A>::resize(const Range<IndexType> &rows,
                                          const Range<IndexType> &cols,
                                          const ElementType &value)
{
    if ((_numRows!=rows.length()) || (_numCols!=cols.length())) {
        _release();
        _numRows = rows.length();
        _numCols = cols.length();
        _firstRow = rows.firstIndex();
        _firstCol = cols.firstIndex();
        _allocate(value);
        return true;
    }
    changeIndexBase(rows.firstIndex(), cols.firstIndex());
    return false;
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename FS>
bool
DeviceFullStorage<T, Order, I, A>::resize(const FS &rhs, const ElementType &value)
{
    return resize(rhs.numRows(), rhs.numCols(),
                  rhs.firstRow(), rhs.firstCol(),
                  value);
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceFullStorage<T, Order, I, A>::fill(const ElementType &value)
{
    ASSERT(_data.get());
    flens::fill_n(data(), numRows()*numCols(), value);
    return true;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceFullStorage<T, Order, I, A>::fill(StorageUpLo  upLo,
                                        const ElementType &value)
{
    ASSERT(_data.get());

    trapezoidalFill(order, upLo, value,
                    numRows(), numCols(),
                    data(), leadingDimension());
    return true;
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceFullStorage<T, Order, I, A>::changeIndexBase(IndexType firstRow,
                                                   IndexType firstCol)
{    
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
const typename DeviceFullStorage<T, Order, I, A>::ConstArrayView
DeviceFullStorage<T, Order, I, A>::arrayView(IndexType firstViewIndex) const
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
typename DeviceFullStorage<T, Order, I, A>::ArrayView
DeviceFullStorage<T, Order, I, A>::arrayView(IndexType firstViewIndex)
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
const typename DeviceFullStorage<T, Order, I, A>::ConstView
DeviceFullStorage<T, Order, I, A>::view(IndexType fromRow, IndexType fromCol,
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
typename DeviceFullStorage<T, Order, I, A>::View
DeviceFullStorage<T, Order, I, A>::view(IndexType fromRow, IndexType fromCol,
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
const typename DeviceFullStorage<T, Order, I, A>::ConstArrayView
DeviceFullStorage<T, Order, I, A>::viewRow(IndexType row,
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
typename DeviceFullStorage<T, Order, I, A>::ArrayView
DeviceFullStorage<T, Order, I, A>::viewRow(IndexType row,
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
const typename DeviceFullStorage<T, Order, I, A>::ConstArrayView
DeviceFullStorage<T, Order, I, A>::viewRow(IndexType row,
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
typename DeviceFullStorage<T, Order, I, A>::ArrayView
DeviceFullStorage<T, Order, I, A>::viewRow(IndexType row,
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
const typename DeviceFullStorage<T, Order, I, A>::ConstArrayView
DeviceFullStorage<T, Order, I, A>::viewCol(IndexType col,
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
typename DeviceFullStorage<T, Order, I, A>::ArrayView
DeviceFullStorage<T, Order, I, A>::viewCol(IndexType col,
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
const typename DeviceFullStorage<T, Order, I, A>::ConstArrayView
DeviceFullStorage<T, Order, I, A>::viewCol(IndexType firstRow, IndexType lastRow,
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
typename DeviceFullStorage<T, Order, I, A>::ArrayView
DeviceFullStorage<T, Order, I, A>::viewCol(IndexType firstRow, IndexType lastRow,
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
const typename DeviceFullStorage<T, Order, I, A>::ConstArrayView
DeviceFullStorage<T, Order, I, A>::viewDiag(IndexType d,
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
typename DeviceFullStorage<T, Order, I, A>::ArrayView
DeviceFullStorage<T, Order, I, A>::viewDiag(IndexType d,
                                            IndexType firstViewIndex)
{
    IndexType _row = (d>0) ? 0 : -d;
    IndexType _col = (d>0) ? d :  0;

    IndexType row = firstRow() + _row;
    IndexType col = firstCol() + _col;

    return ArrayView(std::min(numRows()-_row, numCols()-_col),
                     data(row,col),
                     leadingDimension()+1,
                     firstViewIndex,
                     allocator());
}


// view of d-th diagonal
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceFullStorage<T, Order, I, A>::ConstArrayView
DeviceFullStorage<T, Order, I, A>::viewAntiDiag(IndexType d,
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
typename DeviceFullStorage<T, Order, I, A>::ArrayView
DeviceFullStorage<T, Order, I, A>::viewAntiDiag(IndexType d,
                                                IndexType firstViewIndex)
{
    IndexType _row = (d>0) ? 0 : -d;
    IndexType _col = (d>0) ? d :  0;

    IndexType row = firstRow() + _row;
    IndexType col = firstCol() + _col;

    return ArrayView(std::min(numRows()-_row, numCols()-_col),
                     data(row,lastCol()-col+1),
                     -leadingDimension()+1,
                     firstViewIndex,
                     allocator());
}

//-- Private Methods -----------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceFullStorage<T, Order, I, A>::_setIndexBase(IndexType firstRow,
                                                 IndexType firstCol)
{
    // assume: _data points to allocated memory

    if (Order==RowMajor) {
        _data = _data.shift(-(firstRow*_numCols + firstCol));
    }
    if (Order==ColMajor) {
        _data = _data.shift(-(firstCol*_numRows + firstRow));
    }
    _firstRow = firstRow;
    _firstCol = firstCol;
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceFullStorage<T, Order, I, A>::_raw_allocate()
{
    ASSERT(!_data.get());
    ASSERT(_numRows>0);
    ASSERT(_numCols>0);

    _data = _allocator.allocate(_numRows*_numCols);
#ifndef NDEBUG
    PointerType p = _data;
#endif
    _setIndexBase(_firstRow, _firstCol);
    ASSERT(_data.get());
#ifndef NDEBUG
    ASSERT(p==data());
#endif
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceFullStorage<T, Order, I, A>::_allocate(const ElementType &value)
{
    const IndexType numElements = numRows()*numCols();

    if (numElements==0) {
        return;
    }

    _raw_allocate();
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceFullStorage<T, Order, I, A>::_release()
{
    if (_data.get()) {
        _allocator.deallocate(data(), numRows()*numCols());
        _data = PointerType(NULL, 0);
    }
    ASSERT(_data.get()==0);
}

#endif // HAVE_DEVICE_STORAGE

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_DEVICEFULLSTORAGE_DEVICEFULLSTORAGE_TCC
