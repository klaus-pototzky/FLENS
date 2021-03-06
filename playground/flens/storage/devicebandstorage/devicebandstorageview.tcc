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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEBANDSTORAGE_DEVICEBANDSTORAGEVIEW_TCC
#define PLAYGROUND_FLENS_STORAGE_DEVICEBANDSTORAGE_DEVICEBANDSTORAGEVIEW_TCC 1

#include <flens/auxiliary/auxiliary.h>
#include <flens/storage/bandstorage/bandstorageview.h>
#include <flens/typedefs.h>

namespace flens {
  
#ifdef HAVE_DEVICE_STORAGE

template <typename T, StorageOrder Order, typename I, typename A>
DeviceBandStorageView<T, Order, I, A>::DeviceBandStorageView(IndexType numRows,
                                                             IndexType numCols,
                                                             IndexType numSubDiags,
                                                             IndexType numSuperDiags,
                                                             PointerType data,
                                                             IndexType leadingDimension,
                                                             IndexType firstIndex,
                                                             const Allocator &allocator)
    : _data(data),
      _allocator(allocator),
      _numRows(numRows), _numCols(numCols),
      _numSubDiags(numSubDiags), _numSuperDiags(numSuperDiags),
      _firstIndex(firstIndex),
      _leadingDimension(leadingDimension)
{
    ASSERT(_numRows>=0);
    ASSERT(_numCols>=0);
    ASSERT(_numSubDiags>=0);
    ASSERT(_numSuperDiags>=0);

}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceBandStorageView<T, Order, I, A>::DeviceBandStorageView(const DeviceBandStorageView &rhs)
    : _data(rhs._data),
      _allocator(rhs._allocator),
      _numRows(rhs.numRows()), _numCols(rhs.numCols()),
      _numSubDiags(rhs._numSubDiags), _numSuperDiags(rhs._numSuperDiags),
      _firstIndex(rhs.firstIndex()),
      _leadingDimension(rhs._leadingDimension)
{

}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename RHS>
DeviceBandStorageView<T, Order, I, A>::DeviceBandStorageView(RHS &rhs)
    : _data(rhs.data()),
      _allocator(rhs.allocator()),
      _numRows(rhs.numRows()), _numCols(rhs.numCols()),
      _numSubDiags(rhs.numSubDiags()), _numSuperDiags(rhs.numSuperDiags()),
      _firstIndex(rhs.firstIndex()),
      _leadingDimension(rhs.leadingDimension())
{

}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceBandStorageView<T, Order, I, A>::~DeviceBandStorageView()
{
}

//-- operators -----------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ElementType &
DeviceBandStorageView<T, Order, I, A>::operator()(IndexType row, IndexType col) const
{
    ASSERT(row>=_firstIndex);
    ASSERT(row<_firstIndex+_numRows);
    ASSERT(col>=_firstIndex);
    ASSERT(col<_firstIndex+_numCols);

    ASSERT(max(_firstIndex,col-_numSuperDiags) <= row);
    ASSERT(row <= min(_numRows+_firstIndex-1,col+_numSubDiags));

    if (Order == ColMajor) {
        const IndexType i = _numSuperDiags+row-col;
        const IndexType j = col-_firstIndex;
        return _data[j*_leadingDimension+i];
    }
    const IndexType i = _numSubDiags+col-row;
    const IndexType j = row-_firstIndex;
    return _data[j*_leadingDimension+i];
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::ElementType &
DeviceBandStorageView<T, Order, I, A>::operator()(IndexType row, IndexType col)
{
    ASSERT(row>=_firstIndex);
    ASSERT(row<_firstIndex+_numRows);
    ASSERT(col>=_firstIndex);
    ASSERT(col<_firstIndex+_numCols);

    ASSERT(max(_firstIndex,col-_numSuperDiags) <= row);
    ASSERT(row <= min(_numRows+_firstIndex-1,col+_numSubDiags));
    if (Order == ColMajor) {
        const IndexType i = _numSuperDiags+row-col;
        const IndexType j = col-_firstIndex;
        return _data[j*_leadingDimension+i];
    }
    const IndexType i = _numSubDiags+col-row;
    const IndexType j = row-_firstIndex;
    return _data[j*_leadingDimension+i];
}

//-- methods -------------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::firstRow() const
{
    return _firstIndex;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::lastRow() const
{
    return _firstIndex+_numRows-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::firstCol() const
{
    return _firstIndex;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::lastCol() const
{
    return _firstIndex+_numCols-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::firstIndex() const
{
    return _firstIndex;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::lastIndex() const
{
    return _firstIndex+_numCols-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::numRows() const
{
    return _numRows;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::numCols() const
{
    return _numCols;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::dim() const
{
    ASSERT(_numCols == _numRows);
    return _numCols;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::numSubDiags() const
{
    return _numSubDiags;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::numSuperDiags() const
{
    return _numSuperDiags;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::leadingDimension() const
{
    return _leadingDimension;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::strideRow() const
{
    return (Order==ColMajor) ? 1
                             : leadingDimension()-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::IndexType
DeviceBandStorageView<T, Order, I, A>::strideCol() const
{
    return (Order==ColMajor) ? leadingDimension()-1
                             : 1;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstPointerType
DeviceBandStorageView<T, Order, I, A>::data() const
{
    return _data;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::PointerType
DeviceBandStorageView<T, Order, I, A>::data()
{
    return _data;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstPointerType
DeviceBandStorageView<T, Order, I, A>::data(IndexType row, IndexType col) const
{
    ASSERT(row>=_firstIndex);
    ASSERT(row<_firstIndex+_numRows);
    ASSERT(col>=_firstIndex);
    ASSERT(col<_firstIndex+_numCols);

    ASSERT(max(_firstIndex,col-_numSuperDiags) <= row);
    ASSERT(row <= min(_numRows+_firstIndex-1,col+_numSubDiags));

    if (Order == ColMajor) {
        const IndexType i = _numSuperDiags+row-col;
        const IndexType j = col-_firstIndex;
        return _data.shift(j*_leadingDimension+i);
    }
    const IndexType i = _numSubDiags+col-row;
    const IndexType j = row-_firstIndex;
    return _data.shift(j*_leadingDimension+i);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::PointerType
DeviceBandStorageView<T, Order, I, A>::data(IndexType row, IndexType col)
{
    ASSERT(row>=_firstIndex);
    ASSERT(row<_firstIndex+_numRows);
    ASSERT(col>=_firstIndex);
    ASSERT(col<_firstIndex+_numCols);

    ASSERT(max(_firstIndex,col-_numSuperDiags) <= row);
    ASSERT(row <= min(_numRows+_firstIndex-1,col+_numSubDiags));

    if (Order == ColMajor) {
        const IndexType i = _numSuperDiags+row-col;
        const IndexType j = col-_firstIndex;
        return _data.shift(j*_leadingDimension+i);
    }
    const IndexType i = _numSubDiags+col-row;
    const IndexType j = row-_firstIndex;
    return _data.shift(j*_leadingDimension+i);
}



template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::Allocator &
DeviceBandStorageView<T, Order, I, A>::allocator() const
{
    return _allocator;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceBandStorageView<T, Order, I, A>::resize(IndexType numRows,
                                              IndexType numCols,
                                              IndexType numSubDiags,
                                              IndexType numSuperDiags,
                                              IndexType firstIndex,
                                              const ElementType &)
{
    ASSERT(_numRows==numRows);
    ASSERT(_numCols==numCols);
    ASSERT(_numSubDiags==numSubDiags);
    ASSERT(_numSuperDiags==numSuperDiags);
    changeIndexBase(firstIndex);
    return false;
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename FS>
bool
DeviceBandStorageView<T, Order, I, A>::resize(const FS &rhs, const ElementType &value)
{
    return resize(rhs.numRows(), rhs.numCols(),
                  rhs.numSubDiags(), rhs.numSuperDiags(), rhs.firstCol(),
                  value);
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceBandStorageView<T, Order, I, A>::fill(const ElementType &value)
{

    for (IndexType i=-numSubDiags(); i<=numSuperDiags(); ++i) {
        diag(i) = value; 
    }

    return true;
}


template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceBandStorageView<T, Order, I, A>::fillRandom()
{
    using std::max;
    using std::min;

    for (IndexType row = _firstIndex; row <= _firstIndex+_numRows-1; ++row)
    {
        const IndexType mincol = max(_firstIndex,row-_numSubDiags);
        const IndexType maxcol = min(row+_numSuperDiags,_numCols+_firstIndex-1);
        for (IndexType col = mincol; col <= maxcol; ++col)
            operator()(row, col) = randomValue<T>();
    }

    return true;
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceBandStorageView<T, Order, I, A>::changeIndexBase(IndexType firstIndex)
{
    _firstIndex = firstIndex;
}

// view of fullstorage scheme as an array
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstArrayView
DeviceBandStorageView<T, Order, I, A>::arrayView(IndexType firstViewIndex) const
{

    if (Order==RowMajor) {
        return ConstArrayView(_leadingDimension*_numRows,
                              _data,
                              IndexType(1),
                              firstViewIndex,
                              allocator());
    }

    return ConstArrayView(_leadingDimension*_numRows,
                          _data,
                          IndexType(1),
                          firstViewIndex,
                          allocator());
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::ArrayView
DeviceBandStorageView<T, Order, I, A>::arrayView(IndexType firstViewIndex)
{
    if (Order==RowMajor) {
        return ArrayView(_leadingDimension*_numCols,
                          _data,
                          IndexType(1),
                          firstViewIndex,
                          allocator());
    }

    return ArrayView(_leadingDimension*_numRows,
                      _data,
                      IndexType(1),
                      firstViewIndex,
                      allocator());
}

// View a diagonal
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstArrayView
DeviceBandStorageView<T, Order, I, A>::viewDiag(IndexType diag,
                                          IndexType firstViewIndex) const
{

    ASSERT( diag <= _numSuperDiags);
    ASSERT(-diag <= _numSubDiags);

    using std::min;

    const IndexType i = (diag < 0) ? -diag+firstViewIndex: firstViewIndex;
    const IndexType j = (diag > 0) ?  diag+firstViewIndex: firstViewIndex;
    const IndexType length = (diag<=0) ? min(_numCols, _numRows+diag)
                                       : min(_numCols-diag, _numRows);

    return ConstArrayView(length-firstViewIndex+_firstIndex,
                          data(i, j),
                          _leadingDimension,
                          _firstIndex, _allocator);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::ArrayView
DeviceBandStorageView<T, Order, I, A>::viewDiag(IndexType diag,
                                          IndexType firstViewIndex)
{
    ASSERT( diag <= _numSuperDiags);
    ASSERT(-diag <= _numSubDiags);

    using std::min;

    const IndexType i = (diag < 0) ? -diag+firstViewIndex: firstViewIndex;
    const IndexType j = (diag > 0) ?  diag+firstViewIndex: firstViewIndex;
    const IndexType length = (diag<=0) ? min(_numCols, _numRows+diag)
                                       : min(_numCols-diag, _numRows);

    return ArrayView(length-firstViewIndex+_firstIndex,
                     data(i, j),
                     _leadingDimension,
                     _firstIndex, _allocator);
}


template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstView
DeviceBandStorageView<T, Order, I, A>::viewDiags(IndexType fromDiag,
                                           IndexType toDiag) const
{
    ASSERT(fromDiag<=toDiag);
    IndexType numRows = _numRows;
    IndexType numCols = _numCols;

    if (fromDiag>0) {
        numCols = _numCols - fromDiag;
        if (_numRows<_numCols) {
            if (_numCols-_numRows < fromDiag) {
                numRows = _numCols - fromDiag;
            }
        } else {
            numRows = _numCols - fromDiag;
        }
    }
    if (toDiag<0) {
        numRows = _numRows + toDiag;
        if (_numCols<_numRows) {
            if (_numRows-_numCols < -toDiag) {
              numCols = _numRows + toDiag;
            }
        } else {
            numCols = _numRows + toDiag;
        }
    }

    const IndexType i = _firstIndex - ((toDiag<0) ? toDiag : 0);
    const IndexType j = _firstIndex + ((fromDiag>0) ? fromDiag : 0);

    if (Order == RowMajor ) {
        if (toDiag < 0) {
            return ConstView(numRows, numCols, -fromDiag+toDiag, 0,
                             data(i,j).shift( + fromDiag-toDiag),
                            _leadingDimension,
                             _firstIndex, _allocator);
        }
        if (fromDiag > 0) {
            return ConstView(numRows, numCols, 0, toDiag-fromDiag,
                             data(i,j),
                             _leadingDimension,
                             _firstIndex, _allocator);
        }
        return ConstView(numRows, numCols, -fromDiag, toDiag,
                         _data.shift(_numSubDiags+fromDiag),
                         _leadingDimension,
                         _firstIndex, _allocator);
    }

    if (toDiag < 0) {
        return ConstView(numRows, numCols, -fromDiag+toDiag, 0,
                         data(i,j),
                         _leadingDimension,
                         _firstIndex, _allocator);
    }
    if (fromDiag > 0) {
        return ConstView(numRows, numCols, 0, toDiag-fromDiag,
                         data.shift(fromDiag-toDiag),
                         _leadingDimension,
                         _firstIndex, _allocator);
    }
    return ConstView(_numRows, _numCols, -fromDiag, toDiag,
                     _data.shift(_numSuperDiags-toDiag),
                     _leadingDimension,
                     _firstIndex, _allocator);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::View
DeviceBandStorageView<T, Order, I, A>::viewDiags(IndexType fromDiag, IndexType toDiag)
{
    ASSERT(fromDiag<=toDiag);
    IndexType numRows = _numRows;
    IndexType numCols = _numCols;

    if (fromDiag>0) {
        numCols = _numCols - fromDiag;
        if (_numRows<_numCols) {
            if (_numCols-_numRows < fromDiag) {
                numRows = _numCols - fromDiag;
            }
        } else {
            numRows = _numCols - fromDiag;
        }
    }
    if (toDiag<0) {
        numRows = _numRows + toDiag;
        if (_numCols<_numRows) {
            if (_numRows-_numCols < -toDiag) {
              numCols = _numRows + toDiag;
            }
        } else {
            numCols = _numRows + toDiag;
        }
    }

    const IndexType i = _firstIndex - ((toDiag<0) ? toDiag : 0);
    const IndexType j = _firstIndex + ((fromDiag>0) ? fromDiag : 0);

    if (Order == RowMajor ) {
        if (toDiag < 0) {
            return View(numRows, numCols, -fromDiag+toDiag, 0,
                        data(i,j).shift(fromDiag-toDiag),
                        _leadingDimension,
                        _firstIndex, _allocator);
        }
        if (fromDiag > 0) {
            return View(numRows, numCols, 0, toDiag-fromDiag,
                        data(i,j),
                        _leadingDimension,
                        _firstIndex, _allocator);
        }
        return View(numRows, numCols, -fromDiag, toDiag,
                    _data.shift(_numSubDiags+fromDiag),
                    _leadingDimension,
                    _firstIndex, _allocator);
    }

    if (toDiag < 0) {
        return View(numRows, numCols, -fromDiag+toDiag, 0,
                    data(i,j),
                    _leadingDimension,
                    _firstIndex, _allocator);
    }
    if (fromDiag > 0) {
        return View(numRows, numCols, 0, toDiag-fromDiag,
                    data(i,j).shift(fromDiag-toDiag),
                   _leadingDimension,
                    _firstIndex, _allocator);
    }
    return View(_numRows, _numCols, -fromDiag, toDiag,
                _data.shift(_numSuperDiags-toDiag),
                _leadingDimension,
                _firstIndex, _allocator);
}


// view of single row
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstArrayView
DeviceBandStorageView<T, Order, I, A>::viewRow(IndexType row,
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
    
    IndexType length = 1+min(row-1,numSubDiags())+min(numRows()-row,numSuperDiags());
    return ConstArrayView(length,
                          data(row, firstCol()+max(0,row-1-numSubDiags())),
                          strideCol(),
                          firstViewIndex,
                          allocator());
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::ArrayView
DeviceBandStorageView<T, Order, I, A>::viewRow(IndexType row,
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
    
    IndexType length = 1+min(row-1,numSubDiags())+min(numRows()-row,numSuperDiags());
    return ArrayView(length,
                     data(row, firstCol()+max(0,row-1-numSubDiags())),
                     strideCol(),
                     firstViewIndex,
                     allocator());
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstArrayView
DeviceBandStorageView<T, Order, I, A>::viewRow(IndexType row,
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
typename DeviceBandStorageView<T, Order, I, A>::ArrayView
DeviceBandStorageView<T, Order, I, A>::viewRow(IndexType row,
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

// view of single row
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstArrayView
DeviceBandStorageView<T, Order, I, A>::viewCol(IndexType col,
                                         IndexType firstViewIndex) const
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

    IndexType length = 1+min(col-1,numSuperDiags())+min(numCols()-col,numSubDiags());
    
    return ArrayView(length,
                     data(firstRow()+max(0,col-1-numSuperDiags()), col),
                     strideRow(),
                     firstViewIndex,
                     allocator());
                     
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorageView<T, Order, I, A>::ArrayView
DeviceBandStorageView<T, Order, I, A>::viewCol(IndexType col,
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

    IndexType length = 1+min(col-1,numSuperDiags())+min(numCols()-col,numSubDiags());
    
    return ArrayView(length,
                     data(firstRow()+max(0,col-1-numSuperDiags()), col),
                     strideRow(),
                     firstViewIndex,
                     allocator());
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorageView<T, Order, I, A>::ConstArrayView
DeviceBandStorageView<T, Order, I, A>::viewCol(IndexType firstRow, IndexType lastRow,
                                         IndexType stride, IndexType col,
                                         IndexType firstViewIndex) const
{
    const IndexType length = (lastRow-firstRow)/stride+1;

#   ifndef NDEBUG
    // prevent an out-of-bound assertion in case a view is empty anyway
    if (length==0) {
        return ConstArrayView(length, 0, strideRow()*stride,
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
typename DeviceBandStorageView<T, Order, I, A>::ArrayView
DeviceBandStorageView<T, Order, I, A>::viewCol(IndexType firstRow, IndexType lastRow,
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
                     data(firstRow(), col),
                     strideRow()*stride,
                     firstViewIndex,
                     allocator());
}

#endif

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_DEVICEBANDSTORAGE_DEVICEBANDSTORAGEVIEW_TCC
