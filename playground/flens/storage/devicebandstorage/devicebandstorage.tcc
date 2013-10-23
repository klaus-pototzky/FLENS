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

#ifndef PLAYGROUND_FLENS_STORAGE_DEVICEBANDSTORAGE_DEVICEBANDSTORAGE_TCC
#define PLAYGROUND_FLENS_STORAGE_DEVICEBANDSTORAGE_DEVICEBANDSTORAGE_TCC 1

#include <flens/auxiliary/auxiliary.h>
#include <flens/storage/bandstorage/bandstorage.h>
#include <flens/typedefs.h>

namespace flens {

#ifdef HAVE_DEVICE_STORAGE

//= Constructors

template <typename T, StorageOrder Order, typename I, typename A>
DeviceBandStorage<T, Order, I, A>::DeviceBandStorage()
    :  _data(),
       _numRows(0), _numCols(0),
       _numSubDiags(0), _numSuperDiags(0),
       _firstIndex(I::defaultIndexBase)
{
}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceBandStorage<T, Order, I, A>::DeviceBandStorage(IndexType numRows, IndexType numCols,
                                                     IndexType numSubDiags,
                                                     IndexType numSuperDiags,
                                                     IndexType firstIndex,
                                                     const ElementType &value,
                                                     const Allocator &allocator)
    : _data(), _allocator(allocator),
      _numRows(numRows), _numCols(numCols),
      _numSubDiags(numSubDiags), _numSuperDiags(numSuperDiags),
      _firstIndex(firstIndex)
{

    ASSERT(_numRows>=0);
    ASSERT(_numCols>=0);
    ASSERT(_numSubDiags>=0);
    ASSERT(_numSuperDiags>=0);

    _allocate(value);

}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceBandStorage<T, Order, I, A>::DeviceBandStorage(const DeviceBandStorage &rhs)
    : _data(), _allocator(rhs.allocator()),
      _numRows(rhs.numRows()), _numCols(rhs.numCols()),
      _numSubDiags(rhs.numSubDiags()), _numSuperDiags(rhs.numSuperDiags()),
      _firstIndex(rhs.firstIndex())
{
    _allocate(ElementType());
    const IndexType leadingDimension = _numSubDiags+_numSuperDiags+1;
    if (Order==ColMajor) {
        cxxblas::copy(leadingDimension*_numCols, rhs.data(), 1, _data, 1);
    }
    else {
        cxxblas::copy(leadingDimension*_numRows, rhs.data(), 1, _data, 1);
    }
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename RHS>
DeviceBandStorage<T, Order, I, A>::DeviceBandStorage(const RHS &rhs)
    : _data(), _allocator(rhs.allocator()),
      _numRows(rhs.numRows()), _numCols(rhs.numCols()),
      _numSubDiags(rhs.numSubDiags()), _numSuperDiags(rhs.numSuperDiags()),
      _firstIndex(rhs.firstIndex())
{
    using std::max;
    using std::min;

    _allocate(ElementType());

    for (IndexType row = _firstIndex; row <= _firstIndex+_numRows-1; ++row)
    {
        const IndexType mincol = max(_firstIndex,row-_numSubDiags);
        const IndexType maxcol = min(row+_numSuperDiags,_numCols+_firstIndex-1);
        for (IndexType col = mincol; col <= maxcol; ++col)
            operator()(row, col) = rhs.operator()(row,col);
    }
}

template <typename T, StorageOrder Order, typename I, typename A>
DeviceBandStorage<T, Order, I, A>::~DeviceBandStorage()
{
    _release();
}

//-- operators -----------------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::ElementType &
DeviceBandStorage<T, Order, I, A>::operator()(IndexType row, IndexType col) const
{

    ASSERT(row>=_firstIndex);
    ASSERT(row<_firstIndex+_numRows);
    ASSERT(col>=_firstIndex);
    ASSERT(col<_firstIndex+_numCols);

    ASSERT(max(_firstIndex,col-_numSuperDiags) <= row);
    ASSERT(row <= min(_numRows+_firstIndex-1,col+_numSubDiags));

    const IndexType leadingDimension = _numSubDiags+_numSuperDiags+1;
    if (Order == ColMajor) {
        const IndexType i = _numSuperDiags+row-col;
        const IndexType j = col-_firstIndex;

        return _data[j*leadingDimension+i];
    }

    const IndexType i = _numSubDiags+col-row;
    const IndexType j = row-_firstIndex;
    return _data[j*leadingDimension+i];
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::ElementType &
DeviceBandStorage<T, Order, I, A>::operator()(IndexType row, IndexType col)
{
    using std::max;
    using std::min;
    ASSERT(row>=_firstIndex);
    ASSERT(row<_firstIndex+_numRows);
    ASSERT(col>=_firstIndex);
    ASSERT(col<_firstIndex+_numCols);

    ASSERT(max(_firstIndex,col-_numSuperDiags) <= row);
    ASSERT(row <= min(_numRows+_firstIndex-1,col+_numSubDiags));

    const IndexType leadingDimension = _numSubDiags+_numSuperDiags+1;
    if (Order == ColMajor) {
        const IndexType i = _numSuperDiags+row-col;
        const IndexType j = col-_firstIndex;
        return _data[j*leadingDimension+i];
    }

    const IndexType i = _numSubDiags+col-row;
    const IndexType j = row-_firstIndex;
    return _data[j*leadingDimension+i];

}

//-- Methods -------------------------------------------------------------------
template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::firstRow() const
{
    return _firstIndex;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::lastRow() const
{
    return _firstIndex+_numRows-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::firstCol() const
{
    return _firstIndex;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::lastCol() const
{
    return _firstIndex+_numCols-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::firstIndex() const
{
    return _firstIndex;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::lastIndex() const
{
    return _firstIndex+_numCols-1;
}


template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::numRows() const
{
    return _numRows;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::numCols() const
{
    return _numCols;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::dim() const
{
    ASSERT(_numCols == _numRows);
    return _numCols;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::numSubDiags() const
{
    return _numSubDiags;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::numSuperDiags() const
{
    return _numSuperDiags;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::leadingDimension() const
{
    return std::max(_numSubDiags+_numSuperDiags+1, IndexType(1));
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::strideRow() const
{
    return (Order==ColMajor) ? 1
                             : leadingDimension()-1;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::IndexType
DeviceBandStorage<T, Order, I, A>::strideCol() const
{
    return (Order==ColMajor) ? leadingDimension()-1
                             : 1;
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::ConstPointerType
DeviceBandStorage<T, Order, I, A>::data() const
{
#   ifndef NDEBUG
    if ((numRows()==0) || numCols()==0) {
        return  PointerType(NULL, 0);
    }
#   endif

    return _data;
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::PointerType
DeviceBandStorage<T, Order, I, A>::data()
{
#   ifndef NDEBUG
    if ((numRows()==0) || numCols()==0) {
        return PointerType(NULL, 0);
    }
#   endif

    return _data;
}


template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::ConstPointerType
DeviceBandStorage<T, Order, I, A>::data(IndexType row, IndexType col) const
{
    using std::max;
    using std::min;
    ASSERT(row>=_firstIndex);
    ASSERT(row<_firstIndex+_numRows);
    ASSERT(col>=_firstIndex);
    ASSERT(col<_firstIndex+_numCols);

    ASSERT(max(_firstIndex,col-_numSuperDiags) <= row);
    ASSERT(row <= min(_numRows+_firstIndex-1,col+_numSubDiags));

    const IndexType leadingDimension = _numSubDiags+_numSuperDiags+1;
    if (Order == ColMajor) {
        const IndexType i = _numSuperDiags+row-col;
        const IndexType j = col-_firstIndex;
        return _data.shift(j*leadingDimension+i);
    }

    const IndexType i = _numSubDiags+col-row;
    const IndexType j = row-_firstIndex;
    return _data.shift(j*leadingDimension+i);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::PointerType
DeviceBandStorage<T, Order, I, A>::data(IndexType row, IndexType col)
{
    using std::max;
    using std::min;
    ASSERT(row>=_firstIndex);
    ASSERT(row<_firstIndex+_numRows);
    ASSERT(col>=_firstIndex);
    ASSERT(col<_firstIndex+_numCols);

    ASSERT(max(_firstIndex,col-_numSuperDiags) <= row);
    ASSERT(row <= min(_numRows+_firstIndex-1,col+_numSubDiags));

    const IndexType leadingDimension = _numSubDiags+_numSuperDiags+1;
    if (Order == ColMajor) {
        const IndexType i = _numSuperDiags+row-col;
        const IndexType j = col-_firstIndex;
        return _data.shift(j*leadingDimension+i);
    }

    const IndexType i = _numSubDiags+col-row;
    const IndexType j = row-_firstIndex;
    return _data.shift(j*leadingDimension+i);
}

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::Allocator &
DeviceBandStorage<T, Order, I, A>::allocator() const
{
    return _allocator;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceBandStorage<T, Order, I, A>::resize(IndexType numRows, IndexType numCols,
                                    IndexType numSubDiags,
                                    IndexType numSuperDiags,
                                    IndexType firstIndex,
                                    const ElementType &value)
{
    if ((_numSubDiags!=numSubDiags) ||(_numSuperDiags!=numSuperDiags)
      || (_numRows!=numRows) || (_numCols!=numCols)) {
        _release();
        _numSubDiags = numSubDiags,
        _numSuperDiags = numSuperDiags,
        _numRows = numRows;
        _numCols = numCols;
        _firstIndex = firstIndex;
        _allocate(value);
        return true;
    }
    _setIndexBase(firstIndex);
    return false;
}

template <typename T, StorageOrder Order, typename I, typename A>
template <typename FS>
bool
DeviceBandStorage<T, Order, I, A>::resize(const FS &rhs, const ElementType &value)
{
    return resize(rhs.numRows(), rhs.numCols(),
                  rhs.numSubDiags(), rhs.numSuperDiags(),
                  rhs.firstIndex(),
                  value);
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceBandStorage<T, Order, I, A>::fill(const ElementType &value)
{
    const IndexType m = _numSubDiags+_numSuperDiags+1;
    if (Order==ColMajor) {
        flens::fill_n(_data, m*_numCols, value);
    }
    else {
        flens::fill_n(_data, m*_numRows, value);
    }

    return true;
}

template <typename T, StorageOrder Order, typename I, typename A>
bool
DeviceBandStorage<T, Order, I, A>::fillRandom()
{
    const IndexType m = _numSubDiags+_numSuperDiags+1;
    if (Order==ColMajor) {
        for (IndexType i=0; i<m*_numCols;++i) {
            _data[i] = randomValue<T>();
        }
    }
    else {
        for (IndexType i=0; i<m*_numRows;++i) {
            _data[i] = randomValue<T>();
        }
    }

    return true;
}

// view of fullstorage scheme as an array
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::ConstArrayView
DeviceBandStorage<T, Order, I, A>::arrayView(IndexType firstViewIndex) const
{
    if (Order==RowMajor) {
        return ConstArrayView((_numSubDiags+_numSuperDiags+1)*_numRows,
                              _data,
                              IndexType(1),
                              firstViewIndex,
                              _allocator);
    }

    return ConstArrayView((_numSubDiags+_numSuperDiags+1)*_numRows,
                          _data,
                          IndexType(1),
                          firstViewIndex,
                          _allocator);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::ArrayView
DeviceBandStorage<T, Order, I, A>::arrayView(IndexType firstViewIndex)
{
    if (Order==RowMajor) {
        return ArrayView((_numSubDiags+_numSuperDiags+1)*_numRows,
                              _data,
                              IndexType(1),
                              firstViewIndex,
                              allocator());
    }

    return ArrayView((_numSubDiags+_numSuperDiags+1)*_numCols,
                          _data,
                          IndexType(1),
                          firstViewIndex,
                          allocator());
}


// view of a diagonal
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::ConstArrayView
DeviceBandStorage<T, Order, I, A>::viewDiag(IndexType diag,
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
                          _numSubDiags+_numSuperDiags+1,
                          _firstIndex, _allocator);

}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::ArrayView
DeviceBandStorage<T, Order, I, A>::viewDiag(IndexType diag,
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
                     _numSubDiags+_numSuperDiags+1,
                     _firstIndex, _allocator);


}

// View some diagonals
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::ConstView
DeviceBandStorage<T, Order, I, A>::viewDiags(IndexType fromDiag,
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
                             _numSubDiags+_numSuperDiags+1,
                             _firstIndex, _allocator);
        }
        if (fromDiag > 0) {
            return ConstView(numRows, numCols, 0, toDiag-fromDiag,
                             data(i,j),
                             _numSubDiags+_numSuperDiags+1,
                             _firstIndex, _allocator);
        }
        return ConstView(numRows, numCols, -fromDiag, toDiag,
                         _data.shift(_numSubDiags+fromDiag),
                         _numSubDiags+_numSuperDiags+1,
                         _firstIndex, _allocator);
    }

    if (toDiag < 0) {
        return ConstView(numRows, numCols, -fromDiag+toDiag, 0,
                         data(i,j),
                         _numSubDiags+_numSuperDiags+1,
                         _firstIndex, _allocator);
    }
    if (fromDiag > 0) {
        return ConstView(numRows, numCols, 0, toDiag-fromDiag,
                         data(i,j).shift(+ fromDiag-toDiag),
                         _numSubDiags+_numSuperDiags+1,
                         _firstIndex, _allocator);
    }
    return ConstView(_numRows, _numCols, -fromDiag, toDiag,
                     _data.shift(_numSuperDiags-toDiag),
                     _numSubDiags+_numSuperDiags+1,
                     _firstIndex, _allocator);
}

template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::View
DeviceBandStorage<T, Order, I, A>::viewDiags(IndexType fromDiag, IndexType toDiag)
{
    // WORKING !!!!
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
                        data(i,j).shift( + fromDiag-toDiag ),
                        _numSubDiags+_numSuperDiags+1,
                        _firstIndex, _allocator);
        }
        if (fromDiag > 0) {
            return View(numRows, numCols, 0, toDiag-fromDiag,
                        data(i,j),
                        _numSubDiags+_numSuperDiags+1,
                        _firstIndex, _allocator);
        }
        return View(numRows, numCols, -fromDiag, toDiag,
                    _data.shift(_numSubDiags+fromDiag),
                    _numSubDiags+_numSuperDiags+1,
                    _firstIndex, _allocator);
    }

    if (toDiag < 0) {
        return View(numRows, numCols, -fromDiag+toDiag, 0,
                    data(i,j),
                    _numSubDiags+_numSuperDiags+1,
                    _firstIndex, _allocator);
    }
    if (fromDiag > 0) {
        return View(numRows, numCols, 0, toDiag-fromDiag,
                    data(i,j).shift( + fromDiag-toDiag),
                    _numSubDiags+_numSuperDiags+1,
                    _firstIndex, _allocator);
    }
    return View(_numRows, _numCols, -fromDiag, toDiag,
                _data.shift(_numSuperDiags-toDiag),
                _numSubDiags+_numSuperDiags+1,
                _firstIndex, _allocator);
}

// view of single row
template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::ConstArrayView
DeviceBandStorage<T, Order, I, A>::viewRow(IndexType row,
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
typename DeviceBandStorage<T, Order, I, A>::ArrayView
DeviceBandStorage<T, Order, I, A>::viewRow(IndexType row,
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
const typename DeviceBandStorage<T, Order, I, A>::ConstArrayView
DeviceBandStorage<T, Order, I, A>::viewRow(IndexType row,
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
typename DeviceBandStorage<T, Order, I, A>::ArrayView
DeviceBandStorage<T, Order, I, A>::viewRow(IndexType row,
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
const typename DeviceBandStorage<T, Order, I, A>::ConstArrayView
DeviceBandStorage<T, Order, I, A>::viewCol(IndexType col,
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
typename DeviceBandStorage<T, Order, I, A>::ArrayView
DeviceBandStorage<T, Order, I, A>::viewCol(IndexType col,
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
const typename DeviceBandStorage<T, Order, I, A>::ConstArrayView
DeviceBandStorage<T, Order, I, A>::viewCol(IndexType firstRow, IndexType lastRow,
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
typename DeviceBandStorage<T, Order, I, A>::ArrayView
DeviceBandStorage<T, Order, I, A>::viewCol(IndexType firstRow, IndexType lastRow,
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

template <typename T, StorageOrder Order, typename I, typename A>
const typename DeviceBandStorage<T, Order, I, A>::ConstFullStorageView
DeviceBandStorage<T, Order, I, A>::viewFullStorage() const
{
    return ConstFullStorageView(numSubDiags()+numSuperDiags()+1, 
                                max(numRows(),numCols()), 
                                data(), 
                                leadingDimension());
}


template <typename T, StorageOrder Order, typename I, typename A>
typename DeviceBandStorage<T, Order, I, A>::FullStorageView
DeviceBandStorage<T, Order, I, A>::viewFullStorage()
{
    return FullStorageView(numSubDiags()+numSuperDiags()+1, 
                           max(numRows(),numCols()),
                           data(),
                           leadingDimension());
}

//-- Private Methods -----------------------------------------------------------

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceBandStorage<T, Order, I, A>::_setIndexBase(IndexType firstIndex)
{
    _firstIndex = firstIndex;
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceBandStorage<T, Order, I, A>::_raw_allocate()
{
    ASSERT(!_data.get());
    ASSERT(_numRows>0);
    ASSERT(_numCols>0);

    const IndexType m = _numSubDiags+_numSuperDiags+1;
    if (Order==ColMajor) {
        _data = _allocator.allocate(m*_numCols);
    }
    else {
        _data = _allocator.allocate(m*_numRows);
    }

    _setIndexBase(_firstIndex);

}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceBandStorage<T, Order, I, A>::_allocate(const ElementType &value)
{

    if (numRows()*numCols()==0) {
        return;
    }

    _raw_allocate();
    fill(value);
}

template <typename T, StorageOrder Order, typename I, typename A>
void
DeviceBandStorage<T, Order, I, A>::_release()
{
    if (_data.get()) {
        IndexType numElements = (_numSubDiags+_numSuperDiags+1)*_numCols;
        if (Order == RowMajor)
            numElements = (_numSubDiags+_numSuperDiags+1)*_numRows;

         _allocator.deallocate(data(), numElements);
        _data = PointerType(NULL, 0);
    }
    ASSERT(_data.get()==0);
}

#endif 

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_DEVICEBANDSTORAGE_DEVICEBANDSTORAGE_TCC
