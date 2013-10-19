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

#ifndef PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAY_H
#define PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAY_H 1

#include <playground/flens/auxiliary/devicePtr.h>
#include <playground/flens/auxiliary/customAllocator.h>
#include <flens/storage/indexoptions.h>

namespace flens {
  
#ifdef HAVE_DEVICE_STORAGE

template <typename T, typename I, typename A>
    class ConstDeviceArrayView;

template <typename T, typename I, typename A>
    class DeviceArrayView;

template <typename T,
          typename I = IndexOptions<>,
          typename A = CustomAllocator<T, StorageType::DEFAULT_DEVICE_STORAGE_TYPE> >
class DeviceArray
{
    public:
        typedef T                                ElementType;
        typedef device_ptr<T, A::Type>           PointerType;
        typedef const device_ptr<T, A::Type>     ConstPointerType;
        typedef typename I::IndexType            IndexType;
        typedef A                                Allocator;

        typedef ConstDeviceArrayView<T, I, A>    ConstView;
        typedef DeviceArrayView<T, I, A>         View;
        typedef DeviceArray                      NoView;

        static const IndexType                   defaultIndexBase 
                                                         = I::defaultIndexBase;

        DeviceArray();

        DeviceArray(IndexType length,
                    IndexType firstIndex = defaultIndexBase,
                    const ElementType &value = ElementType(),
                    const Allocator &allocator = Allocator());

        DeviceArray(const DeviceArray &rhs);

        template <typename RHS>
            DeviceArray(const RHS &rhs);

        ~DeviceArray();

        //-- operators ---------------------------------------------------------

        const ElementType &
        operator()(IndexType index) const;

        ElementType &
        operator()(IndexType index);

        //-- methods -----------------------------------------------------------

        IndexType
        firstIndex() const;

        IndexType
        lastIndex() const;

        IndexType
        length() const;

        IndexType
        stride() const;

        const ConstPointerType
        data() const;

        PointerType
        data();

        const ConstPointerType
        data(IndexType index) const;

        PointerType
        data(IndexType index);
	
        const Allocator &
        allocator() const;

        bool
        resize(IndexType length,
               IndexType firstIndex = defaultIndexBase,
               const ElementType &value = ElementType());

        template <typename ARRAY>
            bool
            resize(const ARRAY &rhs, const ElementType &value = ElementType());

        bool
        fill(const ElementType &value = ElementType(0));

        void
        changeIndexBase(IndexType firstIndex);

        const ConstView
        view(IndexType from, IndexType to,
             IndexType stride = IndexType(1),
             IndexType firstViewIndex = defaultIndexBase) const;

        View
        view(IndexType from, IndexType to,
             IndexType stride = IndexType(1),
             IndexType firstViewIndex = defaultIndexBase);

    private:
        void
        _raw_allocate();

        void
        _allocate(const ElementType &value  = ElementType());

        void
        _release();

        PointerType _data;
        Allocator   _allocator;
        IndexType   _length, _firstIndex;
};

#endif // defined(HAVE_CUBLAS) || defined(HAVE_CLBLAS)

} // namespace flens

#endif // PLAYGROUND_FLENS_STORAGE_ARRAY_DEVICEARRAY_H
