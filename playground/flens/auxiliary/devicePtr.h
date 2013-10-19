/*
 *   Copyright (c) 2013, Klaus Pototzky
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

#ifndef PLAYGROUND_FLENS_AUXILIARY_DEVICEPTR_H
#define PLAYGROUND_FLENS_AUXILIARY_DEVICEPTR_H 1

namespace flens {

enum StorageType {
    CUDA,
    OpenCL
};

template <typename DataType, StorageType Storage>
class device_ptr {

};

#ifdef HAVE_OPENCL

template <typename DataType>
class device_ptr<DataType, StorageType::OpenCL> {

    public:
        device_ptr();

        device_ptr(const cl_mem &_ptr, int _offset);

        device_ptr(const device_ptr<DataType, StorageType::OpenCL> &other);

        cl_mem &
        get();

        const cl_mem & 
        get() const;

        device_ptr<DataType, StorageType::OpenCL>
        shift(int dist);
        
        device_ptr<DataType, StorageType::OpenCL>
        shift(int dist) const;
        
        int &
        getOffset();

        const int & 
        getOffset() const ;

    private:
        cl_mem           ptr;
        int              offset;
};

template <typename DataTypeThis, typename DataTypeThat>
bool operator == (const device_ptr<DataTypeThis, StorageType::OpenCL> & compareThis, 
                  const device_ptr<DataTypeThat, StorageType::OpenCL> & compareThat);

template <typename DataTypeThis, typename DataTypeThat>
bool operator != (const device_ptr<DataTypeThis, StorageType::OpenCL> & compareThis, 
                  const device_ptr<DataTypeThat, StorageType::OpenCL> & compareThat);

#endif // HAVE_OPENCL

#ifdef WITH_CUBLAS

template <typename DataType>
class device_ptr<DataType, StorageType::CUDA> {

    public:
        device_ptr();

        device_ptr(DataType *_ptr, int _deviceID);

        device_ptr(const device_ptr<DataType, StorageType::CUDA> &other) ;

        DataType 
        *get() const;

        device_ptr<DataType, StorageType::CUDA> 
        shift(int dist);

        const device_ptr<DataType, StorageType::CUDA> 
        shift(int dist) const;

        int       
        getDeviceID() const;

    private:
        DataType*        ptr;
        int              deviceID;
};


template <typename DataTypeThis, typename DataTypeThat>
bool operator == (const device_ptr<DataTypeThis, StorageType::CUDA> & compareThis, 
                  const device_ptr<DataTypeThat, StorageType::CUDA> & compareThat);
template <typename DataTypeThis, typename DataTypeThat>
bool operator != (const device_ptr<DataTypeThis, StorageType::CUDA> & compareThis, 
                  const device_ptr<DataTypeThat, StorageType::CUDA> & compareThat);

#endif // WITH_CUBLAS



}

#endif // PLAYGROUND_FLENS_AUXILIARY_DEVICEPTR_H
