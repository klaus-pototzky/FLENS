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

#ifndef PLAYGROUND_FLENS_AUXILIARY_DEVICEPTR_TCC
#define PLAYGROUND_FLENS_AUXILIARY_DEVICEPTR_TCC 1

namespace flens {

#ifdef HAVE_OPENCL

template <typename DataType>
device_ptr<DataType, StorageType::OpenCL>::device_ptr()
    : ptr(NULL), offset(0)
{
}

template <typename DataType>
device_ptr<DataType, StorageType::OpenCL>::device_ptr(const cl_mem &_ptr, int _offset)
    : ptr(_ptr), offset(_offset)
{
}

template <typename DataType>
device_ptr<DataType, StorageType::OpenCL>::device_ptr(const device_ptr<typename std::add_const<DataType>::type, StorageType::OpenCL> &other)
    : ptr(other.get()), offset(other.getOffset())
{
}

template <typename DataType>
device_ptr<DataType, StorageType::OpenCL>::device_ptr(const device_ptr<typename std::remove_const<DataType>::type, StorageType::OpenCL> &other)
    : ptr(other.get()), offset(other.getOffset())
{
}


template <typename DataType>
cl_mem &
device_ptr<DataType, StorageType::OpenCL>::get()
{
    return ptr;
}

template <typename DataType>
const cl_mem &
device_ptr<DataType, StorageType::OpenCL>::get() const
{
    return ptr;
}

template <typename DataType>
device_ptr<DataType, StorageType::OpenCL>
device_ptr<DataType, StorageType::OpenCL>::shift(int dist)
{
    return device_ptr<DataType, StorageType::OpenCL>(ptr, offset+dist);
}

template <typename DataType>
device_ptr<DataType, StorageType::OpenCL>
device_ptr<DataType, StorageType::OpenCL>::shift(int dist) const
{
    return device_ptr<DataType, StorageType::OpenCL>(ptr, offset+dist);
}

template <typename DataType>
int &
device_ptr<DataType, StorageType::OpenCL>::getOffset()
{
    return offset;
}

template <typename DataType>
const int &
device_ptr<DataType, StorageType::OpenCL>::getOffset() const
{
    return offset;
}

template <typename DataTypeThis, typename DataTypeThat>
bool operator == (const device_ptr<DataTypeThis, StorageType::OpenCL> & compareThis, 
                  const device_ptr<DataTypeThat, StorageType::OpenCL> & compareThat) {
    return (compareThis.get()==compareThat.get() 
            && compareThis.getOffset()==compareThat.getOffset());
}

template <typename DataTypeThis, typename DataTypeThat>
bool operator != (const device_ptr<DataTypeThis, StorageType::OpenCL> & compareThis, 
                  const device_ptr<DataTypeThat, StorageType::OpenCL> & compareThat) {
    return (compareThis.get()!=compareThat.get() 
            || compareThis.getOffset()!=compareThat.getOffset());
}

template <typename DataTypeThis, typename IndexType>
device_ptr<DataTypeThis, StorageType::OpenCL>
operator+ (const device_ptr<DataTypeThis, StorageType::OpenCL> &left, IndexType right)
{
    return left.shift(right);
}

#endif // HAVE_OPENCL

#ifdef WITH_CUBLAS

template <typename DataType>
device_ptr<DataType, StorageType::CUDA>::device_ptr() 
    : ptr(NULL), deviceID(0)
{
}

template <typename DataType>
device_ptr<DataType, StorageType::CUDA>::device_ptr(DataType *_ptr, int _deviceID) 
    : ptr(_ptr), deviceID(_deviceID)
{
}

template <typename DataType>
device_ptr<DataType, StorageType::CUDA>::device_ptr(const device_ptr<typename std::add_const<DataType>::type, StorageType::CUDA> &other)
    : ptr(other.get()), deviceID(other.getDeviceID())
{
}

template <typename DataType>
device_ptr<DataType, StorageType::CUDA>::device_ptr(const device_ptr<typename std::remove_const<DataType>::type, StorageType::CUDA> &other)
    : ptr(other.get()), deviceID(other.getDeviceID())
{
}

template <typename DataType>
DataType *
device_ptr<DataType, StorageType::CUDA>::get() const
{
    return ptr;
}

template <typename DataType>
device_ptr<DataType, StorageType::CUDA>
device_ptr<DataType, StorageType::CUDA>::shift(int dist)
{
    return device_ptr<DataType, StorageType::CUDA>(ptr+dist, deviceID);
}

template <typename DataType>
const device_ptr<DataType, StorageType::CUDA>
device_ptr<DataType, StorageType::CUDA>::shift(int dist) const
{
    return device_ptr<DataType, StorageType::CUDA>(ptr+dist, deviceID);
}

template <typename DataType>
int
device_ptr<DataType, StorageType::CUDA>::getDeviceID() const
{
    return deviceID;
}

template <typename DataTypeThis, typename DataTypeThat>
bool operator == (const device_ptr<DataTypeThis, StorageType::CUDA> & compareThis, 
                  const device_ptr<DataTypeThat, StorageType::CUDA> & compareThat) {
    return (reinterpret_cast<void*>(compareThis.get())==reinterpret_cast<void*>(compareThat.get()) 
            && compareThis.getDeviceID()==compareThat.getDeviceID());
}

template <typename DataTypeThis, typename DataTypeThat>
bool operator != (const device_ptr<DataTypeThis, StorageType::CUDA> & compareThis, 
                  const device_ptr<DataTypeThat, StorageType::CUDA> & compareThat) {
    return (reinterpret_cast<void*>(compareThis.get())!=reinterpret_cast<void*>(compareThat.get())  
            || compareThis.getDeviceID()!=compareThat.getDeviceID());
}

template <typename DataTypeThis, typename IndexType>
device_ptr<DataTypeThis, StorageType::CUDA>
operator+ (const device_ptr<DataTypeThis, StorageType::CUDA> &left, IndexType right)
{
    return left.shift(right);
}
#endif // WITH_CUBLAS



}

#endif // PLAYGROUND_FLENS_AUXILIARY_DEVICEPTR_TCC
