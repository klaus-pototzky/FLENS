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

#ifndef PLAYGROUND_CXXBLAS_LEVEL2_TRSV_TCC
#define PLAYGROUND_CXXBLAS_LEVEL2_TRSV_TCC 1

#include <cxxblas/cxxblas.h>

namespace cxxblas {

#ifdef HAVE_CLBLAS

// strsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<float, flens::StorageType::OpenCL> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasStrsv");

    cl_int status = CLBLAS_IMPL(Strsv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                       0, NULL, NULL);

    flens::checkStatus(status);
    
}

// dtrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<double, flens::StorageType::OpenCL> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasDtrsv");

    cl_int status = CLBLAS_IMPL(Dtrsv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

// ctrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCtrsv");

    cl_int status = CLBLAS_IMPL(Ctrsv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                       CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                       n,
                                       A.get(), A.getOffset(), ldA,
                                       x.get(), x.getOffset(), incX,
                                       1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}

// ztrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA,
     flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZtrsv");
    
    cl_int status = CLBLAS_IMPL(Ztrsv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(upLo),
                                        CLBLAS::getClblasType(transA), CLBLAS::getClblasType(diag),
                                        n,
                                        A.get(), A.getOffset(), ldA,
                                        x.get(), x.getOffset(), incX,
                                        1, flens::OpenCLEnv::getQueuePtr(),
                                        0, NULL, NULL);

    flens::checkStatus(status);
}
    
#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// strsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const float, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<float, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasStrsv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        trsv(ColMajor, upLo, transA, diag, n, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasStrsv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n,
                                        A.get(), ldA,
                                        x.get(), incX);
    
    flens::checkStatus(status);
}

// dtrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const double, flens::StorageType::CUDA> A, IndexType ldA,
      flens::device_ptr<double, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasDtrsv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        trsv(ColMajor, upLo, transA, diag, n, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasDtrsv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n,
                                        A.get(), ldA,
                                        x.get(), incX);
    
    flens::checkStatus(status);
}
// ctrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasCtrsv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        trsv(ColMajor, upLo, transA, diag, n, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasCtrsv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n,
                                        reinterpret_cast<const cuFloatComplex*>(A.get()), ldA,
                                        reinterpret_cast<cuFloatComplex*>(x.get()), incX);
    
    flens::checkStatus(status);
}

// ztrsv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
trsv(StorageOrder order, StorageUpLo upLo,
     Transpose transA, Diag diag,
     IndexType n,
     const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA,
     flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> x, IndexType incX)
{
    CXXBLAS_DEBUG_OUT("cublasZtrsv");
    
    if (order==RowMajor) {
        transA = Transpose(transA^Trans);
        upLo = (upLo==Upper) ? Lower : Upper;
        trsv(ColMajor, upLo, transA, diag, n, A, ldA, x, incX);
        return;
    }
    cublasStatus_t status = cublasZtrsv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(upLo), CUBLAS::getCublasType(transA),
                                        CUBLAS::getCublasType(diag),
                                        n,
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()), ldA,
                                        reinterpret_cast<cuDoubleComplex*>(x.get()), incX);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS


} // namespace flens

#endif // PLAYGROUND_CXXBLAS_LEVEL2_TRSV_TCC
