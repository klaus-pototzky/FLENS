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

#ifndef PLAYGROUND_CXXBLAS_LEVEL2_GEMV_TCC
#define PLAYGROUND_CXXBLAS_LEVEL2_GEMV_TCC 1

#include <complex>
#include <cxxblas/cxxblas.h>

namespace cxxblas {


#ifdef HAVE_CLBLAS

// sgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     float alpha,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> A, IndexType ldA,
     const flens::device_ptr<const float, flens::StorageType::OpenCL> x, IndexType incX,
     float beta,
     flens::device_ptr<float, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasSgemv");

    cl_int status = CLBLAS_IMPL_EX(Sgemv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(trans),
                                          m,  n,
                                          alpha,
                                          A.get(), A.getOffset(), ldA,
                                          x.get(), x.getOffset(), incX,
                                          beta,
                                          y.get(), y.getOffset(), incY,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
}

// dgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     double alpha,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> A, IndexType ldA,
     const flens::device_ptr<const double, flens::StorageType::OpenCL> x, IndexType incX,
     double beta,
     flens::device_ptr<double, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasDgemv");

    cl_int status = CLBLAS_IMPL_EX(Dgemv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(trans),
                                          m,  n,
                                          alpha,
                                          A.get(), A.getOffset(), ldA,
                                          x.get(), x.getOffset(), incX,
                                          beta,
                                          y.get(), y.getOffset(), incY,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
}

// cgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
     IndexType m, IndexType n,
     ComplexFloat alpha,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> A, IndexType ldA,
     const flens::device_ptr<const ComplexFloat, flens::StorageType::OpenCL> x, IndexType incX,
     ComplexFloat beta,
     flens::device_ptr<ComplexFloat, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasCgemv");
    
    if (trans==Conj) {
        order  = (order==RowMajor) ? ColMajor : RowMajor;
        gemv(order, ConjTrans, n, m, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }
    
    cl_int status = CLBLAS_IMPL_EX(Cgemv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(trans),
                                          m,  n,
                                          *(reinterpret_cast<cl_double2*>(&alpha)),
                                          A.get(), A.getOffset(), ldA,
                                          x.get(), x.getOffset(), incX,
                                          *(reinterpret_cast<cl_double2*>(&beta)),
                                          y.get(), y.getOffset(), incY,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
}

// zgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
      IndexType m, IndexType n,
      ComplexDouble alpha,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> A, IndexType ldA,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::OpenCL> x, IndexType incX,
      ComplexDouble beta,
      flens::device_ptr<ComplexDouble, flens::StorageType::OpenCL> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("clAmdBlasZgemv");
    
    if (trans==Conj) {
        order  = (order==RowMajor) ? ColMajor : RowMajor;
        gemv(order, ConjTrans, n, m, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }

    cl_int status = CLBLAS_IMPL_EX(Zgemv)(CLBLAS::getClblasType(order), CLBLAS::getClblasType(trans),
                                          m,  n,
                                          *(reinterpret_cast<cl_double2*>(&alpha)),
                                          A.get(), A.getOffset(), ldA,
                                          x.get(), x.getOffset(), incX,
                                          *(reinterpret_cast<cl_double2*>(&beta)),
                                          y.get(), y.getOffset(), incY,
                                          1, flens::OpenCLEnv::getQueuePtr(),
                                          0, NULL, NULL);

    flens::checkStatus(status);
}

#endif // HAVE_CLBLAS

#ifdef HAVE_CUBLAS

// sgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
      IndexType m, IndexType n,
      float alpha,
      const flens::device_ptr<const float, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const float, flens::StorageType::CUDA> x, IndexType incX,
      float beta,
      flens::device_ptr<float, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasSgemv");
    
    if (order==RowMajor) {
        trans = Transpose(trans^Trans);
        gemv(ColMajor, trans, n, m, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }

    cublasStatus_t status = cublasSgemv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(trans),
                                        m,  n,
                                        &alpha,
                                        A.get(), ldA,
                                        x.get(), incX,
                                        &beta,
                                        y.get(), incY);
    
    flens::checkStatus(status);
}

// dgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
      IndexType m, IndexType n,
      double alpha,
      const flens::device_ptr<const double, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const double, flens::StorageType::CUDA> x, IndexType incX,
      double beta,
      flens::device_ptr<double, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasDgemv");
    
    if (order==RowMajor) {
        trans = Transpose(trans^Trans);
        gemv(ColMajor, trans, n, m, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }

    cublasStatus_t status = cublasDgemv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(trans),
                                        m,  n,
                                        &alpha,
                                        A.get(), ldA,
                                        x.get(), incX,
                                        &beta,
                                        y.get(), incY);
    
    flens::checkStatus(status);
}

// cgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
      IndexType m, IndexType n,
      const ComplexFloat &alpha,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const ComplexFloat, flens::StorageType::CUDA> x, IndexType incX,
      const ComplexFloat &beta,
      flens::device_ptr<ComplexFloat, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasCgemv");
    
    if (order==RowMajor) {
        trans = Transpose(trans^Trans);
        gemv(ColMajor, trans, n, m, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }

    cublasStatus_t status = cublasCgemv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(trans),
                                        m,  n,
                                        reinterpret_cast<const cuFloatComplex*>(&alpha),
                                        reinterpret_cast<const cuFloatComplex*>(A.get()), ldA,
                                        reinterpret_cast<const cuFloatComplex*>(x.get()), incX,
                                        reinterpret_cast<const cuFloatComplex*>(&beta),
                                        reinterpret_cast<cuFloatComplex*>(y.get()), incY);
    
    flens::checkStatus(status);
}

// zgemv
template <typename IndexType>
typename If<IndexType>::isBlasCompatibleInteger
gemv(StorageOrder order, Transpose trans,
      IndexType m, IndexType n,
      const ComplexDouble &alpha,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> A, IndexType ldA,
      const flens::device_ptr<const ComplexDouble, flens::StorageType::CUDA> x, IndexType incX,
      const ComplexDouble &beta,
      flens::device_ptr<ComplexDouble, flens::StorageType::CUDA> y, IndexType incY)
{
    CXXBLAS_DEBUG_OUT("cublasZgemv");
    
    if (order==RowMajor) {
        trans = Transpose(trans^Trans);
        gemv(ColMajor, trans, n, m, alpha, A, ldA,
             x, incX, beta, y, incY);
        return;
    }

    cublasStatus_t status = cublasZgemv(flens::CudaEnv::getHandle(), 
                                        CUBLAS::getCublasType(trans),
                                        m,  n,
                                        reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                        reinterpret_cast<const cuDoubleComplex*>(A.get()), ldA,
                                        reinterpret_cast<const cuDoubleComplex*>(x.get()), incX,
                                        reinterpret_cast<const cuDoubleComplex*>(&beta),
                                        reinterpret_cast<cuDoubleComplex*>(y.get()), incY);
    
    flens::checkStatus(status);
}

#endif // HAVE_CUBLAS

} // namespace cxxblas

#endif // PLAYGROUND_CXXBLAS_LEVEL2_GEMV_TCC
