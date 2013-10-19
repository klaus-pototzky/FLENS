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

#ifndef PLAYGROUND_CXXDFT_MULTIPLE_TCC
#define PLAYGROUND_CXXDFT_MULTIPLE_TCC 1

#include <cmath>
#include <string.h>
#include <cxxblas/typedefs.h>
#include <cxxblas/drivers/drivers.h>
#include <flens/auxiliary/auxiliary.h>
#include <playground/cxxdft/single.tcc>
#include <playground/cxxdft/direction.h>

namespace cxxdft {

template <typename IndexType, typename VIN, typename VOUT>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
             const VIN *x, IndexType strideX, IndexType distX,
             VOUT *y, IndexType strideY, IndexType distY,
             DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple");
    
    for (IndexType i=0; i<m; ++i) {
        dft_single_generic(n, x+i*distX, strideX, y+i*distY, strideY, direction);
    }
    
}

#ifdef HAVE_FFTW

#ifdef HAVE_FFTW_FLOAT

template <typename IndexType>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
             cxxblas::ComplexFloat *x, IndexType strideX, IndexType distX,
             cxxblas::ComplexFloat *y, IndexType strideY, IndexType distY,
             DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple [FFTW interface, double]");

#   if defined(FFTW_WISDOM_IMPORT) && !defined(WITH_MKLBLAS)
        ASSERT(strcmp(FFTW_WISDOM_FILENAME,""));
        fftwf_import_wisdom_from_filename(FFTW_WISDOM_FILENAME);
#   endif 

    fftwf_plan p = fftwf_plan_many_dft(1, &n, m,
                                      reinterpret_cast<fftwf_complex*>(x), NULL, strideX, distX,
                                      reinterpret_cast<fftwf_complex*>(y), NULL, strideY, distY,
                                      direction, FFTW_PLANNER_FLAG);
    fftwf_execute(p);
    
#   if defined(FFTW_WISDOM_EXPORT) && !defined(WITH_MKLBLAS)
        ASSERT(strcmp(FFTW_WISDOM_FILENAME,""));
        fftwf_export_wisdom_to_filename(FFTW_WISDOM_FILENAME);
#   endif
    
    fftwf_destroy_plan(p);
}
#endif // HAVE_FFTW_FLOAT

#ifdef HAVE_FFTW_DOUBLE

template <typename IndexType>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
             cxxblas::ComplexDouble *x, IndexType strideX, IndexType distX,
             cxxblas::ComplexDouble *y, IndexType strideY, IndexType distY,
             DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple [FFTW interface, double]");

#   if defined(FFTW_WISDOM_IMPORT) && !defined(WITH_MKLBLAS)
        ASSERT(strcmp(FFTW_WISDOM_FILENAME,""));
        fftw_import_wisdom_from_filename(FFTW_WISDOM_FILENAME);
#   endif 

    fftw_plan p = fftw_plan_many_dft(1, &n, m,
                                     reinterpret_cast<fftw_complex*>(x), NULL, strideX, distX,
                                     reinterpret_cast<fftw_complex*>(y), NULL, strideY, distY,
                                     direction, FFTW_PLANNER_FLAG);
    fftw_execute(p);
    
#  if defined(FFTW_WISDOM_EXPORT) && !defined(WITH_MKLBLAS)
        ASSERT(strcmp(FFTW_WISDOM_FILENAME,""));
        fftw_export_wisdom_to_filename(FFTW_WISDOM_FILENAME);
#   endif
    
    fftw_destroy_plan(p);
}

#endif // HAVE_FFTW_DOUBLE

#ifdef HAVE_FFTW_LONGDOUBLE

template <typename IndexType>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
             std::complex<long double> *x, IndexType strideX, IndexType distX,
             std::complex<long double> *y, IndexType strideY, IndexType distY,
             DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple [FFTW interface, long double]");
   
#   ifdef FFTW_WISDOM_IMPORT
        ASSERT(strcmp(FFTW_WISDOM_FILENAME,""));
        fftwl_import_wisdom_from_filename(FFTW_WISDOM_FILENAME);
#   endif 

    fftwl_plan p = fftwl_plan_many_dft(1, &n, m,
                                      reinterpret_cast<fftwl_complex*>(x), NULL, strideX, distX,
                                      reinterpret_cast<fftwl_complex*>(y), NULL, strideY, distY,
                                      direction, FFTW_PLANNER_FLAG);
    fftwl_execute(p);
    
#   ifdef FFTW_WISDOM_EXPORT
        ASSERT(strcmp(FFTW_WISDOM_FILENAME,""));
        fftwl_export_wisdom_to_filename(FFTW_WISDOM_FILENAME);
#   endif
    
    fftwl_destroy_plan(p);
}

#endif // HAVE_FFTW_LONGDOUBLE

#ifdef HAVE_FFTW_QUAD

template <typename IndexType>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
             std::complex<__float128> *x, IndexType strideX, IndexType distX,
             std::complex<__float128> *y, IndexType strideY, IndexType distY,
             DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple [FFTW interface, quad]");
    
#   ifdef FFTW_WISDOM_IMPORT
        ASSERT(strcmp(FFTW_WISDOM_FILENAME,""));
        fftwq_import_wisdom_from_filename(FFTW_WISDOM_FILENAME);
#   endif    

    fftwq_plan p = fftwq_plan_many_dft(1, &n, m,
                                      reinterpret_cast<fftwq_complex*>(x), NULL, strideX, distX,
                                      reinterpret_cast<fftwq_complex*>(y), NULL, strideY, distY,
                                      direction, FFTW_PLANNER_FLAG);
    fftwq_execute(p);
    
#   ifdef FFTW_WISDOM_EXPORT
        ASSERT(strcmp(FFTW_WISDOM_FILENAME,""));
        fftwq_export_wisdom_to_filename(FFTW_WISDOM_FILENAME);
#   endif

    fftwq_destroy_plan(p);
}

#endif // HAVE_FFTW_QUAD

#endif // HAVE_FFTW

#ifdef HAVE_CLFFT
template <typename IndexType>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
	    flens::device_ptr<cxxblas::ComplexFloat, flens::StorageType::OpenCL> x, IndexType strideX, IndexType distX, 
	    flens::device_ptr<cxxblas::ComplexFloat, flens::StorageType::OpenCL> y, IndexType strideY, IndexType distY,
	    DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple [CLFFT interface, complex float]");
    
    cl_int err;
    CLFFT_IMPL(PlanHandle) planHandle;
    CLFFT_IMPL(Dim) dim = CLFFT_1D;
    size_t clLengths = n;
    size_t iStride   = strideX;
    size_t oStride   = strideY;
    size_t iDist     = distX;
    size_t oDist     = distY;
    size_t batchSize = m;
    
    // Create a default plan for a complex FFT. 
    err = CLFFT_IMPL(CreateDefaultPlan) (&planHandle, flens::OpenCLEnv::getContext(), dim, &clLengths);
    flens::checkStatus(err); 
    
    // Set plan parameters.
    err = CLFFT_IMPL(SetPlanPrecision) (planHandle, CLFFT_SINGLE);
    flens::checkStatus(err);
    err = CLFFT_IMPL(SetLayout) (planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    flens::checkStatus(err);
    err = CLFFT_IMPL(SetResultLocation) (planHandle, CLFFT_OUTOFPLACE);
    flens::checkStatus(err);
    
    // Set strides
    err = CLFFT_IMPL(SetPlanInStride) (planHandle, CLFFT_1D, &iStride);
    flens::checkStatus(err);
    err = CLFFT_IMPL(SetPlanOutStride) (planHandle, CLFFT_1D, &oStride);
    flens::checkStatus(err);
    
    // Set distances
    err = CLFFT_IMPL(SetPlanDistance) (planHandle, iDist, oDist);
    flens::checkStatus(err);
    
    // set Batch number
    err = CLFFT_IMPL(SetPlanBatchSize) (planHandle, batchSize);
    flens::checkStatus(err);
    
    // Bake the plan. 
    err = CLFFT_IMPL(BakePlan) (planHandle, 1, flens::OpenCLEnv::getQueuePtr(), NULL, NULL);
    flens::checkStatus(err);

    if (direction==DFTDirection::Forward) {

      err = CLFFT_IMPL(EnqueueTransform) (planHandle, CLFFT_FORWARD, 1, flens::OpenCLEnv::getQueuePtr(), 0, NULL, NULL, &(x.get()), &(y.get()), NULL);
	
    } else {
        err = CLFFT_IMPL(SetPlanScale) (planHandle, CLFFT_BACKWARD, cl_float(1));
        flens::checkStatus(err); 
        err = CLFFT_IMPL(EnqueueTransform) (planHandle, CLFFT_BACKWARD, 1, flens::OpenCLEnv::getQueuePtr(), 0, NULL, NULL, &(x.get()), &(y.get()), NULL);
	
    }
    flens::checkStatus(err);

    err = CLFFT_IMPL(DestroyPlan) ( &planHandle );
    flens::checkStatus(err);
}

template <typename IndexType>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
	    flens::device_ptr<cxxblas::ComplexDouble, flens::StorageType::OpenCL> x, IndexType strideX, IndexType distX, 
	    flens::device_ptr<cxxblas::ComplexDouble, flens::StorageType::OpenCL> y, IndexType strideY, IndexType distY,
	    DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple [CLFFT interface, complex double]");
    
    cl_int err;
    CLFFT_IMPL(PlanHandle) planHandle;
    CLFFT_IMPL(Dim) dim = CLFFT_1D;
    size_t clLengths = n;
    size_t iStride   = strideX;
    size_t oStride   = strideY;
    size_t iDist     = distX;
    size_t oDist     = distY;
    size_t batchSize = m;
    
    // Create a default plan for a complex FFT. 
    err = CLFFT_IMPL(CreateDefaultPlan) (&planHandle, flens::OpenCLEnv::getContext(), dim, &clLengths);
    flens::checkStatus(err); 
    
    // Set plan parameters.
    err = CLFFT_IMPL(SetPlanPrecision) (planHandle, CLFFT_DOUBLE);
    flens::checkStatus(err);
    err = CLFFT_IMPL(SetLayout) (planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    flens::checkStatus(err);
    err = CLFFT_IMPL(SetResultLocation) (planHandle, CLFFT_OUTOFPLACE);
    flens::checkStatus(err);
    
    // Set strides
    err = CLFFT_IMPL(SetPlanInStride) (planHandle, CLFFT_1D, &iStride);
    flens::checkStatus(err);
    err = CLFFT_IMPL(SetPlanOutStride) (planHandle, CLFFT_1D, &oStride);
    flens::checkStatus(err);
    
    // Set distances
    err = CLFFT_IMPL(SetPlanDistance) (planHandle, iDist, oDist);
    flens::checkStatus(err);
    
    // set Batch number
    err = CLFFT_IMPL(SetPlanBatchSize) (planHandle, batchSize);
    flens::checkStatus(err);
    
    // Bake the plan. 
    err = CLFFT_IMPL(BakePlan) (planHandle, 1, flens::OpenCLEnv::getQueuePtr(), NULL, NULL);
    flens::checkStatus(err);

    if (direction==DFTDirection::Forward) {

        err = CLFFT_IMPL(EnqueueTransform) (planHandle, CLFFT_FORWARD, 1, flens::OpenCLEnv::getQueuePtr(), 0, NULL, NULL, &(x.get()), &(y.get()), NULL);
	
    } else {
        err = CLFFT_IMPL(SetPlanScale) (planHandle, CLFFT_BACKWARD, cl_double(1));
        flens::checkStatus(err); 
        err = CLFFT_IMPL(EnqueueTransform) (planHandle, CLFFT_BACKWARD, 1, flens::OpenCLEnv::getQueuePtr(), 0, NULL, NULL, &(x.get()), &(y.get()), NULL);
	
    }
    flens::checkStatus(err);

    err = CLFFT_IMPL(DestroyPlan) ( &planHandle );
    flens::checkStatus(err);
}
#endif // HAVE_CLFFT
    
#ifdef HAVE_CUFFT
    
template <typename IndexType>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
	    flens::device_ptr<cxxblas::ComplexFloat, flens::StorageType::CUDA> x, IndexType strideX, IndexType distX, 
	    flens::device_ptr<cxxblas::ComplexFloat, flens::StorageType::CUDA> y, IndexType strideY, IndexType distY,
	    DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple [CUFFT interface, complex float]");
    
    cufftHandle plan;
    cufftResult status;
     
    IndexType inembed = (n*strideX)*(m*distX);
    IndexType onembed = (n*strideY)*(m*distY);   
    status = cufftPlanMany(&plan, 1, &n, 
                           &inembed, strideX, distX,
                           &onembed, strideY, distY,
                           CUFFT_C2C, m);
    flens::checkStatus(status);

    status = cufftSetStream(plan, flens::CudaEnv::getStream());
    flens::checkStatus(status);

    if (direction==DFTDirection::Forward) {
        status = cufftExecC2C(plan, 
                              reinterpret_cast<cufftComplex *>(x.get()), 
                              reinterpret_cast<cufftComplex *>(y.get()), CUFFT_FORWARD);
    } else {
        status = cufftExecC2C(plan, 
                              reinterpret_cast<cufftComplex *>(x.get()), 
                              reinterpret_cast<cufftComplex *>(y.get()), CUFFT_INVERSE);
    }
    flens::checkStatus(status);
    cufftDestroy(plan);
}

template <typename IndexType>
typename cxxblas::If<IndexType>::isBlasCompatibleInteger
dft_multiple(IndexType n, IndexType m,
	    flens::device_ptr<cxxblas::ComplexDouble, flens::StorageType::CUDA> x, IndexType strideX, IndexType distX, 
	    flens::device_ptr<cxxblas::ComplexDouble, flens::StorageType::CUDA> y, IndexType strideY, IndexType distY,
	    DFTDirection direction)
{
    CXXBLAS_DEBUG_OUT("dft_multiple [CUFFT interface, complex double]");
    
    cufftHandle plan;
    cufftResult status;
    
    IndexType inembed = (n*strideX)*(m*distX);
    IndexType onembed = (n*strideY)*(m*distY);
    status = cufftPlanMany(&plan, 1, &n, 
                           &inembed, strideX, distX,
                           &onembed, strideY, distY,
                           CUFFT_Z2Z, m);
    flens::checkStatus(status);

    status = cufftSetStream(plan, flens::CudaEnv::getStream());
    flens::checkStatus(status);

    if (direction==DFTDirection::Forward) {
        status = cufftExecZ2Z(plan, 
                              reinterpret_cast<cufftDoubleComplex *>(x.get()), 
                              reinterpret_cast<cufftDoubleComplex *>(y.get()), CUFFT_FORWARD);
    } else {
        status = cufftExecZ2Z(plan, 
                              reinterpret_cast<cufftDoubleComplex *>(x.get()), 
                              reinterpret_cast<cufftDoubleComplex *>(y.get()), CUFFT_INVERSE);
    }
    flens::checkStatus(status);
    cufftDestroy(plan);
}
#endif // HAVE_CUFFT

} // namespace cxxdft

#endif // PLAYGROUND_CXXDFT_MULTIPLE_TCC
