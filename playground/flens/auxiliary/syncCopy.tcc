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

#ifndef PLAYGROUND_FLENS_AUXILIARY_SYNCCOPY_TCC
#define PLAYGROUND_FLENS_AUXILIARY_SYNCCOPY_TCC 1

namespace flens {

inline void 
enableSyncCopy()
{
#ifdef HAVE_CUBLAS
    CudaEnv::enableSyncCopy();
#endif
#ifdef HAVE_OPENCL
    OpenCLEnv::enableSyncCopy();
#endif
}

inline void 
disableSyncCopy()
{
#ifdef HAVE_CUBLAS
    CudaEnv::disableSyncCopy();
#endif
#ifdef HAVE_OPENCL
    OpenCLEnv::disableSyncCopy();
#endif 
}

inline void 
enableASyncCopy()
{
#ifdef HAVE_CUBLAS
    CudaEnv::disableSyncCopy();
#endif
#ifdef HAVE_OPENCL
    OpenCLEnv::disableSyncCopy();
#endif
}

inline void 
disableASyncCopy()
{
#ifdef HAVE_CUBLAS
    CudaEnv::enableSyncCopy();
#endif
#ifdef HAVE_OPENCL
    OpenCLEnv::enableSyncCopy();
#endif 
}

inline bool
isSyncCopyEnabled(){
#ifdef HAVE_CUBLAS
    return CudaEnv::isSyncCopyEnabled();
#endif
#ifdef HAVE_OPENCL
    return OpenCLEnv::isSyncCopyEnabled();
#endif 
    ASSERT(0);
    return false;
}
    
inline bool
isASyncCopyEnabled(){
#ifdef HAVE_CUBLAS
    return !(CudaEnv::isSyncCopyEnabled());
#endif
#ifdef HAVE_OPENCL
    return !(OpenCLEnv::isSyncCopyEnabled());
#endif
    ASSERT(0);
    return false;
}
    
} //namespace flens

#endif // PLAYGROUND_FLENS_AUXILIARY_SYNCCOPY_TCC
