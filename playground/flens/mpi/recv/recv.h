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

#ifndef PLAYGROUND_FLENS_MPI_RECV_RECV_H
#define PLAYGROUND_FLENS_MPI_RECV_RECV_H 1

#ifdef WITH_MPI
#    include "mpi.h"
#endif

#include<vector>
#include<playground/flens/mpi/types.h>


namespace flens { namespace mpi {

#ifdef WITH_MPI

template <typename T>
    typename RestrictTo<MPI_Type<T>::Compatible,
                        T>::Type
    MPI_recv(const int source,
             const MPI::Comm &communicator = MPI::COMM_WORLD);

template <typename T>
    typename RestrictTo<MPI_Type<T>::Compatible,
                        void>::Type
    MPI_recv(T &x, const int source,
             const MPI::Comm &communicator = MPI::COMM_WORLD);

template <typename IndexType, typename T>
    typename RestrictTo<MPI_Type<T>::Compatible,
                        void>::Type
    MPI_recv(const IndexType n, T *x, const int source,
             const MPI::Comm &communicator = MPI::COMM_WORLD);

template <typename VX>
    typename RestrictTo<IsDenseVector<VX>::value,
                        void>::Type
    MPI_recv(VX &&x, const int source,
             const MPI::Comm &communicator = MPI::COMM_WORLD);

template <typename MA>
    typename RestrictTo<IsGeMatrix<MA>::value,
                        void>::Type
    MPI_recv(MA &&A, const int source,
             const MPI::Comm &communicator = MPI::COMM_WORLD);

template <typename X>
    typename RestrictTo<IsDenseVector<X>::value ||
                        IsGeMatrix<X>::value,
                        void>::Type
    MPI_recv_all(std::vector<X> &x,
                 const MPI::Comm &communicator = MPI::COMM_WORLD);

#endif
} }

#endif // PLAYGROUND_FLENS_MPI_RECV_RECV_H
