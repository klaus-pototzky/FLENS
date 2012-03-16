#ifndef CXXLAPACK_NETLIB_NETLIB_H
#define CXXLAPACK_NETLIB_NETLIB_H 1

#ifdef LAPACK_IMPL
#   undef   LAPACK_IMPL
#endif
#define  LAPACK_IMPL(x)           x##_

extern "C" {
#   include <cxxlapack/netlib/interface/lapack.in.h>
} // extern "C"

#endif //  CXXLAPACK_NETLIB_NETLIB_H