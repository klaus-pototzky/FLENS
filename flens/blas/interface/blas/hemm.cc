#include <flens/blas/interface/blas/config.h>


using namespace flens;

extern "C" {

void
BLAS(chemm)(const char      *SIDE,
            const char      *UPLO,
            const INTEGER   *M,
            const INTEGER   *N,
            const cfloat    *ALPHA,
            const cfloat    *_A,
            const INTEGER   *LDA,
            const cfloat    *_B,
            const INTEGER   *LDB,
            const cfloat    *BETA,
            cfloat          *_C,
            const INTEGER   *LDC)
{

#   ifdef TEST_DIRECT_CBLAS

        char    _SIDE   = toupper(*SIDE);
        char    _UPLO   = toupper(*UPLO);

        Side           side   = Side(_SIDE);
        StorageUpLo    upLo   = StorageUpLo(_UPLO);

        cblas_chemm(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(side),
                    cxxblas::CBLAS::getCblasType(upLo),
                    *M, *N,
                    reinterpret_cast<const float *>(ALPHA),
                    reinterpret_cast<const float *>(_A), *LDA,
                    reinterpret_cast<const float *>(_B), *LDB,
                    reinterpret_cast<const float *>(BETA),
                    reinterpret_cast<float *>(_C), *LDC);

#   else

        using std::abs;
        using std::max;

        char    _SIDE = toupper(*SIDE);
        char    _UPLO = toupper(*UPLO);
        INTEGER nRowA = (_SIDE=='L') ? *M : *N;

#       ifndef NO_INPUT_CHECK
            INTEGER info    = 0;
            if (_SIDE!='L' && _SIDE!='R') {
                info = 1;
            } else if (_UPLO!='U' && _UPLO!='L') {
                info = 2;
            } else if (*M<0) {
                info = 3;
            } else if (*N<0) {
                info = 4;
            } else if (*LDA<max(INTEGER(1),nRowA)) {
                info = 7;
            } else if (*LDB<max(INTEGER(1),*M)) {
                info = 9;
            } else if (*LDC<max(INTEGER(1),*M)) {
                info = 12;
            }
            if (info!=0) {
                BLAS(xerbla)("CHEMM ", &info);
                return;
            }
#       endif

        Side           side = Side(_SIDE);
        StorageUpLo    upLo = StorageUpLo(_UPLO);
        const INTEGER  ka   = side==Left ? *M : *N;

        CHeMatrixConstView  A(CFullConstView(ka, ka, _A, *LDA), upLo);
        CGeMatrixConstView  B = CFullConstView(*M, *N, _B, *LDB);
        CGeMatrixView       C = CFullView(*M, *N, _C, *LDC);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha = *ALPHA;
            const auto beta  = *BETA;

            if (side==Left) {
                C = beta*C + alpha*A*B;
            } else if (side==Right) {
                C = beta*C + alpha*B*A;
            }
#       else
            blas::mm(side, *ALPHA, A, B, *BETA, C);
#       endif
#   endif
}


void
BLAS(zhemm)(const char      *SIDE,
            const char      *UPLO,
            const INTEGER   *M,
            const INTEGER   *N,
            const cdouble   *ALPHA,
            const cdouble   *_A,
            const INTEGER   *LDA,
            const cdouble   *_B,
            const INTEGER   *LDB,
            const cdouble   *BETA,
            cdouble         *_C,
            const INTEGER   *LDC)
{


#   ifdef TEST_DIRECT_CBLAS

        char    _SIDE   = toupper(*SIDE);
        char    _UPLO   = toupper(*UPLO);

        Side           side   = Side(_SIDE);
        StorageUpLo    upLo   = StorageUpLo(_UPLO);

        cblas_zhemm(CBLAS_ORDER::CblasColMajor,
                    cxxblas::CBLAS::getCblasType(side),
                    cxxblas::CBLAS::getCblasType(upLo),
                    *M, *N,
                    reinterpret_cast<const double *>(ALPHA),
                    reinterpret_cast<const double *>(_A), *LDA,
                    reinterpret_cast<const double *>(_B), *LDB,
                    reinterpret_cast<const double *>(BETA),
                    reinterpret_cast<double *>(_C), *LDC);
#   else

        using std::abs;
        using std::max;

        char    _SIDE = toupper(*SIDE);
        char    _UPLO = toupper(*UPLO);
        INTEGER nRowA = (_SIDE=='L') ? *M : *N;

#       ifndef NO_INPUT_CHECK
            INTEGER info    = 0;
            if (_SIDE!='L' && _SIDE!='R') {
                info = 1;
            } else if (_UPLO!='U' && _UPLO!='L') {
                info = 2;
            } else if (*M<0) {
                info = 3;
            } else if (*N<0) {
                info = 4;
            } else if (*LDA<max(INTEGER(1),nRowA)) {
                info = 7;
            } else if (*LDB<max(INTEGER(1),*M)) {
                info = 9;
            } else if (*LDC<max(INTEGER(1),*M)) {
                info = 12;
            }
            if (info!=0) {
                BLAS(xerbla)("ZHEMM ", &info);
                return;
            }
#       endif

        Side           side = Side(_SIDE);
        StorageUpLo    upLo = StorageUpLo(_UPLO);
        const INTEGER  ka   = side==Left ? *M : *N;

        ZHeMatrixConstView  A(ZFullConstView(ka, ka, _A, *LDA), upLo);
        ZGeMatrixConstView  B = ZFullConstView(*M, *N, _B, *LDB);
        ZGeMatrixView       C = ZFullView(*M, *N, _C, *LDC);

#       ifdef TEST_OVERLOADED_OPERATORS
            const auto alpha = *ALPHA;
            const auto beta  = *BETA;

            if (side==Left) {
                C = beta*C + alpha*A*B;
            } else if (side==Right) {
                C = beta*C + alpha*B*A;
            }
#       else
            blas::mm(side, *ALPHA, A, B, *BETA, C);
#       endif
#   endif
}

} // extern "C"
