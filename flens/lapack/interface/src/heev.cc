#include <flens/lapack/interface/include/config.h>


namespace flens { namespace lapack {

extern "C" {

//-- zheev ---------------------------------------------------------------------
void
LAPACK_DECL(zheev)(const char           *JOBZ,
                   const char           *UPLO,
                   const INTEGER        *N,
                   DOUBLE_COMPLEX       *A,
                   const INTEGER        *LDA,
                   DOUBLE               *W,
                   DOUBLE_COMPLEX       *WORK,
                   const INTEGER        *LWORK,
                   DOUBLE               *RWORK,
                   INTEGER              *INFO)
{
    using std::max;
    using std::min;
//
//  Test the input parameters so that we pass LAPACK error checks
//
    *INFO = 0;
    bool lQuery = (*LWORK==-1);
    bool wantZ = (*JOBZ=='V');
    bool lower = (*UPLO=='L');

    if ((!wantZ) && (*JOBZ!='N')) {
        *INFO = 1;
    } else if ((!lower) && (*UPLO!='U')) {
        *INFO = 2;
    } else if (*N<0) {
        *INFO = 3;
    } else if (*LDA<max(INTEGER(1),*N)) {
        *INFO = 5;
    } else if (*LWORK<max(INTEGER(1), 2*(*N)-1) && !lQuery) {
        *INFO = 8;
    }

    if (*INFO!=0) {
        LAPACK_ERROR("ZHEEV", INFO);
        *INFO = -(*INFO);
        return;
    }

//
//  Setup FLENS matrix/vector types
//
    auto zA     = reinterpret_cast<CXX_DOUBLE_COMPLEX *>(A);
    auto zWORK  = reinterpret_cast<CXX_DOUBLE_COMPLEX *>(WORK);

    ZHeMatrixView       _A(ZFSView(*N, *N, zA, *LDA), lower ? Lower : Upper);
    DDenseVectorView    _W      = DArrayView(*N, W, 1);
    ZDenseVectorView    _WORK   = ZArrayView(*LWORK, zWORK, 1);
    DDenseVectorView    _RWORK  = DArrayView(3*(*N)-2, RWORK, 1);

//
//  Call FLENS implementation
//
    ev(wantZ, _A, _W, _WORK, _RWORK);
}

} // extern "C"

} } // namespace lapack, flens
