///
/// Enable features for CUBLAS
///
#define USE_PLAYGROUND
///
/// Enable features for CUBLAS
///
#define WITH_CUBLAS

///
/// Enable features for MAGMA
///
#define USE_CXXMAGMA

///
/// Singular value decomposition not yet
/// implemented in FLENS-LAPACK
/// -> We need an external LAPACK implementation
///
#define USE_CXXLAPACK

///
/// Set flag for main file (only necessary once in a project)
/// Ensures internal global variables are only set once
///
#define MAIN_FILE

///
/// Include FLENS as usual
///
#include <flens/flens.cxx>

using namespace std;
using namespace flens;

int main() {

    typedef double                              PT;
    typedef complex<PT>                         T;

    ///
    /// Usual typedef, better performance for
    /// copy operations is achieved with
    ///
    /// PinnedHostFullStorage and PinnedHostDeviceArray
    ///
    typedef GeMatrix<FullStorage<T> >           HostComplexMatrix;
    typedef DenseVector<Array<PT> >             HostRealVector;
    typedef HostComplexMatrix::IndexType        IndexType;


    ///
    /// Then we setup another toy problem
    ///
    const IndexType n = 5;

    HostComplexMatrix         A_host(n,n),U(n,n),VT(n,n);
    HostRealVector            s(n);

    ///
    /// Initialize MAGMA
    ///

    magma_init();

    A_host =  T(2,3),   T(3,3),  T(-1,7),  T(0,2),  T(2,-1),
              T(-6,0),  T(-5,3), T(0,2),   T(2,0),  T(-6,6),
              T(2,2),   T(3,1),  T(-1,7),  T(0,2),  T(2,-1),
              T(-2,0),  T(-5,1), T(0,2),   T(2,1),  T(6,-6),
              T(4,1),   T(-1,1), T(-1,7),  T(-1,2), T(2,-1);



    ///
    /// (1) Calculate it with lapack
    /// (temporary workspace is allocated automatically)
    ///
    lapack::svd(lapack::SVD::All, lapack::SVD::All, A_host, s, U, VT);

    cout << "s = " << s << endl;
    cout << "U = " << U << endl;
    cout << "VT = " << VT << endl;

    // Restore Matrix and Vector

    A_host =  T(2,3),   T(3,3),  T(-1,7),  T(0,2),  T(2,-1),
              T(-6,0),  T(-5,3), T(0,2),   T(2,0),  T(-6,6),
              T(2,2),   T(3,1),  T(-1,7),  T(0,2),  T(2,-1),
              T(-2,0),  T(-5,1), T(0,2),   T(2,1),  T(6,-6),
              T(4,1),   T(-1,1), T(-1,7),  T(-1,2), T(2,-1);

    s  = PT(0);
    U  = T(0);
    VT = T(0);


    ///
    /// (2) Solve the problem on the GPU
    /// (temporary workspace is allocated automatically)
    ///
    ///

    magma::svd(magma::SVD::All, magma::SVD::All, A_host, s, U, VT);

    cout << "s = " << s << endl;
    cout << "U = " << U << endl;
    cout << "VT = " << VT << endl;

    ///
    /// Finalize MAGMA
    ///

    magma_finalize();
    return 0;
}
