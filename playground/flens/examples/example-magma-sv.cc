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
    typedef GeMatrix<FullStorage<T> >           HostMatrix;
    typedef GeMatrix<DeviceFullStorage<T> >     DeviceMatrix;
    typedef DenseVector<Array<T> >              HostVector;
    typedef DenseVector<DeviceArray<T> >        DeviceVector;
    typedef HostMatrix::IndexType               IndexType;
    typedef DenseVector<Array<IndexType> >      IndexVector;

    ///
    /// Then we setup another toy problem
    ///
    const IndexType n = 3;

    HostMatrix         A_host(n,n);
    DeviceMatrix       A_device(n,n);
    HostVector         b_host(n);
    DeviceVector       b_device(n);

    ///
    /// Initialize MAGMA
    ///

    magma_init();

    ///
    /// Pivots vectors are *always* on Host
    ///
    IndexVector        piv_for_host(n), piv_for_device(n);

    A_host = T(1, 0), T( 1,-1), T(  2,20),
             T(0, 1), T( 1, 2), T(-10, 8),
             T(0,-1), T(-1, 1), T( 40, 6);

    b_host = T( 1, 0),
             T(-1, 1),
             T(-2,-1);



    ///
    /// (1) Calculate it with lapack
    ///
    lapack::sv(A_host, piv_for_host, b_host);

    cout << "b = " << b_host << endl;

    // Restore Matrix and Vector

    A_host = T(1, 0), T( 1,-1), T(  2,20),
             T(0, 1), T( 1, 2), T(-10, 8),
             T(0,-1), T(-1, 1), T( 40, 6);

    b_host = T( 1, 0),
             T(-1, 1),
             T(-2,-1);

    ///
    /// (2) Solve the problem on the GPU and
    ///     and let MAGMA do the transfers
    ///

    magma::sv(A_host, piv_for_host, b_host);

    cout << "b = " << b_host << endl;

    // Restore Matrix and Vector

    A_host = T(1, 0), T( 1,-1), T(  2,20),
             T(0, 1), T( 1, 2), T(-10, 8),
             T(0,-1), T(-1, 1), T( 40, 6);

    b_host = T( 1, 0),
             T(-1, 1),
             T(-2,-1);

    ///
    /// (3) Solve the problem on the GPU and
    ///     and do the transfers manually (already done above)
    ///

    A_device = A_host;
    b_device = b_host;

    magma::sv(A_device, piv_for_device, b_device);

    // copy result to host
    b_host = b_device;

    cout << "b = " << b_host << endl;

    ///
    /// Finalize MAGMA
    ///

    magma_finalize();
    return 0;
}
