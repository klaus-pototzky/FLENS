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

    HostVector         work_host;
    DeviceVector       work_device;

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
    lapack::trf(A_host, piv_for_host);
    lapack::trs(NoTrans, A_host, piv_for_host, b_host);
    lapack::tri(A_host, piv_for_host, work_host);

    cout << "b = " << b_host << endl;
    cout << "inv(A) = " << A_host << endl;

    ///
    /// (2) Calcuate the factorization on the GPU and
    ///     let MAGMA do the transfers
    ///

    // restore values
    A_host = T(1, 0), T( 1,-1), T(  2,20),
             T(0, 1), T( 1, 2), T(-10, 8),
             T(0,-1), T(-1, 1), T( 40, 6);

    b_host = T( 1, 0),
             T(-1, 1),
             T(-2,-1);

    magma::trf(A_host, piv_for_host);
    lapack::trs(NoTrans, A_host, piv_for_host, b_host);
    lapack::tri(A_host, piv_for_host, work_host);

    cout << "b = " << b_host << endl;
    cout << "inv(A) = " << A_host << endl;

    ///
    /// (3) Solve the problem on the GPU and
    ///     and do the transfers manually
    ///

    // restore values and copy them to the GPU
    A_host = T(1, 0), T( 1,-1), T(  2,20),
             T(0, 1), T( 1, 2), T(-10, 8),
             T(0,-1), T(-1, 1), T( 40, 6);

    b_host = T( 1, 0),
             T(-1, 1),
             T(-2,-1);

    A_device = A_host;
    b_device = b_host;

    magma::trf(A_device, piv_for_device);
    magma::trs(NoTrans, A_device, piv_for_device, b_device);
    magma::tri(A_device, piv_for_device, work_device);

    // copy result to host
    b_host = b_device;
    A_host = A_device;

    cout << "b = " << b_host << endl;
    cout << "inv(A) = " << A_host << endl;

    ///
    /// Finalize MAGMA
    ///

    magma_finalize();
    return 0;
}
