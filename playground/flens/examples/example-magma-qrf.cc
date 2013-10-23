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

    ///
    ///  Define an underscore operator for convenient matrix slicing
    ///
    typedef HostMatrix::IndexType   IndexType;
    const Underscore<IndexType>   _;

    const IndexType m = 4,
                    n = 5;

    HostMatrix    Ab_host(m, n);
    DeviceMatrix  Ab_device(m, n);

    Ab_host =  2,   3,  -1,   0,   20,
              -6,  -5,   0,   2,  -33,
               2,  -5,   6,  -6,  -43,
               4,   6,   2,  -3,   49;


    HostVector   tau_for_host(std::min(m,n)),  tau_for_device(std::min(m,n));
    HostVector   work_host;
    DeviceVector work_device;

    const auto A_host   = Ab_host(_,_(1,m));
    auto       B_host   = Ab_host(_,_(m+1,n));

    const auto A_device = Ab_device(_,_(1,m));
    auto       B_device = Ab_device(_,_(m+1,n));

    ///
    /// Initialize MAGMA
    ///

    magma_init();

    ///
    ///  (1) Compute the QR factorization with LAPACK
    ///      and solve linear equation
    ///
    lapack::qrf(Ab_host, tau_for_host, work_host);

    blas::sm(Left, NoTrans, 1, A_host.upper(), B_host);

    cout << "X = " << B_host << endl;


    ///
    /// Restore values
    ///
    work_host.resize(0);
    Ab_host =  2,   3,  -1,   0,   20,
              -6,  -5,   0,   2,  -33,
               2,  -5,   6,  -6,  -43,
               4,   6,   2,  -3,   49;


    ///
    ///  (2) Compute the QR factorization with MAGMA
    ///      and solve linear equation
    ///      Transfers are done by MAGMA
    ///
    magma::qrf(Ab_host, tau_for_host, work_host);

    blas::sm(Left, NoTrans, 1, A_host.upper(), B_host);

    cout << "X = " << B_host << endl;


    ///
    /// Restore values
    ///

    Ab_host =  2,   3,  -1,   0,   20,
              -6,  -5,   0,   2,  -33,
               2,  -5,   6,  -6,  -43,
               4,   6,   2,  -3,   49;


    ///
    ///  (3) Compute the QR factorization with MAGMA
    ///      and solve linear equation
    ///      Transfers are done manually
    ///
    Ab_device = Ab_host;

    magma::qrf(Ab_device, tau_for_device, work_device);

    blas::sm(Left, NoTrans, 1, A_device.upper(), B_device);

    B_host = B_device;
    cout << "X = " << B_host << endl;

    ///
    /// Finalize MAGMA
    ///

    magma_finalize();



    return 0;
}
