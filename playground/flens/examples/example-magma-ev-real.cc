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

    typedef double                              T;

    ///
    /// Usual typedef, better performance for
    /// copy operations is achieved with
    ///
    /// PinnedHostFullStorage and PinnedHostDeviceArray
    ///
    typedef GeMatrix<FullStorage<T> >           HostMatrix;
    typedef DenseVector<Array<T> >              HostVector;
    typedef HostMatrix::IndexType               IndexType;


    ///
    /// Then we setup another toy problem
    ///
    const IndexType n = 5;

    HostMatrix         A_host(n,n),VL(n,n),VR(n,n);
    HostVector         wr(n), wi(n);

    ///
    /// Initialize MAGMA
    ///

    magma_init();

    A_host =  2,   3,  -1,   0,  2,
             -6,  -5,   0,   2, -6,
              2,  -5,   6,  -6,  2,
              2,   3,  -1,   0,  8,
             -6,  -5,  10,   2, -6;



    ///
    /// (1) Calculate it with lapack
    /// (temporary workspace is allocated automatically)
    ///
    lapack::ev(true,true,A_host,wr,wi,VL,VR);

    cout << "wr = " << wr << endl;
    cout << "wi = " << wi << endl;
    cout << "VL = " << VL << endl;
    cout << "VR = " << VR << endl;

    // Restore Matrix and Vector

    A_host =  2,   3,  -1,   0,  2,
             -6,  -5,   0,   2, -6,
              2,  -5,   6,  -6,  2,
              2,   3,  -1,   0,  8,
             -6,  -5,  10,   2, -6;

    wr = 0;
    wi = 0;
    VL = 0;
    VR = 0;


    ///
    /// (2) Solve the problem on the GPU 
    /// (temporary workspace is allocated automatically)
    ///
    ///

    magma::ev(true,true,A_host,wr,wi,VL,VR);

    cout << "wr = " << wr << endl;
    cout << "wi = " << wi << endl;
    cout << "VL = " << VL << endl;
    cout << "VR = " << VR << endl;

    ///
    /// Finalize MAGMA
    ///

    magma_finalize();
    return 0;
}
