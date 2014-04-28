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
    typedef DenseVector<Array<T> >              HostVector;
    typedef HostMatrix::IndexType               IndexType;


    ///
    /// Then we setup another toy problem
    ///
    const IndexType n = 5;

    HostMatrix         A_host(n,n),VL(n,n),VR(n,n);
    HostVector         w(n);

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
    lapack::ev(true,true,A_host,w,VL,VR);

    cout << "w = " << w << endl;
    cout << "VL = " << VL << endl;
    cout << "VR = " << VR << endl;

    // Restore Matrix and Vector

    A_host =  T(2,3),   T(3,3),  T(-1,7),  T(0,2),  T(2,-1),
              T(-6,0),  T(-5,3), T(0,2),   T(2,0),  T(-6,6),
              T(2,2),   T(3,1),  T(-1,7),  T(0,2),  T(2,-1),
              T(-2,0),  T(-5,1), T(0,2),   T(2,1),  T(6,-6),
              T(4,1),   T(-1,1), T(-1,7),  T(-1,2), T(2,-1);

    w  = T(0);
    VL = T(0);
    VR = T(0);


    ///
    /// (2) Solve the problem on the GPU
    /// (temporary workspace is allocated automatically)
    ///
    ///

    magma::ev(true,true,A_host,w,VL,VR);

    cout << "w = " << w << endl;
    cout << "VL = " << VL << endl;
    cout << "VR = " << VR << endl;

    ///
    /// Finalize MAGMA
    ///

    magma_finalize();
    return 0;
}
