#define USE_PLAYGROUND

///
/// Enable features for CUBLAS
///
#define WITH_CUBLAS

///
/// The same code works with OpenCL
///
/// OpenCL code can run on the CPU
/// -> set #define OPENCL_ON_CPU
///
///#define WITH_CLAMDBLAS // <- clAmdBlas available
///#define WITH_CLBLAS    // <- clBlas available


///
/// Set flag for main file (only necessary once in a project)
/// Ensures internal global variables are only set once
///
#define MAIN_FILE

///
/// Include FLENS as usual
///
#include <flens/flens.cxx>

///
/// First minimal example with GPU (more a proof of concept)
///
/// Working: BLAS Level 1 functions 
///
/// Not yet working (will require some hacks):
/// - operations like y = 1, y+=1, y-=1, ...  
///
using namespace std;
using namespace flens;

int main() {

	///
	/// Print info about the Device
	///
	cout << getInfo() << endl;

    int N = 4;

    ///
    /// Usual typedefs
    ///
    typedef double                        PT;
    typedef complex<PT>                   T;
    typedef DenseVector<Array<T > >       HostVector;
    typedef DenseVector<DeviceArray<T > > DeviceVector;

    const Underscore<int> _;
    HostVector   x_host(N),   z_host(2*N);
    DeviceVector x_device(N), z_device(2*N);

    auto z_view1 = z_device(_(1  ,N));
    auto z_view2 = z_device(_(N+1,2*N));

    x_host = T(-1), T(2), T(3), T(4);

    // Copy to GPU
    x_device = x_host;

    // Perform some operations on the GPU
    z_view1 = T(2.0)*x_device;
    z_view2 = T(3.0)*x_device;

    blas::swap(z_view1, z_view2);

    x_device = z_view1 + z_view2;

    // Return values are on the CPU, corresponding stream is synchronized
    cout << "x^T*x = " << blas::dot(z_view1, z_view1) << endl;
    cout << "|x|_2 = " << blas::nrm2(z_view1) << endl;
    cout << "|x|_1 = " << blas::asum(z_view1) << endl;
    cout << "iamax(x) = " << blas::iamax(z_view1) << endl;


    // Copy back to RAM
    x_host = x_device;
    cout << x_host << endl;

    ///
    /// Not working by design
    ///

    //
    //cout << x_device << endl;    //(direct element access fails)
    // x_host += x_device;         //(add GPU vector to Host vector and vice versa)
    // ...                         //(any mixture of GPU and Host vector, except copy)

    return 0;
}
