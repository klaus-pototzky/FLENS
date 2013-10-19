#define USE_PLAYGROUND

///
/// Enable features for CUBLAS and CUFFT
///
#define WITH_CUBLAS
#define WITH_CUFFT

///
/// The same code works with OpenCL
///
/// OpenCL code can run on the CPU
/// -> set #define OPENCL_ON_CPU
///
///#define WITH_CLAMDBLAS // <- clAmdBlas available
///#define WITH_CLAMDFFT  // <- clAmdFFT available
///#define WITH_CLBLAS    // <- clBlas available
///#define WITH_CLFFT     // <- clFFT available

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

    int N = 4;

    ///
    /// Usual typedefs
    ///
    typedef double                                 PT;
    typedef std::complex<PT>                       T;

    typedef DenseVector<Array<T> >                 Vector;
    typedef DenseVector<DeviceArray<T> >           DeviceVector;

    Vector           x_host(N)  , y_host(N);
    DeviceVector     x_device(N), y_device(N);

    fillRandom(x_host);
    x_device = x_host;

    ///
    /// Perform Fourier transformation (forward)
    ///
    dft::dft_forward(x_host  , y_host);
    dft::dft_forward(x_device, y_device);

    ///
    /// Print result of host
    ///
    cout << y_host << endl;

    ///
    /// Copy device result to host and print it
    ///
    y_host = y_device;
    cout << y_host << endl;




    return 0;
}
