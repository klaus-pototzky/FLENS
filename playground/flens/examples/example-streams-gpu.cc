#define USE_PLAYGROUND

///
/// Enable features for CUBLAS
///
#define WITH_CUBLAS

///
/// Set flag for main file (only necessary once in a project)
/// Ensures internal global variables are only set once
///
#define MAIN_FILE

///
/// Include FLENS as usual
///
#include <flens/flens.cxx>
#include <ctime>

///
/// CUDA specific example to demonstrate
/// the concepts of streams
/// as well as the difference between
/// pinned and non-pinnend memory.
///
/// The output should look like:
///
///    # 1 Stream: 1.17 [s]
///    # 4 Streams [pinned memory]: 0.8 [s]
///    # 4 Streams [non-pinned memory]: 1.2 [s]
///


using namespace std;
using namespace flens;

int main() {

    int N = 512;

    ///
    /// Usual typedefs
    ///
    typedef double                                 PT;
    typedef std::complex<PT>                       T;

    typedef GeMatrix<FullStorage<T> >              HostMatrix;
    typedef GeMatrix<PinnedHostFullStorage<T> >    PinnedHostMatrix;
    typedef GeMatrix<DeviceFullStorage<T> >        DeviceMatrix;

    const Underscore<int> _;

    HostMatrix         A_host_1(N,N),       A_host_2(N,N),       A_host_3(N,N),       A_host_4(N,N);
    PinnedHostMatrix   A_pinnedhost_1(N,N), A_pinnedhost_2(N,N), A_pinnedhost_3(N,N), A_pinnedhost_4(N,N);
    DeviceMatrix       A_device_1(N,N),     A_device_2(N,N),     A_device_3(N,N),     A_device_4(N,N);
    DeviceMatrix       B_device(N,N),       C_device(N,N);

    //
    // Fill one Matrix with random Data
    //
    fillRandom(A_pinnedhost_1);

    A_device_1 = A_pinnedhost_1;
    A_device_2 = A_pinnedhost_1;
    A_device_3 = A_pinnedhost_1;
    A_device_4 = A_pinnedhost_1;

    B_device   = A_pinnedhost_1;
    C_device   = A_pinnedhost_1;

    //
    // warm-up of CUDA
    //
    A_device_1 = B_device*C_device;



    //
    // Run first test: Everything in one stream
    //
    clock_t begin = clock();
    for (int i = 0; i<10; ++i) {

    	A_device_1 = B_device*C_device;
    	A_pinnedhost_1   = A_device_1;

    	A_device_2 = B_device*C_device;
    	A_pinnedhost_2   = A_device_2;

    	A_device_3 = B_device*C_device;
    	A_pinnedhost_3   = A_device_3;

    	A_device_4 = B_device*C_device;
    	A_pinnedhost_4   = A_device_4;
    }
    cout << "# 1 Stream: " << double(clock() - begin) / CLOCKS_PER_SEC<< " [s]" << endl;

    //
    // Enable asynchronous copy
    //
    enableASyncCopy();
    begin = clock();
    for (int i = 0; i<10; ++i) {

    	///
    	/// Each matrix operatos on its own stream
    	/// This allows streams to overlap
    	///
    	setStream(0);
    	A_device_1 = B_device*C_device;
    	A_pinnedhost_1   = A_device_1;

    	setStream(1);
    	A_device_2 = B_device*C_device;
    	A_pinnedhost_2   = A_device_2;

    	setStream(2);
    	A_device_3 = B_device*C_device;
    	A_pinnedhost_3   = A_device_3;

    	setStream(3);
    	A_device_4 = B_device*C_device;
    	A_pinnedhost_4   = A_device_4;
    }

    //
    // sync all four streams
    //
    syncStream(0, 1, 2, 3);
    cout << "# 4 Streams [pinned memory]: " << double(clock() - begin) / CLOCKS_PER_SEC<< " [s]" << endl;


    begin = clock();
    for (int i = 0; i<10; ++i) {

    	setStream(0);
    	A_device_1 = B_device*C_device;
    	A_host_1   = A_device_1;

    	setStream(1);
    	A_device_2 = B_device*C_device;
    	A_host_2   = A_device_2;

    	setStream(2);
    	A_device_3 = B_device*C_device;
    	A_host_3   = A_device_3;

    	setStream(3);
    	A_device_4 = B_device*C_device;
    	A_host_4   = A_device_4;
    }

    //
    // sync all four streams
    //
    syncStream(0, 1, 2, 3);
    cout << "# 4 Streams [non-pinned memory]: " << double(clock() - begin) / CLOCKS_PER_SEC<< " [s]" << endl;

    return 0;
}
