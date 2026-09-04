// ============================================================================
//  示例: 查询并打印 MUSA 设备基础属性
//
//  学习目标:
//    1. 使用 musaGetDeviceCount 和 musaGetDeviceProperties 枚举设备。
//    2. 观察每张卡的 SM 数、线程上限、shared memory、寄存器、纹理尺寸等字段。
//    3. 用 musaDeviceCanAccessPeer 判断设备之间是否支持 peer access。
//
//  注意:
//    设备属性决定 kernel 配置上限, 但不等于性能最优参数。调优仍需要结合实际
//    kernel 的寄存器、shared memory 和访存模式。
// ============================================================================
#include <iostream>
#include <iomanip>
#include "musa_runtime.h"

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    exit(EXIT_FAILURE);

#define MUSA_CHECK(error)                                                                          \
    if (error != musaSuccess) {                                                                    \
        printf("%serror: '%s'(%d) at %s:%d%s\n", KRED, musaGetErrorString(error), error, __FILE__, \
               __LINE__, KNRM);                                                                    \
        failed("API returned error code.");                                                        \
    }

void printCompilerInfo() { printf("compiler: mcc\n"); }

double bytesToKB(size_t s) { return (double)s / (1024.0); }
double bytesToGB(size_t s) { return (double)s / (1024.0 * 1024.0 * 1024.0); }

#define printLimit(w1, limit, units)                                                               \
    {                                                                                              \
        size_t val;                                                                                \
        MUSA_CHECK(musaDeviceGetLimit(&val, limit));                                               \
        std::cout << setw(w1) << #limit ": " << val << " " << units << std::endl;                  \
    }

void printDeviceProp(int deviceId) {
    using namespace std;
    const int w1 = 34;

    cout << left;

    cout << setw(w1)
         << "--------------------------------------------------------------------------------"
         << endl;
    cout << setw(w1) << "device#" << deviceId << endl;

    musaDeviceProp props;
    MUSA_CHECK(musaGetDeviceProperties(&props, deviceId));

    cout << setw(w1) << "Name: " << props.name << endl;
    cout << setw(w1) << "pciBusID: " << hex << "0x" << props.pciBusID << endl;
    cout << setw(w1) << "pciDeviceID: " << "0x" << props.pciDeviceID << endl;
    cout << setw(w1) << "pciDomainID: " << "0x" << props.pciDomainID << dec << endl;
    cout << setw(w1) << "multiProcessorCount: " << props.multiProcessorCount << endl;
    cout << setw(w1) << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor
         << endl;
    cout << setw(w1) << "isMultiGpuBoard: " << props.isMultiGpuBoard << endl;
    cout << setw(w1) << "clockRate: " << (float)props.clockRate / 1000.0 << " Mhz" << endl;
    cout << setw(w1) << "memoryClockRate: " << (float)props.memoryClockRate / 1000.0 << " Mhz"
         << endl;
    cout << setw(w1) << "memoryBusWidth: " << props.memoryBusWidth << endl;
    cout << setw(w1) << "totalGlobalMem: " << fixed << setprecision(2)
         << bytesToGB(props.totalGlobalMem) << " GB" << endl;
    cout << setw(w1) << "sharedMemPerMultiprocessor: " << fixed << setprecision(2)
         << bytesToKB(props.sharedMemPerMultiprocessor) << " KB" << endl;
    cout << setw(w1) << "totalConstMem: " << props.totalConstMem << endl;
    cout << setw(w1) << "sharedMemPerBlock: " << (float)props.sharedMemPerBlock / 1024.0 << " KB"
         << endl;
    cout << setw(w1) << "canMapHostMemory: " << props.canMapHostMemory << endl;
    cout << setw(w1) << "regsPerBlock: " << props.regsPerBlock << endl;
    cout << setw(w1) << "warpSize: " << props.warpSize << endl;
    cout << setw(w1) << "l2CacheSize: " << props.l2CacheSize << endl;
    cout << setw(w1) << "computeMode: " << props.computeMode << endl;
    cout << setw(w1) << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << endl;
    cout << setw(w1) << "maxThreadsDim.x: " << props.maxThreadsDim[0] << endl;
    cout << setw(w1) << "maxThreadsDim.y: " << props.maxThreadsDim[1] << endl;
    cout << setw(w1) << "maxThreadsDim.z: " << props.maxThreadsDim[2] << endl;
    cout << setw(w1) << "maxGridSize.x: " << props.maxGridSize[0] << endl;
    cout << setw(w1) << "maxGridSize.y: " << props.maxGridSize[1] << endl;
    cout << setw(w1) << "maxGridSize.z: " << props.maxGridSize[2] << endl;
    cout << setw(w1) << "major: " << props.major << endl;
    cout << setw(w1) << "minor: " << props.minor << endl;
    cout << setw(w1) << "concurrentKernels: " << props.concurrentKernels << endl;
    cout << setw(w1) << "cooperativeLaunch: " << props.cooperativeLaunch << endl;
    cout << setw(w1) << "cooperativeMultiDeviceLaunch: " << props.cooperativeMultiDeviceLaunch
         << endl;
    cout << setw(w1) << "isIntegrated: " << props.integrated << endl;
    cout << setw(w1) << "maxTexture1D: " << props.maxTexture1D << endl;
    cout << setw(w1) << "maxTexture2D.width: " << props.maxTexture2D[0] << endl;
    cout << setw(w1) << "maxTexture2D.height: " << props.maxTexture2D[1] << endl;
    cout << setw(w1) << "maxTexture3D.width: " << props.maxTexture3D[0] << endl;
    cout << setw(w1) << "maxTexture3D.height: " << props.maxTexture3D[1] << endl;
    cout << setw(w1) << "maxTexture3D.depth: " << props.maxTexture3D[2] << endl;

    int deviceCnt;
    MUSA_CHECK(musaGetDeviceCount(&deviceCnt));
    cout << setw(w1) << "peers: ";
    for (int i = 0; i < deviceCnt; i++) {
        int isPeer;
        MUSA_CHECK(musaDeviceCanAccessPeer(&isPeer, i, deviceId));
        if (isPeer) {
            cout << "device#" << i << " ";
        }
    }
    cout << endl;
    cout << setw(w1) << "non-peers: ";
    for (int i = 0; i < deviceCnt; i++) {
        int isPeer;
        MUSA_CHECK(musaDeviceCanAccessPeer(&isPeer, i, deviceId));
        if (!isPeer) {
            cout << "device#" << i << " ";
        }
    }
    cout << endl;

#ifdef __HIP_PLATFORM_NVCC__
    // Limits:
    cout << endl;
    printLimit(w1, musaLimitStackSize, "bytes/thread");
    printLimit(w1, musaLimitPrintfFifoSize, "bytes/device");
    printLimit(w1, musaLimitMallocHeapSize, "bytes/device");
    printLimit(w1, musaLimitDevRuntimeSyncDepth, "grids");
    printLimit(w1, musaLimitDevRuntimePendingLaunchCount, "launches");
#endif

    cout << endl;

    size_t free, total;
    MUSA_CHECK(musaMemGetInfo(&free, &total));

    cout << fixed << setprecision(2);
    cout << setw(w1) << "memInfo.total: " << bytesToGB(total) << " GB" << endl;
    cout << setw(w1) << "memInfo.free:  " << bytesToGB(free) << " GB (" << setprecision(0)
         << (float)free / total * 100.0 << "%)" << endl;
}

int main() {
    using namespace std;

    cout << endl;

    printCompilerInfo();

    int deviceCnt;

    MUSA_CHECK(musaGetDeviceCount(&deviceCnt));

    for (int i = 0; i < deviceCnt; i++) {
        MUSA_CHECK(musaSetDevice(i));
        printDeviceProp(i);
    }

    std::cout << std::endl;
}
