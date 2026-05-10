// ============================================================================
//  文件：03_device_info.mu
//  标题：Week 1 · 示例 3 · 设备查询
//  目标：用 Runtime API 获取 GPU 硬件规格，建立对硬件的具体感知
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 1.1 ──────────────────────────────────────────────────────────────────
// │  为什么要查询设备信息？
// └──────────────────────────────────────────────────────────────────────────
//
//  GPU 优化几乎处处依赖硬件常数：
//    • block size 应是 warp size 的倍数      → warpSize
//    • shared memory 用量受限                → sharedMemPerBlock
//    • occupancy 计算                        → multiProcessorCount, maxThreadsPerBlock
//    • 算理论带宽                             → memoryClockRate × memoryBusWidth
//
//  写跨设备代码时，不能写死常量，要在运行时查。

// ┌─ § 1.2 ──────────────────────────────────────────────────────────────────
// │  Runtime API 的错误处理范式
// └──────────────────────────────────────────────────────────────────────────
//
//      musaError_t err = musaXxx(...);
//      if (err != musaSuccess) { ... }
//
//  几乎所有 musa API 都返回 musaError_t。
//  CHECK 宏把这个模板化，让代码更短、错误信息更明确（带文件名和行号）。

// ┌─ § 1.3 ──────────────────────────────────────────────────────────────────
// │  musaDeviceProp 的关键字段
// └──────────────────────────────────────────────────────────────────────────
//
//      name                   设备名（字符串）
//      multiProcessorCount    SM 数量
//      warpSize               一个 warp 多少线程（NV=32, MUSA=128）
//      maxThreadsPerBlock     一个 block 最多多少线程
//      sharedMemPerBlock      每 block shared memory 字节数
//      totalGlobalMem         显存总大小
//      major / minor          compute capability
//      regsPerBlock           每 block 寄存器数
//      memoryClockRate        显存频率（kHz）
//      memoryBusWidth         显存位宽（bit）

// ┌─ § 1.4 ──────────────────────────────────────────────────────────────────
// │  多设备支持
// └──────────────────────────────────────────────────────────────────────────
//
//  一台机器可以有多张 GPU。
//    1. musaGetDeviceCount() 拿到数量
//    2. 循环查询每个设备
//    3. 在某 device 上算 kernel 前先 musaSetDevice(i)


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
//  编译：make          运行：./03_device_info

#include <musa_runtime.h>
#include <cstdio>

// 错误检查宏：所有 musa API 调用都包一层
#define CHECK(call) do {                                       \
    musaError_t _e = (call);                                   \
    if (_e != musaSuccess) {                                   \
        fprintf(stderr, "MUSA error %d at %s:%d\n",            \
                (int)_e, __FILE__, __LINE__);                  \
        return 1;                                              \
    }                                                          \
} while (0)

int main() {
    // 1. 查总共几张 GPU
    int count = 0;
    CHECK(musaGetDeviceCount(&count));
    printf("Detected %d MUSA device(s)\n\n", count);

    // 2. 逐个查询硬件规格
    for (int i = 0; i < count; ++i) {
        musaDeviceProp p{};                          // 必须值初始化
        CHECK(musaGetDeviceProperties(&p, i));

        printf("Device %d: %s\n", i, p.name);
        printf("  SM count                 : %d\n",     p.multiProcessorCount);
        printf("  Warp size                : %d\n",     p.warpSize);
        printf("  Max threads per block    : %d\n",     p.maxThreadsPerBlock);
        printf("  Shared memory per block  : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  Total global memory      : %zu MB\n", p.totalGlobalMem  / (1024UL*1024));
        printf("  Compute capability       : %d.%d\n",  p.major, p.minor);
        printf("\n");
    }

    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  练习题与解答（对应 exercises.md E1.3 / E1.4）                ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: <<<1, 4096>>> 能跑吗？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 思路：单 block 线程数受 maxThreadsPerBlock 限制（通常 1024）。
//          4096 远超上限，启动会失败。
//
//  ▸ 答案：不能。kernel launch 立即失败，
//          musaGetLastError() 通常返回 musaErrorInvalidConfiguration（错误码 9）。
//
//  ▸ 关键：launch 失败不会 abort 程序，但 kernel 没执行。
//          一定要在 launch 后调用 CHECK(musaGetLastError())，否则 bug 会被吞掉。

// ★ Q2: 扩展打印这些字段
// ──────────────────────────────────────────────────────────────────────────
//
//  字段：regsPerBlock、memoryClockRate、memoryBusWidth、l2CacheSize
//
//  ▸ 代码片段：
//      printf("  Regs per block           : %d\n",     p.regsPerBlock);
//      printf("  Memory clock rate        : %d kHz\n", p.memoryClockRate);
//      printf("  Memory bus width         : %d bit\n", p.memoryBusWidth);
//      printf("  L2 cache size            : %d KB\n",  p.l2CacheSize / 1024);

// ★ Q3: 算理论显存带宽
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 公式：
//      bw (GB/s) = 2 * memoryClockRate(kHz) * memoryBusWidth(bit) / 8 / 1e6
//
//  ▸ 解读：
//      • 系数 2：DDR (Double Data Rate)，一周期传两次
//      • /8    ：bit 转 byte
//      • /1e6  ：单位推导后转 GB/s
//
//  ▸ 代码：
//      double bw = 2.0 * p.memoryClockRate * p.memoryBusWidth / 8.0 / 1e6;
//      printf("  Theoretical bandwidth    : %.1f GB/s\n", bw);
//
//  这个值会在 Week 3 的 03_saxpy_bandwidth 中作为基准计算"带宽利用率"。

// ★ Q4 (扩展): 多 GPU 的选择策略
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 答案：
//      • 默认使用 device 0
//      • 显式切换：musaSetDevice(i)
//      • 自动选最佳设备：musaChooseDevice + musaDeviceProp（按需求过滤）
//      • 多卡并行：每个 host 线程绑定一张卡，配合 stream 使用
//      • 跨卡 collective：走 NCCL/MCCL
// ============================================================================
