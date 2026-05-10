// ============================================================================
//  文件:musa_common.h
//  作用:整个仓库共享的基础工具
//        ─ MUSA_CHECK / MUSA_CHECK_KERNEL  错误检查宏
//        ─ CpuTimer                         host 端 wall-clock 计时
//        ─ GpuTimer                         device 端 musaEvent 计时(精度更高)
//
//  约定:
//    • 所有 .mu 用 #include "musa_common.h" 引入
//    • CMakeLists 已把 code/include 加进 include path,直接写文件名即可
//    • 早期(Week1)示例为了"演示 CHECK 宏长什么样",还会在文件里 inline 一份
//      生产代码请直接用本文件提供的版本
// ============================================================================
#pragma once

#include <musa_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

// ── 错误检查 ────────────────────────────────────────────────────────────────
//
//  范式:
//      MUSA_CHECK(musaMalloc(&d, bytes));
//
//      kernel<<<g, b>>>(...);
//      MUSA_CHECK_KERNEL();             // 同步错误 + 异步错误一次抓
//
//  失败时:打印 文件名:行号 + 错误码 + 人类可读消息,然后 exit(1)。
//  生产代码可以把 exit 换成 abort 或抛异常。
//
#define MUSA_CHECK(call)                                                       \
    do {                                                                       \
        musaError_t _e = (call);                                               \
        if (_e != musaSuccess) {                                               \
            std::fprintf(stderr, "[MUSA] %s:%d  error %d (%s)\n",              \
                         __FILE__, __LINE__,                                   \
                         (int)_e, musaGetErrorString(_e));                     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// kernel launch 后的"标准两步":
//   1) musaGetLastError      ── 同步错误(launch 配置非法)
//   2) musaDeviceSynchronize ── 异步错误(执行期崩溃)
#define MUSA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        MUSA_CHECK(musaGetLastError());                                        \
        MUSA_CHECK(musaDeviceSynchronize());                                   \
    } while (0)


// ── CPU 计时(wall clock,毫秒)────────────────────────────────────────────
//
//  用于:测整段代码、测 launch+sync 总耗时、CPU 端逻辑
//  注意:量 GPU 计算时间不要只用这个 ── kernel 是异步的,要先 sync
//        最准还是 GpuTimer
//
class CpuTimer {
public:
    void start() {
        t0_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point t0_{};
};


// ── GPU 计时(musaEvent,毫秒)────────────────────────────────────────────
//
//  范式:
//      GpuTimer t;
//      t.start();
//      kernel<<<g, b>>>(...);
//      t.stop();
//      printf("kernel: %.3f ms\n", t.elapsed_ms());
//
//  要点:
//    • 使用 musaEvent,精度 ~µs 级,且使用 GPU 自身时基,准确
//    • elapsed_ms 内部会做 musaEventSynchronize,自动等 stop 事件触发
//    • 默认在 stream 0,需要别的 stream 调 start(s) / stop(s)
//
class GpuTimer {
public:
    GpuTimer() {
        musaEventCreate(&t0_);
        musaEventCreate(&t1_);
    }

    ~GpuTimer() {
        musaEventDestroy(t0_);
        musaEventDestroy(t1_);
    }

    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(musaStream_t s = 0) { musaEventRecord(t0_, s); }
    void stop (musaStream_t s = 0) { musaEventRecord(t1_, s); }

    float elapsed_ms() const {
        musaEventSynchronize(t1_);
        float ms = 0.0f;
        musaEventElapsedTime(&ms, t0_, t1_);
        return ms;
    }

private:
    musaEvent_t t0_{};
    musaEvent_t t1_{};
};
