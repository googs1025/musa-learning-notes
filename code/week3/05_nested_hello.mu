// 预计输出：
//   parent block=0
//   parent block=1
//   child block=0
//   child block=1
// 注意：这里用 host 端分两阶段启动，避免依赖当前 SDK 不支持的动态并行。
// parent 完成后才启动 child，因此两组输出的阶段顺序是确定的。
//
//   host ──launch──> parent grid (2 blocks × 4 threads)
//    │                         │
//    │                         └─ 每个 block 的 thread 0 打印一次
//    │
//    └─ synchronize ──launch──> child grid (2 blocks × 4 threads)
//                                  └─ 每个 block 的 thread 0 打印一次

#include "musa_common.h"
#include <cstdio>

__global__ void child_kernel() {
    // child grid 有 2 个 block，每个 block 只让 thread 0 打印一次。
    if (threadIdx.x == 0) printf("child block=%d\n", blockIdx.x);
}

__global__ void parent_kernel() {
    // parent grid 有 2 个 block；每个 parent block 的 thread 0
    // 只打印一次。child kernel 由 host 在 parent 完成后单独启动。
    if (threadIdx.x == 0) {
        printf("parent block=%d finished\n", blockIdx.x);
    }
}

int main() {
    // 第一阶段：host 启动 parent grid，并等待它完成。
    parent_kernel<<<2, 4>>>();
    MUSA_CHECK_KERNEL();

    // 第二阶段：确认 parent 完成后，host 再启动 child grid。
    // 这不是动态并行，而是最基础、兼容性更好的两阶段 kernel 调度。
    child_kernel<<<2, 4>>>();
    MUSA_CHECK_KERNEL();
}
