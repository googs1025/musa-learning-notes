#include "musa_common.h"
#include <cstdio>

__global__ void child_kernel() {
    if (threadIdx.x == 0) printf("child block=%d\n", blockIdx.x);
}

__global__ void parent_kernel() {
    if (threadIdx.x == 0) {
        printf("parent launches child from block=%d\n", blockIdx.x);
        child_kernel<<<2, 4>>>();
    }
}

int main() {
    parent_kernel<<<2, 4>>>();
    MUSA_CHECK_KERNEL();
    std::printf("If this fails to compile or launch, dynamic parallelism is not enabled in this SDK/config.\n");
}
