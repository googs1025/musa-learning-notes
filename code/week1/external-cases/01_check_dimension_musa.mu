// ============================================================================
//  来源主题：Tony-Tan/CUDA_Freshman 1_check_dimension
//  MUSA 改写：检查 threadIdx / blockIdx / blockDim / gridDim 的实际取值
//  原始参考：https://github.com/Tony-Tan/CUDA_Freshman/tree/master/1_check_dimension
// ============================================================================

#include <musa_runtime.h>
#include <cstdio>

#define CHECK(call) do {                                      \
    musaError_t err = (call);                                 \
    if (err != musaSuccess) {                                 \
        fprintf(stderr, "MUSA error %d at %s:%d\n",           \
                (int)err, __FILE__, __LINE__);                \
        return 1;                                             \
    }                                                         \
} while (0)

__global__ void check_dimension() {
    printf("threadIdx=(%d,%d,%d) blockIdx=(%d,%d,%d) "
           "blockDim=(%d,%d,%d) gridDim=(%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main() {
    const int n = 6;
    dim3 block(3);
    dim3 grid((n + block.x - 1) / block.x);

    printf("host grid=(%d,%d,%d)\n", grid.x, grid.y, grid.z);
    printf("host block=(%d,%d,%d)\n", block.x, block.y, block.z);

    check_dimension<<<grid, block>>>();
    CHECK(musaGetLastError());
    CHECK(musaDeviceSynchronize());

    return 0;
}
