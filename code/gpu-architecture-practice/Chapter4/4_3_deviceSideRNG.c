// ============================================================================
//  示例: muRAND device-side 随机数生成
//
//  学习目标:
//    1. 每个 GPU 线程维护自己的 murandState。
//    2. 用相同 seed + 不同 sequence 让线程生成互不相同的随机序列。
//    3. 比较整数随机数、均匀分布和正态分布的简单统计结果。
//
//  数据流:
//    setup_kernel 初始化 devStates -> generate_*_kernel 写 devResults -> host 汇总比例
//
//  阅读顺序:
//    先看每个线程如何取得自己的 state, 再看 state 是否在 kernel 结束后写回 global memory。
//
//  注意:
//    state 要回写到 global memory, 否则下一次 kernel 仍会从旧状态继续。
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <musa_runtime.h>
#include <murand_kernel.h>
#define MUSA_CALL(x)                                                                               \
    do {                                                                                           \
        if ((x) != musaSuccess) {                                                                  \
            printf("Error at %s:%d\n", __FILE__, __LINE__);                                        \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (0)

__global__ void setup_kernel(murandState *states) {
    // 每个线程拥有独立的 state; id 同时决定它在 states 中的槽位和随机序列号。
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    /* 各个线程对随机数生成器进行初始化，使用相同的种子，不同的序列号，不设置偏置量 */
    murand_init(1111, id, 0, &states[id]);
}

__global__ void generate_kernel(murandState *states, int n, unsigned int *result) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    unsigned int x;
    // 先把 state 拷到寄存器/局部变量中, 循环结束后再写回, 减少反复访问 global memory。
    murandState localState = states[id];
    /* 生成无符号整型伪随机数序列 */
    for (int i = 0; i < n; i++) {
        x = murand(&localState);
        /* 检查最低位是否为1 */

        if (x & 1)
            count++;
    }

    states[id] = localState;
    // 每个线程只写自己的 result[id], 因此这里不需要原子加。
    result[id] += count;
}

__global__ void generate_uniform_kernel(murandState *states, int n, unsigned int *result) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    float x;
    murandState localState = states[id];
    /* 生成均匀分布的伪随机数序列 */
    for (int i = 0; i < n; i++) {
        x = murand_uniform(&localState);
        /* 检查结果大于0.5的随即值 */
        if (x > .5)
            count++;
    }
    states[id] = localState;
    result[id] += count;
}

__global__ void generate_normal_kernel(murandState *states, int n, unsigned int *result) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int count = 0;
    float x;
    murandState localState = states[id];
    for (int i = 0; i < n; i++) {
        x = murand_normal(&localState);
        /* 检查结果在1个标准方差间的随机数 */
        if ((x > -1.0) && (x < 1.0))
            count++;
    }
    states[id] = localState;
    result[id] += count;
}

int main() {
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = 64;
    const unsigned int totalThreads = threadsPerBlock * blockCount;
    unsigned int i;
    unsigned int total;
    murandState *devStates;
    unsigned int *devResults, *hostResults;
    int sampleCount = 10000;

    /* 为计算结果在Host端分配空间 */
    hostResults = (unsigned int *)calloc(totalThreads, sizeof(int));
    /* 为计算结果在Device端分配空间 */
    MUSA_CALL(musaMalloc((void **)&devResults, totalThreads * sizeof(unsigned int)));
    /* 设置为 0 */
    MUSA_CALL(musaMemset(devResults, 0, totalThreads * sizeof(unsigned int)));

    /* 为随机数生成器的状态分配空间 */
    MUSA_CALL(musaMalloc((void **)&devStates, totalThreads * sizeof(murandState)));

    /* 设置生成器状态 */
    // 64 个 block × 64 个线程, 总计 totalThreads 个独立 RNG state。
    setup_kernel<<<64, 64>>>(devStates);

    /* 生成伪随机数 */
    generate_kernel<<<64, 64>>>(devStates, sampleCount, devResults);

    /* 将结果从device拷贝到host */
    MUSA_CALL(musaMemcpy(hostResults, devResults, totalThreads * sizeof(unsigned int),
                         musaMemcpyDeviceToHost));

    total = 0;
    for (i = 0; i < totalThreads; i++)
        total += hostResults[i];

    printf("最低位为1的随机数占比(%%)                    : %10.13f\n",
           (float)total / (totalThreads * sampleCount) * (float)100);

    /* 结果置 0 */
    MUSA_CALL(musaMemset(devResults, 0, totalThreads * sizeof(unsigned int)));

    /* 生成均匀分布的伪随机数序列 */
    generate_uniform_kernel<<<64, 64>>>(devStates, sampleCount, devResults);

    /* 将结果从device拷贝到host */
    MUSA_CALL(musaMemcpy(hostResults, devResults, totalThreads * sizeof(unsigned int),
                         musaMemcpyDeviceToHost));

    total = 0;
    for (i = 0; i < totalThreads; i++)
        total += hostResults[i];

    printf("均匀分布中随机数大于 0.5 的占比(%%)          : %10.13f\n",
           (float)total / (totalThreads * sampleCount) * (float)100);

    /* 结果置 0 */
    MUSA_CALL(musaMemset(devResults, 0, totalThreads * sizeof(unsigned int)));

    /* 生成正态分布的伪随机数序列 */
    generate_normal_kernel<<<64, 64>>>(devStates, sampleCount, devResults);

    /* 将结果从device拷贝到host */
    MUSA_CALL(musaMemcpy(hostResults, devResults, totalThreads * sizeof(unsigned int),
                         musaMemcpyDeviceToHost));

    total = 0;
    for (i = 0; i < totalThreads; i++)
        total += hostResults[i];

    printf("标准正态分布中随机数在(-1, +1)之间的占比(%%) : %10.13f\n",
           (float)total / (totalThreads * sampleCount) * (float)100);

    /* 资源释放 */

    MUSA_CALL(musaFree(devStates));
    MUSA_CALL(musaFree(devResults));
    free(hostResults);

    return 0;
}
