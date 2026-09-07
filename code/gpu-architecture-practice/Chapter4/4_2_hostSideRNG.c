/*
 * 示例: muRAND host-side 随机数生成
 *
 * 学习目标:
 *   1. 在 host 侧创建 murandGenerator_t。
 *   2. 设置 seed 后生成可复现的伪随机数序列。
 *   3. 把随机数写入 device buffer, 再拷回 host 查看。
 *
 * 注意:
 *   host-side API 适合一次性批量生成随机数; 如果每个 GPU 线程需要独立采样,
 *   参考 4_3_deviceSideRNG.c 的 device-side state 写法。
 *
 * 阅读顺序:
 *   先看 generator/seed, 再看 device buffer、生成调用和 D2H 拷贝。
 */
#include <stdio.h>
#include <stdlib.h>
#include <musa_runtime.h>
#include <murand.h>

#define MUSA_CALL(x)                                                                               \
    do {                                                                                           \
        if ((x) != musaSuccess) {                                                                  \
            printf("Error at %s:%d\n", __FILE__, __LINE__);                                        \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (0)
#define MURAND_CALL(x)                                                                             \
    do {                                                                                           \
        if ((x) != MURAND_STATUS_SUCCESS) {                                                        \
            printf("Error at %s:%d\n", __FILE__, __LINE__);                                        \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (0)

int main() {
    size_t n = 100;
    size_t i;

    murandGenerator_t gen;
    float *devData, *hostData;

    /* 在host端分配n个浮点数的空间 */
    hostData = (float *)calloc(n, sizeof(float));

    /* 在device端分配n个浮点数的空间 */
    MUSA_CALL(musaMalloc((void **)&devData, n * sizeof(float)));

    /* 创建伪随机数生成器 */
    MURAND_CALL(murandCreateGenerator(&gen, MURAND_RNG_PSEUDO_DEFAULT));

    /* 设置伪随机数种子 */
    MURAND_CALL(murandSetPseudoRandomGeneratorSeed(gen, 1111ULL));

    /* 在device端生成n个伪随机数 */
    MURAND_CALL(murandGenerateUniform(gen, devData, n));

    /* 拷贝device端伪随机数到host端 */
    MUSA_CALL(musaMemcpy(hostData, devData, n * sizeof(float), musaMemcpyDeviceToHost));

    /* 打印结果 */
    for (i = 0; i < n; i++)
        printf("%1.4f\n", hostData[i]);

    printf("\n");

    /* 释放分配的内存*/
    MURAND_CALL(murandDestroyGenerator(gen));
    MUSA_CALL(musaFree(devData));
    free(hostData);
    return 0;
}
