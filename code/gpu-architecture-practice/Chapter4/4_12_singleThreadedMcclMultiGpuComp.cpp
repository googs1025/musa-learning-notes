// ============================================================================
//  示例: 单线程控制多 GPU 执行 MCCL 通信
//
//  学习目标:
//    1. 在一个进程线程中遍历多个 device, 分别分配 buffer 和创建 stream。
//    2. 创建 MCCL communicator, 在多卡之间执行 collective。
//    3. 理解每次操作前的 musaSetDevice 对当前线程上下文的影响。
//
//  阅读顺序:
//    先看 device 循环和 musaSetDevice, 再看每张卡的 stream、buffer 与 MCCL communicator。
//
//  注意:
//    单线程写法简单, 但所有 device 的提交逻辑串在一个控制流里; 多卡数量增加后,
//    代码容易变长, 也更依赖正确的设备上下文切换。
// ============================================================================
#include <stdlib.h>
#include <stdio.h>
#include <musa_runtime.h>
#include <mccl.h>
#define MUSACHECK(cmd)                                                                             \
    do {                                                                                           \
        musaError_t err = cmd;                                                                     \
        if (err != musaSuccess) {                                                                  \
            printf("Failed: MUSA error %s:%d '%s'\n", __FILE__, __LINE__,                          \
                   musaGetErrorString(err));                                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
#define MCCLCHECK(cmd)                                                                             \
    do {                                                                                           \
        mcclResult_t res = cmd;                                                                    \
        if (res != mcclSuccess) {                                                                  \
            printf("Failed, MCCL error %s:%d '%s'\n", __FILE__, __LINE__,                          \
                   mcclGetErrorString(res));                                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
int main(int argc, char *argv[]) {
    int num_gpus = 2;
    if (argc < 2) {
        printf("Usage: %s <num_gpus>\n", argv[0]);
        printf("default num_gpus %d\n", num_gpus);
    }
    num_gpus = atoi(argv[1]);
    if (num_gpus <= 0 || num_gpus > 8) {
        printf("Error: num_gpus must be between 1 and 8\n");
        return -1;
    }
    printf("[INFO] Using num_gpus = %d\n", num_gpus);
    // managing communicators
    mcclComm_t comms[num_gpus];
    // managing devices
    int ndev = num_gpus;
    int size = 32 << 20;
    int devs[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    // allocating and initializing device buffers
    float **sendbuff = (float **)malloc(ndev * sizeof(float *));
    float **recvbuff = (float **)malloc(ndev * sizeof(float *));
    musaStream_t *s = (musaStream_t *)malloc(sizeof(musaStream_t) * ndev);
    // create device buffers and streams
    for (int i = 0; i < ndev; ++i) {
        MUSACHECK(musaSetDevice(i));
        MUSACHECK(musaMalloc((void **)sendbuff + i, size * sizeof(float)));
        MUSACHECK(musaMalloc((void **)recvbuff + i, size * sizeof(float)));
        MUSACHECK(musaMemset(sendbuff[i], 1, size * sizeof(float)));
        MUSACHECK(musaMemset(recvbuff[i], 0, size * sizeof(float)));
        MUSACHECK(musaStreamCreate(s + i));
    }
    // initializing MCCL communicators
    MCCLCHECK(mcclCommInitAll(comms, ndev, devs));
    // calling MCCL communication API. Group API is required when using multiple devices per thread
    MCCLCHECK(mcclGroupStart());
    for (int i = 0; i < ndev; ++i)
        MCCLCHECK(mcclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, mcclFloat,
                                mcclSum, comms[i], s[i]));
    MCCLCHECK(mcclGroupEnd());
    // synchronizing on MUSA streams to wait for completion of MCCL operation
    for (int i = 0; i < ndev; ++i) {
        MUSACHECK(musaSetDevice(i));
        MUSACHECK(musaStreamSynchronize(s[i]));
    }
    // free device buffers
    for (int i = 0; i < ndev; ++i) {
        MUSACHECK(musaSetDevice(i));
        MUSACHECK(musaFree(sendbuff[i]));
        MUSACHECK(musaFree(recvbuff[i]));
    }
    // finalizing MCCL
    for (int i = 0; i < ndev; ++i) {
        MCCLCHECK(mcclCommDestroy(comms[i]));
    }
    printf("Multi devices(%d) on one process one thread Success\n", num_gpus);
    return 0;
}
