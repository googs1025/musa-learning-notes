// ============================================================================
//  示例: 双缓冲 Stream 流水线
//
//  学习目标:
//    1. 把输入拆成两个 chunk, 用两个 stream 交替提交。
//    2. 让 H2D 拷贝、kernel 计算、D2H 拷贝尽量重叠。
//    3. 理解 double buffering 的核心是“当前 chunk 计算时, 下一段数据在路上”。
//
//  数据流:
//    h_input[i] --H2D async--> d_input[i] --process--> d_output[i] --D2H async--> h_output[i]
//
//  注意:
//    真正的拷贝/计算重叠依赖 pinned host memory、硬件 copy engine 和 stream 使用方式。
// ============================================================================
#include <musa_runtime.h>
#include <iostream>
const int N = 1000000;
const int blockSize = 256;
const int numStreams = 2;
const int chunkSize = N / numStreams;
__global__ void process(float *input, float *output, int size) {
    // 示例计算很简单: 每个线程处理一个元素的平方。真实场景通常替换成更重的 kernel。
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}
int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;

    // 分配页锁定内存（提升传输性能）
    musaMallocHost(&h_input, N * sizeof(float));
    musaMallocHost(&h_output, N * sizeof(float));

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // 分配设备内存
    musaMalloc(&d_input, N * sizeof(float));
    musaMalloc(&d_output, N * sizeof(float));

    // 创建流
    musaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        musaStreamCreate(&streams[i]);
    }

    // 双缓冲执行：计算与传输重叠
    dim3 grid((chunkSize + blockSize - 1) / blockSize);
    dim3 block(blockSize);

    for (int i = 0; i < numStreams; i++) {
        int offset = i * chunkSize;

        // 异步传输: 第 i 个 stream 只处理自己的 chunk。
        // h_input 使用 musaMallocHost 分配, 满足异步 H2D 更常见的前提。
        musaMemcpyAsync(d_input + offset, h_input + offset, chunkSize * sizeof(float),
                        musaMemcpyHostToDevice, streams[i]);

        // 异步计算: kernel 被提交到同一个 stream, 因此会等本 stream 的 H2D 完成后执行。
        process<<<grid, block, 0, streams[i]>>>(d_input + offset, d_output + offset, chunkSize);

        // 异步回传: 同 stream 内保持顺序, 会等 kernel 完成后再把结果拷回 host。
        musaMemcpyAsync(h_output + offset, d_output + offset, chunkSize * sizeof(float),
                        musaMemcpyDeviceToHost, streams[i]);
    }

    // 等待所有流完成
    for (int i = 0; i < numStreams; i++) {
        musaStreamSynchronize(streams[i]);
    }

    // 清理资源
    for (int i = 0; i < numStreams; i++) {
        musaStreamDestroy(streams[i]);
    }
    musaFree(d_input);
    musaFree(d_output);
    musaFreeHost(h_input);
    musaFreeHost(h_output);

    return 0;
}
