// ============================================================================
//  示例: 两个 MUSA Stream 并行处理一维向量
//
//  学习目标:
//    1. 使用 pinned host memory 让 musaMemcpyAsync 具备异步传输条件。
//    2. 把同一个向量拆成两个 chunk, 分别放入 stream1/stream2。
//    3. 理解每个 stream 内部保持 H2D -> kernel -> D2H 顺序, 不同 stream 之间可并行。
//
//  数据流:
//    h_a/h_b --H2D--> d_a/d_b --kernel--> d_c --D2H--> h_c
//
//  注意:
//    musaStreamSynchronize(stream) 是观察结果前的同步边界; 只包 kernel launch 计时
//    会低估真实计算时间。
// ============================================================================
#include <musa_runtime.h>
#include <iostream>

__global__ void kernel(float *a, float *b, float *c, int n) {
    // 全局线程号: 每个线程负责一个元素, 多出来的线程通过边界判断退出。
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1000000;
    const size_t size = n * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // 分配主机和设备内存
    musaMallocHost(&h_a, size);
    musaMallocHost(&h_b, size);
    musaMallocHost(&h_c, size);
    musaMalloc(&d_a, size);
    musaMalloc(&d_b, size);
    musaMalloc(&d_c, size);

    // 初始化主机数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 创建两个流
    musaStream_t stream1, stream2;
    musaStreamCreate(&stream1);
    musaStreamCreate(&stream2);

    const int chunk_size = n / 2;
    const size_t chunk_bytes = chunk_size * sizeof(float);

    // 流1: 传输前半段数据 -> 执行核函数 -> 传回前半段结果。
    // 同一个 stream 中的操作按提交顺序执行, 所以不需要在三步之间手动同步。
    musaMemcpyAsync(d_a, h_a, chunk_bytes, musaMemcpyHostToDevice, stream1);
    musaMemcpyAsync(d_b, h_b, chunk_bytes, musaMemcpyHostToDevice, stream1);
    kernel<<<(chunk_size + 255) / 256, 256, 0, stream1>>>(d_a, d_b, d_c, chunk_size);
    musaMemcpyAsync(h_c, d_c, chunk_bytes, musaMemcpyDeviceToHost, stream1);

    // 流2: 处理后半段。两个 stream 没有数据依赖, runtime 可以重叠拷贝和计算。
    musaMemcpyAsync(d_a + chunk_size, h_a + chunk_size, chunk_bytes, musaMemcpyHostToDevice,
                    stream2);
    musaMemcpyAsync(d_b + chunk_size, h_b + chunk_size, chunk_bytes, musaMemcpyHostToDevice,
                    stream2);
    kernel<<<(chunk_size + 255) / 256, 256, 0, stream2>>>(d_a + chunk_size, d_b + chunk_size,
                                                          d_c + chunk_size, chunk_size);
    musaMemcpyAsync(h_c + chunk_size, d_c + chunk_size, chunk_bytes, musaMemcpyDeviceToHost,
                    stream2);

    // 等待所有流完成
    musaStreamSynchronize(stream1);
    musaStreamSynchronize(stream2);

    // 清理资源
    musaStreamDestroy(stream1);
    musaStreamDestroy(stream2);
    musaFree(d_a);
    musaFree(d_b);
    musaFree(d_c);
    musaFreeHost(h_a);
    musaFreeHost(h_b);
    musaFreeHost(h_c);

    return 0;
}
