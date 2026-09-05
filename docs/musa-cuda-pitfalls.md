# MUSA / CUDA 高频混淆点

这份清单不是 API 大全，而是写 MUSA / CUDA kernel 时最容易把概念混在一起的地方。建议和 `code/week*/learning-notes.md` 对照读。

## 速查索引

| 类别 | 最容易错的判断 | 对应章节 |
|---|---|---|
| grid / block / thread | `threadIdx` 不是全局下标 | [`code/week1/learning-notes.md`](../code/week1/learning-notes.md) |
| host / device 内存 | device pointer 不能在 host 直接解引用 | [`code/week1/learning-notes.md`](../code/week1/learning-notes.md) |
| 错误检查 | launch 错误和 kernel 执行错误暴露时机不同 | [`code/week1/learning-notes.md`](../code/week1/learning-notes.md) |
| stream / event | 异步提交不等于已经执行完成 | [`code/week2/learning-notes.md`](../code/week2/learning-notes.md) |
| pinned / unified memory | 易用性和性能不是一回事 | [`code/week2/learning-notes.md`](../code/week2/learning-notes.md) |
| warp / reduce | CUDA 的 32-wide 假设不能照搬到 MUSA | [`code/week3/learning-notes.md`](../code/week3/learning-notes.md) |
| global memory | 结果正确不代表访存高效 | [`code/week4/learning-notes.md`](../code/week4/learning-notes.md) |
| shared memory | shared 不是自动加速器, 仍有同步和 bank conflict | [`code/week5/learning-notes.md`](../code/week5/learning-notes.md) |
| GEMM / BLAS | row-major 直觉容易撞上 BLAS column-major 语义 | [`code/week5/learning-notes.md`](../code/week5/learning-notes.md) |
| 多卡 / 框架 | rank、device、stream、tensor device 必须逐层对齐 | [`code/week6/learning-notes.md`](../code/week6/learning-notes.md) |

## 1. Launch 和索引

- `<<<grid, block>>>` 不是“启动多少线程”的两个数字。`grid` 是 block 数量，`block` 是每个 block 的 thread 数量。
- `threadIdx.x` 是 block 内局部编号。全局数组下标通常是 `blockIdx.x * blockDim.x + threadIdx.x`。
- `gridDim.x * blockDim.x` 才是一维方向的总线程数。
- 二维矩阵里通常 `x -> col`，`y -> row`；row-major 展平成 `idx = row * width + col`。
- `if (idx < N)` 是正确性保护，不是可选优化。

## 2. 内存和拷贝

- `h_` 指针在 host RAM，`d_` 指针在 device global memory。host 代码里直接读 `d_ptr[0]` 是错的。
- `musaMalloc` 只分配 device memory，不初始化，也不自动拷贝 host 数据。
- `musaMemcpy(dst, src, bytes, direction)` 里 `dst/src` 和方向必须一致。
- `musaMemset(d, 1, bytes)` 是按字节写 `0x01`，不能把 float 数组设成 `1.0f`。
- `musaMemcpy(DeviceToHost)` 常带来隐式同步，但不要把“结果对了”误认为“同步模型理解对了”。

## 3. 异步、Stream 和 Event

- kernel launch 通常是异步入队，host 线程不会等 GPU 算完。
- `musaDeviceSynchronize()` 是全设备同步，调试方便，但会破坏多流并发。
- `musaStreamSynchronize(stream)` 只等某个 stream，更适合 Week 2 之后的代码。
- Event 表达的是“某个点完成后再继续”，不是 CPU 侧普通时间戳。
- `musaMemcpyAsync` 想真正异步，通常还需要 pinned host memory 和正确的 stream 使用方式。

## 4. Warp、同步和规约

- block 是协作边界，warp 是硬件执行边界。二者不是一回事。
- `__syncthreads()` 只同步同一个 block 内线程，不能同步不同 block。
- block 间规约通常要写多个 kernel 或让 host 做 final reduce。
- CUDA 常见 warp size 是 32；MUSA 上要查 `musaDeviceProp::warpSize`，不要把 32 写死进算法。
- warp-level shuffle / reduce 的 mask、循环边界、活跃线程数都要重新检查。

## 5. 访存和性能

- GPU kernel 常常不是算术慢，而是 global memory 访问慢。
- 相邻线程访问相邻地址通常更容易合并访存；stride、offset、AoS 都可能破坏吞吐。
- shared memory 只有在数据会被复用时才值得搬进去。
- shared memory 也会有 bank conflict；padding 是常见处理方式。
- unroll 不一定更快，可能增加寄存器压力或放大错误的访存模式。

## 6. 库调用和矩阵布局

- 手写 row-major kernel 和 BLAS 库的 column-major 语义容易打架。
- `lda/ldb/ldc` 是 leading dimension，不是随便填矩阵宽度。
- `alpha/beta` 通常是 host 标量指针，矩阵 A/B/C 是 device 指针。
- 库调用成功不代表数值结果正确，仍要做小矩阵 reference 对拍。
- muBLAS / muDNN / muSOLVER / MCCL 的 handle、descriptor、workspace 都有生命周期。

## 7. 多卡和框架

- 多卡代码里要分清 global rank、local rank、device id、communicator rank。
- 每次跨 device 操作前都要确认当前线程的 `musaSetDevice`。
- 每张卡通常有自己的 buffer、stream、event 和 communicator 关系。
- PyTorch / torch_musa 里 `tensor.device`、模型 device、custom op 内部分配 device 必须一致。
- 多卡错误要记录 SDK、驱动、容器、GPU 数量、rank 映射和完整启动命令。

## 读代码时的 6 个问题

1. 这个 kernel 的全局索引怎么算？
2. 最后一个 block / tile 的边界处理了吗？
3. 当前指针在 host 还是 device？
4. 当前操作是同步还是异步？
5. 相邻线程访问的地址是否相邻？
6. 这里有没有写死 CUDA 的 warp size、架构号或库名？
