# Week 1 概念速查题

定位：10-15 分钟快速复习。这里不要求写代码，重点是把 Week 1 六个示例背后的易错概念说清楚。

建议用法：

1. 先遮住“短答”，自己口头回答。
2. 答不出来就回到对应 `.mu` 文件看 `PART III`。
3. 面试或复盘时优先扫“易错点”。

## 1-3：线程模型 + 硬件视角

### Q1. `kernel<<<grid, block>>>` 里的两个参数分别是什么意思？

短答：`grid` 决定启动多少个 block，`block` 决定每个 block 里有多少个 thread。

易错点：`<<<2, 4>>>` 不是 2 个线程，也不是 4 个线程，而是 2 个 block、每个 block 4 个线程，总共 8 个线程。

### Q2. 一维全局线程索引怎么写？

短答：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

易错点：`threadIdx.x` 只是 block 内编号。多 block 时只用 `threadIdx.x` 会让不同 block 的线程写同一批位置。

### Q3. 为什么大多数 kernel 都要写 `if (idx < N)`？

短答：因为 block 数通常用向上取整计算，最后一个 block 可能有多余线程。

易错点：`blocks = (N + threads - 1) / threads` 会保证线程数不少于 `N`，但不保证刚好等于 `N`。多出来的线程必须主动退出。

### Q4. `threadIdx.x` 和全局数组下标是一回事吗？

短答：不是。`threadIdx.x` 是 block 内局部编号；全局数组下标通常要用 `blockIdx.x * blockDim.x + threadIdx.x`。

易错点：单 block 时 `threadIdx.x` 恰好等于全局下标，所以很容易形成错觉。多 block 时每个 block 都有自己的 `threadIdx.x = 0..blockDim.x-1`。

### Q5. 二维矩阵的 `(row, col)` 怎么从 block/thread 算出来？

短答：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

易错点：通常 `x` 对应列，`y` 对应行。写矩阵和图像 kernel 时，不要把 `x` 当 row、`y` 当 col。

### Q6. 二维坐标怎么变成一维数组下标？

短答：row-major 存储下：

```cpp
int idx = row * width + col;
```

易错点：`width` 是列数，不是行数。写成 `col * height + row` 通常会导致转置式访问或越界。

### Q7. `gridDim` 和 `blockDim` 有什么区别？

短答：`gridDim` 是 grid 中 block 的数量；`blockDim` 是每个 block 中 thread 的数量。

易错点：`gridDim.x * blockDim.x` 才是一维方向上的总线程数。只看 `gridDim.x` 或只看 `blockDim.x` 都不完整。

### Q8. GPU 上 `printf` 的输出顺序能代表执行顺序吗？

短答：不能。block 和 thread 的调度顺序由硬件决定，输出可能交错。

易错点：看到打印顺序不是 `0, 1, 2...` 不代表索引错了。验证逻辑时要看索引值和结果，不要依赖打印顺序。

### Q9. `blockIdx`、`threadIdx`、`blockDim`、`gridDim` 分别是谁决定的？

短答：`blockDim` 和 `gridDim` 来自 launch config；`blockIdx` 和 `threadIdx` 是 GPU 为每个 block/thread 自动给出的内置编号。

易错点：它们只能在 device/kernel 代码里直接用，host 代码里没有这些内置变量。

### Q10. block 和 warp 的区别是什么？

短答：block 是调度和协作边界；warp 是硬件实际执行的一组线程。MUSA warp size 通常按 128 理解，CUDA 常见是 32。

易错点：Week 1 的 elementwise kernel 暂时不需要手写 warp 逻辑，但后面做 shuffle、warp reduce、occupancy 时必须重新考虑 MUSA 的 warp size。

### Q11. 一个 block 会不会跨多个 SM 执行？

短答：不会。一个 block 会被调度到同一个 SM 上执行。

易错点：block 内线程可以用 shared memory 和 `__syncthreads()` 协作；不同 block 之间不能靠普通方式直接同步。

### Q12. `Max threads per block` 超了会怎样？

短答：kernel launch 会失败，通常通过 launch 后的错误检查暴露。

易错点：这类错误不是 kernel 内运行到一半才失败，而是 launch config 本身非法。launch 后要检查 `musaGetLastError()`。

## 4-6：显存 + 错误 + 异步

### Q13. host 指针和 device 指针最大的区别是什么？

短答：host 指针指向 CPU 内存，device 指针指向 GPU 显存；CPU 不能把 device 指针当普通数组直接解引用。

易错点：`d_ptr[0]` 写在 host 代码里通常是错的。要通过 `musaMemcpy` 拷回 host，再读 `h_ptr[0]`。

### Q14. `musaMalloc` 分配出来的内存在哪里？

短答：在 device/global memory 上。

易错点：`musaMalloc(&d, bytes)` 只是在 GPU 侧分配显存，不会自动把 host 数据放进去，也不会初始化成你想要的值。

### Q15. `musaMemcpy` 的方向参数为什么容易错？

短答：因为 `dst/src` 和 `musaMemcpyHostToDevice`、`musaMemcpyDeviceToHost` 必须一致。

易错点：H2D 应该是 `musaMemcpy(d_ptr, h_ptr, bytes, musaMemcpyHostToDevice)`；D2H 应该是 `musaMemcpy(h_ptr, d_ptr, bytes, musaMemcpyDeviceToHost)`。

### Q16. `musaMemset(d, 1, bytes)` 能把 float 数组设成 `1.0f` 吗？

短答：不能。`musaMemset` 是按字节填充，不是按元素赋值。

易错点：float 的 `1.0f` 二进制不是每个字节都等于 `0x01`。`musaMemset(d, 1, bytes)` 得到的是类似 `0x01010101` 的 bit pattern。

### Q17. `musaGetLastError()` 和 `musaDeviceSynchronize()` 分别抓什么错误？

短答：`musaGetLastError()` 常用于抓 launch 配置错误；`musaDeviceSynchronize()` 会等待 kernel 完成，并暴露异步执行中的错误。

易错点：只写其中一个不够稳。调试期常用范式是：

```cpp
kernel<<<grid, block>>>(...);
MUSA_CHECK(musaGetLastError());
MUSA_CHECK(musaDeviceSynchronize());
```

### Q18. kernel launch 为什么说是异步的？

短答：host 线程把 kernel 提交给 GPU 后通常立刻返回，不等 GPU 真的算完。

易错点：CPU 计时如果只包住 `kernel<<<...>>>`，量到的大多是 launch overhead，不是 kernel 执行时间。

### Q19. 什么情况下必须同步？

短答：当 host 要使用 GPU 计算结果、要准确计时、要确认 kernel 是否报错时，需要同步。

易错点：没有同步点前，不要假设 GPU 已经写完结果。`musaDeviceSynchronize()` 是最直接的全设备同步，但 Week 2 多 stream 后会更偏向 stream/event 级同步。

### Q20. `musaMemcpy(DeviceToHost)` 为什么经常“看起来救了同步”？

短答：D2H 拷贝需要拿到 GPU 写完的数据，因此它会隐式等待相关 GPU 工作完成。

易错点：结果正确不代表你理解了同步。有些代码删掉显式 sync 后仍然正确，是因为后面的 D2H `musaMemcpy` 顺手同步了。

### Q21. launch overhead 什么时候会成为瓶颈？

短答：当 kernel 本身很小、调用次数很多时，提交 kernel 的固定开销会占主要时间。

易错点：一个很小的 elementwise kernel 单次看不慢，但循环 launch 成千上万次就可能被 launch overhead 吃掉。Week 2 的 Graph 就是为减少重复提交开销而准备的概念之一。

### Q22. `musaDeviceSynchronize()` 是不是越多越安全？

短答：调试时多同步便于定位错误；性能代码里同步太多会破坏并发和流水线。

易错点：Week 1 为了看清行为可以频繁 sync。到了 Week 2 多 stream，乱加 device sync 会把本来能重叠的 H2D/kernel/D2H 全部串行化。

## CUDA -> MUSA 基础迁移

### Q23. 从 CUDA 迁移到 MUSA，哪些东西通常不变？

短答：`__global__`、`__device__`、`threadIdx`、`blockIdx`、`blockDim`、`gridDim`、`<<<grid, block>>>` 这些 kernel 语法和线程索引概念基本不变。

易错点：语法相似不代表性能参数也能照搬。尤其是 warp size 和硬件调度细节。

### Q24. 哪些 Runtime API 名字最常见？

短答：`cudaMalloc -> musaMalloc`，`cudaMemcpy -> musaMemcpy`，`cudaMemset -> musaMemset`，`cudaFree -> musaFree`，`cudaDeviceSynchronize -> musaDeviceSynchronize`。

易错点：机械替换能解决很多入门代码，但遇到库、调试器、profiler、warp-level intrinsic 时要查 MUSA 文档或仓库里的对照表。

## 面试口述模板

### Q25. 如何解释一个最小 GPU 程序的执行流程？

短答：host 分配并准备数据，把输入拷到 device，launch kernel 并行计算，检查 launch 错误，必要时同步，再把结果拷回 host，最后释放 host/device 资源。

易错点：不要漏掉错误检查和同步语义。kernel launch 不是普通函数调用。

### Q26. 如何判断一个 Week 1 kernel 是否写对？

短答：先检查全局索引和边界保护，再检查 H2D/D2H 方向，再检查 launch 后错误，最后用小输入对拍结果。

易错点：不要一上来跑大数据。小输入能更快暴露索引、越界和拷贝方向错误。

### Q27. Week 1 到 Week 2 的关键过渡是什么？

短答：Week 1 学会单个 kernel 的线程、显存、错误和异步；Week 2 开始把 H2D、kernel、D2H 放进 stream/event/graph 里组织执行。

易错点：如果 Week 1 的异步和隐式同步没理解，Week 2 看多 stream 时间线时会很容易误判性能结果。
