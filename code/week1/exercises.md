# Week 1 习题

> 完成后把答案 / 截图放到 `notes/week1.md`。

## E1.1 修改 hello world 的启动配置（基础）

把 `01_hello_world.mu` 的 `<<<1, 5>>>` 改成：
- (a) `<<<2, 4>>>`
- (b) `<<<4, 8>>>`

预测会打印多少行？运行验证。GPU 输出顺序是固定的吗？为什么？

## E1.2 全局线程索引（基础）

修改 `02_thread_index.mu`，把 grid 改成 `(3, 2)`、block 改成 `(4, 2)`，
让 kernel 同时打印 `threadIdx.x / .y` 和 `blockIdx.x / .y`，
并算出 **全局 (gx, gy) 索引**。

公式：
```
gx = blockIdx.x * blockDim.x + threadIdx.x
gy = blockIdx.y * blockDim.y + threadIdx.y
```

## E1.3 输出超出最大线程数会怎样（边界）

把启动配置改成 `<<<1, 4096>>>`。在你设备上能跑吗？
对比 `03_device_info` 输出的 `Max threads per block`。

写一段失败时的错误信息到 `notes/week1.md`。

## E1.4 设备查询扩展（动手）

仿照 `03_device_info.mu`，再打印这些字段（如 SDK 提供）：
- `regsPerBlock`（每 block 寄存器数）
- `memoryClockRate`（显存频率）
- `memoryBusWidth`（显存位宽）
- `l2CacheSize`（L2 大小）

算出理论显存带宽：`bw = 2 * memoryClockRate * memoryBusWidth / 8 / 1e6` (GB/s)。

## E1.5 CUDA → MUSA 速查表（笔记）

整理一份命名对照表放 `notes/cuda-to-musa.md`，至少包含 15 条：

| CUDA | MUSA |
|---|---|
| cudaMalloc | musaMalloc |
| cudaMemcpy | musaMemcpy |
| cudaMemcpyHostToDevice | musaMemcpyHostToDevice |
| ... | ... |
| nvcc | mcc |
| nvidia-smi | mthreads-gmi |

完成后你应该能"盲翻译" CUDA 教程的代码到 MUSA。

## E1.6 显存指针不能在 host 用（边界）

修改 `04_memory_basics.mu`，把 `musaMemcpy(h, d, ...)` 那一行删掉，
直接 `printf("%f\n", d[0]);`。

预测会发生什么？运行验证。这是为什么？

> 提示：写笔记时把命名规范也定下来 —— `h_` 前缀 = host 指针，`d_` 前缀 = device 指针。

## E1.7 musaMemset 的"按字节"陷阱（基础）

把 `04_memory_basics.mu` 里的 `fill_const` kernel 调用注释掉，
然后在 `musaMemset(d, 0, BYTES)` 后面再加一行 `musaMemset(d, 1, BYTES)`，
拷回 host 后打印 `h[0]`。

得到的不是 1.0f，而是 ~2.36e-38。解释这个数字怎么来的。

## E1.8 主动触发各类错误（动手）

按 `05_error_check.mu` 的写法，再设计两个错误场景：

- (a) 给 `musaMemcpy` 传错方向（比如把 H→D 拷贝写成 `musaMemcpyDeviceToHost`）
- (b) `musaMalloc` 0 字节，再 `musaFree` 这个指针，会出错吗？

把每种情况下 `musaGetLastError` / `musaDeviceSynchronize` 的返回码记录到 `notes/week1.md`。

## E1.9 launch overhead 测量（动手）

修改 `06_async_kernel.mu`，把 `LOOPS` 调到 `100`（几乎不算）和 `50000000`（很重）两组。

填这张表：

| LOOPS | t_launch (ms) | t_wait (ms) | 比例 |
|---|---|---|---|
| 100 | ? | ? | ? |
| 5_000_000 | ? | ? | ? |
| 50_000_000 | ? | ? | ? |

回答：什么场景下 `t_launch` 会成为瓶颈？这预示了下周哪个特性的价值？

## E1.10 把同步删了会怎样（思考 + 验证）

修改 `06_async_kernel.mu`：
- (a) 删掉 `musaDeviceSynchronize()`，立刻 `return 0`。kernel 还会跑完吗？
- (b) 在删掉同步的版本后面加一句 `musaMemcpy(&host_sink, d_sink, ..., D2H)`。结果对吗？

写下你的观察，并解释 musaMemcpy 的"隐式同步"是怎么救你的。
