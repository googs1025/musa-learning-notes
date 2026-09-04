# Week 1 习题

> 完成后把答案 / 截图放到 `notes/week1.md`。

## E1.1 修改 hello world 的启动配置（基础）

把 `01_hello_world.mu` 的 `<<<1, 5>>>` 改成：
- (a) `<<<2, 4>>>`
- (b) `<<<4, 8>>>`

预测会打印多少行？运行验证。GPU 输出顺序是固定的吗？为什么？

预计输出 / 预期现象：

```text
CPU: Hello world!
GPU: tid=0 Hello world!
GPU: tid=1 Hello world!
...
```

- `<<<2, 4>>>`：GPU 一共打印 8 行，`tid=0..3` 会各出现 2 次。
- `<<<4, 8>>>`：GPU 一共打印 32 行，`tid=0..7` 会各出现 4 次。
- GPU 行的顺序不保证固定；不同 block / warp 的调度和 `printf` 缓冲写入顺序都可能变化。

## E1.2 全局线程索引（基础）

修改 `02_thread_index.mu`，把 grid 改成 `(3, 2)`、block 改成 `(4, 2)`，
让 kernel 同时打印 `threadIdx.x / .y` 和 `blockIdx.x / .y`，
并算出 **全局 (gx, gy) 索引**。

公式：
```
gx = blockIdx.x * blockDim.x + threadIdx.x
gy = blockIdx.y * blockDim.y + threadIdx.y
```

预计输出 / 预期现象：

```text
block=(0,0) thread=(0,0) global=(0,0)
block=(0,0) thread=(1,0) global=(1,0)
...
block=(2,1) thread=(3,1) global=(11,3)
```

- 总线程数是 `3 * 2 * 4 * 2 = 48`，所以应该打印 48 行。
- 全局 `gx` 范围是 `0..11`，全局 `gy` 范围是 `0..3`。
- 输出顺序仍然不固定，但每个 `(gx, gy)` 坐标应该只出现一次。

## E1.3 输出超出最大线程数会怎样（边界）

把启动配置改成 `<<<1, 4096>>>`。在你设备上能跑吗？
对比 `03_device_info` 输出的 `Max threads per block`。

写一段失败时的错误信息到 `notes/week1.md`。

预计输出 / 预期现象：

```text
CPU: Hello world!
MUSA error ... invalid configuration argument ...
```

- 如果设备的 `Max threads per block < 4096`，kernel 不会正常启动，通常会在 `musaGetLastError` 或 `musaDeviceSynchronize` 处报 launch configuration 类错误。
- 如果示例里没有做错误检查，可能只看到 CPU 输出，看不到 GPU 输出；这也是要加错误检查的原因。

## E1.4 设备查询扩展（动手）

仿照 `03_device_info.mu`，再打印这些字段（如 SDK 提供）：
- `regsPerBlock`（每 block 寄存器数）
- `memoryClockRate`（显存频率）
- `memoryBusWidth`（显存位宽）
- `l2CacheSize`（L2 大小）

算出理论显存带宽：`bw = 2 * memoryClockRate * memoryBusWidth / 8 / 1e6` (GB/s)。

预计输出 / 预期现象：

```text
Device 0: ...
SM count: ...
Warp size: 128
Max threads per block: ...
regsPerBlock: ...
memoryClockRate: ...
memoryBusWidth: ...
l2CacheSize: ...
Theoretical memory bandwidth: ... GB/s
```

- 具体数字跟设备和 SDK 字段支持情况有关。
- 如果某个字段编译不过，说明当前 MUSA SDK 的 `musaDeviceProp` 没暴露该字段，记录“SDK 不支持此字段”即可。

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

预计输出 / 预期现象：

这题不运行程序，产物应该是 `notes/cuda-to-musa.md` 里的一张对照表。最低应包含类似：

| CUDA | MUSA |
|---|---|
| `cudaMalloc` | `musaMalloc` |
| `cudaFree` | `musaFree` |
| `cudaMemcpy` | `musaMemcpy` |
| `cudaDeviceSynchronize` | `musaDeviceSynchronize` |
| `cudaGetLastError` | `musaGetLastError` |
| `nvcc` | `mcc` |

## E1.6 显存指针不能在 host 用（边界）

修改 `04_memory_basics.mu`，把 `musaMemcpy(h, d, ...)` 那一行删掉，
直接 `printf("%f\n", d[0]);`。

预测会发生什么？运行验证。这是为什么？

> 提示：写笔记时把命名规范也定下来 —— `h_` 前缀 = host 指针，`d_` 前缀 = device 指针。

预计输出 / 预期现象：

```text
Segmentation fault
```

或程序直接崩溃 / 打印异常值。`d` 是 device pointer，host CPU 不能直接解引用；正确做法是先 `musaMemcpy` 拷回 `h_` 指针，再在 CPU 侧打印。

## E1.7 musaMemset 的"按字节"陷阱（基础）

把 `04_memory_basics.mu` 里的 `fill_const` kernel 调用注释掉，
然后在 `musaMemset(d, 0, BYTES)` 后面再加一行 `musaMemset(d, 1, BYTES)`，
拷回 host 后打印 `h[0]`。

得到的不是 1.0f，而是 ~2.36e-38。解释这个数字怎么来的。

预计输出 / 预期现象：

```text
h[0] = 0.000000
```

如果用科学计数法打印：

```text
h[0] = 2.369428e-38
```

`musaMemset(d, 1, BYTES)` 写的是每个字节 `0x01`，一个 float 变成 bit pattern `0x01010101`，按 IEEE 754 解释约为 `2.36e-38`，不是 `1.0f`。

## E1.8 主动触发各类错误（动手）

按 `05_error_check.mu` 的写法，再设计两个错误场景：

- (a) 给 `musaMemcpy` 传错方向（比如把 H→D 拷贝写成 `musaMemcpyDeviceToHost`）
- (b) `musaMalloc` 0 字节，再 `musaFree` 这个指针，会出错吗？

把每种情况下 `musaGetLastError` / `musaDeviceSynchronize` 的返回码记录到 `notes/week1.md`。

预计输出 / 预期现象：

```text
MUSA error ... at 05_error_check.mu:...
```

- 传错 `musaMemcpy` 方向通常会立刻返回错误，因为 runtime 能发现 host/device 指针和方向不匹配。
- `musaMalloc(0)` 在不同 SDK 中可能返回成功并给出空指针，也可能返回错误；以你机器上的返回码为准。
- `musaGetLastError` 主要抓最近一次 launch/runtime 错误；`musaDeviceSynchronize` 会暴露异步 kernel 错误。

## E1.9 launch overhead 测量（动手）

修改 `06_async_kernel.mu`，把 `LOOPS` 调到 `100`（几乎不算）和 `50000000`（很重）两组。

填这张表：

| LOOPS | t_launch (ms) | t_wait (ms) | 比例 |
|---|---|---|---|
| 100 | ? | ? | ? |
| 5_000_000 | ? | ? | ? |
| 50_000_000 | ? | ? | ? |

回答：什么场景下 `t_launch` 会成为瓶颈？这预示了下周哪个特性的价值？

预计输出 / 预期现象：

```text
LOOPS=100       t_launch=... ms  t_wait=... ms
LOOPS=5000000   t_launch=... ms  t_wait=... ms
LOOPS=50000000  t_launch=... ms  t_wait=... ms
```

- `t_launch` 通常变化不大，它主要是 CPU 把 kernel 入队的开销。
- `t_wait` 会随 `LOOPS` 明显变大，因为 GPU 真正计算变重。
- 当 kernel 很小、调用次数很多时，launch overhead 会成为瓶颈；这引出 Week 2 的 Stream / Graph。

## E1.10 把同步删了会怎样（思考 + 验证）

修改 `06_async_kernel.mu`：
- (a) 删掉 `musaDeviceSynchronize()`，立刻 `return 0`。kernel 还会跑完吗？
- (b) 在删掉同步的版本后面加一句 `musaMemcpy(&host_sink, d_sink, ..., D2H)`。结果对吗？

写下你的观察，并解释 musaMemcpy 的"隐式同步"是怎么救你的。

预计输出 / 预期现象：

- (a) 可能看不到 GPU `printf`，或者 kernel 还没 flush 程序就退出；不能依赖它跑完。
- (b) 加了 D2H `musaMemcpy` 后结果通常正确，因为默认流里的 D2H 拷贝会等待前面的 kernel 完成。

```text
launch returned immediately
host_sink = <正确结果>
```

关键点：kernel launch 是异步的，`musaMemcpyDeviceToHost` 在默认流上有隐式同步效果。
