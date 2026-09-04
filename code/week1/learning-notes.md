# Week 1 学习材料

Week 1 的目标不是写复杂 kernel，而是建立 GPU 程序的最小心智模型：host 发起工作，device 并行执行，数据通过显式内存拷贝在两侧流动，错误和结果都要通过同步点暴露。

## 阅读顺序

1. `01_hello_world.mu`: 先确认 kernel 启动语法和 GPU printf 行为。
2. `02_thread_index.mu`: 再理解每个 thread 如何得到自己的全局编号。
3. `03_device_info.mu`: 把抽象的 grid/block 放到真实硬件参数里看。
4. `04_memory_basics.mu`: 建立 host/device 两套内存的边界。
5. `05_error_check.mu`: 学会 launch 后的两步错误检查。
6. `06_async_kernel.mu`: 理解 kernel launch 为什么不是普通函数调用。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_hello_world.mu` | `__global__`、`<<<grid, block>>>`、`musaDeviceSynchronize()` | 以为 GPU printf 顺序代表执行顺序 |
| `02_thread_index.mu` | `blockIdx.x * blockDim.x + threadIdx.x` | 只用 `threadIdx.x` 当全局索引 |
| `03_device_info.mu` | `musaGetDeviceCount`、`musaGetDeviceProperties` | 写死 warp size、block 上限和显存参数 |
| `04_memory_basics.mu` | `musaMalloc`、`musaMemcpy`、`musaFree` | 在 host 直接解引用 device pointer |
| `05_error_check.mu` | `musaGetLastError` + `musaDeviceSynchronize` | launch 错误和执行错误混为一谈 |
| `06_async_kernel.mu` | launch 入队、同步等待、隐式同步 | 用 CPU clock 只包 launch 计 kernel 时间 |

## 代码阅读抓手

读每个 `.mu` 文件时先找三类代码：

- launch config: `<<<grid, block>>>` 决定并行规模。
- index formula: `blockIdx.x * blockDim.x + threadIdx.x` 决定每个线程处理哪份数据。
- sync/error check: `musaGetLastError()` 和 `musaDeviceSynchronize()` 决定错误在哪里暴露。

Week 1 跑通后再看 Week 2。否则 stream、event、graph 里的异步行为会很容易误判。

## CUDA_Freshman 对照

优先看这些 CUDA 主题，然后回到 MUSA 版本改写：

- `0_hello_world`: 对照 `01_hello_world.mu`
- `1_check_dimension` / `2_grid_block` / `5_thread_index`: 对照 `02_thread_index.mu`
- `7_device_information`: 对照 `03_device_info.mu`

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
