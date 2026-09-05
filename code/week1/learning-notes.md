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

## Grid / Block / Thread 容易混淆点

先把一句话记牢：

```text
一次 kernel launch = 1 个 grid
1 个 grid          = 很多个 block
1 个 block         = 很多个 thread
```

示意图：

```text
kernel<<<grid, block>>>
          │      │
          │      └─ blockDim: 每个 block 里有多少 thread
          └──────── gridDim:  grid 里有多少 block

Kernel Launch
└── Grid
    ├── Block 0
    │   ├── Thread 0
    │   ├── Thread 1
    │   ├── ...
    │   └── Thread blockDim.x - 1
    ├── Block 1
    │   ├── Thread 0
    │   ├── Thread 1
    │   └── ...
    └── Block gridDim.x - 1
        └── ...
```

`kernel<<<grid, block>>>(...)` 里的两个参数不是“线程编号”，而是“启动形状”：

- `grid` / `gridDim`: grid 里有多少个 block。
- `block` / `blockDim`: 每个 block 里有多少个 thread。
- `blockIdx`: 当前 block 在 grid 里的坐标。
- `threadIdx`: 当前 thread 在 block 里的坐标。

### 一维索引：最常见的数组写法

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

这个公式的意思是：

```text
全局索引 = 前面 block 已经覆盖的线程数 + 当前 block 内的局部线程号
```

假设 `blockDim.x = 8`：

```text
blockDim.x = 8

┌──────────────────── blockIdx.x = 0 ────────────────────┐
│ threadIdx.x:  0   1   2   3   4   5   6   7             │
│ global idx:   0   1   2   3   4   5   6   7             │
└─────────────────────────────────────────────────────────┘

┌──────────────────── blockIdx.x = 1 ────────────────────┐
│ threadIdx.x:  0   1   2   3   4   5   6   7             │
│ global idx:   8   9  10  11  12  13  14  15             │
└─────────────────────────────────────────────────────────┘

┌──────────────────── blockIdx.x = 2 ────────────────────┐
│ threadIdx.x:  0   1   2   3   4   5   6   7             │
│ global idx:  16  17  18  19  20  21  22  23             │
└─────────────────────────────────────────────────────────┘

global idx = blockIdx.x * blockDim.x + threadIdx.x
```

最容易错的是只写：

```cpp
int idx = threadIdx.x;
```

这只在单 block 示例里“看起来没问题”。一旦有多个 block，所有 block 都会得到同一批 `0..blockDim.x-1`，不同 block 会重复写同一段数组。

### 为什么需要 `if (idx < N)`

block 数通常这样算：

```cpp
int blocks = (N + threads - 1) / threads;
```

它保证“线程总数不少于 N”，不保证“刚好等于 N”。

例如 `N = 10`、`threads = 8`：

```text
blocks = (10 + 8 - 1) / 8 = 2
实际启动线程数 = 2 * 8 = 16
有效数据下标 = 0..9
多出来的线程 = 10..15
```

所以 kernel 里要写：

```cpp
if (idx < N) {
    out[idx] = in[idx];
}
```

或者更直接地：

```cpp
if (idx >= N) return;
```

这不是性能优化，而是边界正确性。

### 二维索引：矩阵和图像

二维时不要把 `x/y` 想成“第几个线程”，而要想成“列/行坐标”：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

如果矩阵按 row-major 存在一维数组里：

```cpp
int idx = row * width + col;
```

示意图：

```text
grid 是二维 block 网格:

          blockIdx.x →
            0          1          2
blockIdx.y  ┌────────┬────────┬────────┐
     0      │ (0,0)  │ (1,0)  │ (2,0)  │
            ├────────┼────────┼────────┤
     1      │ (0,1)  │ (1,1)  │ (2,1)  │
            └────────┴────────┴────────┘

选中 blockIdx = (1, 1), blockDim = (4, 3):

threadIdx.y
   ↓
0      (0,0)  (1,0)  (2,0)  (3,0)
1      (0,1)  (1,1)  (2,1)  (3,1)
2      (0,2)  (1,2)  (2,2)  (3,2)
          └──────── threadIdx.x →

如果当前 threadIdx = (2, 1):

col = blockIdx.x * blockDim.x + threadIdx.x = 1 * 4 + 2 = 6
row = blockIdx.y * blockDim.y + threadIdx.y = 1 * 3 + 1 = 4

矩阵 row-major 展平:

row 0: [ 0] [ 1] [ 2] [ 3] [ 4] [ 5] [ 6] [ 7]
row 1: [ 8] [ 9] [10] [11] [12] [13] [14] [15]
row 2: [16] [17] [18] [19] [20] [21] [22] [23]
row 3: [24] [25] [26] [27] [28] [29] [30] [31]
row 4: [32] [33] [34] [35] [36] [37] [38] [39]
                                      ↑
                         idx = row * width + col
                             = 4 * 8 + 6 = 38
```

完整模式：

```cpp
__global__ void fill(float* a, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) return;

    int idx = row * width + col;
    a[idx] = 1.0f;
}
```

启动时也用二维：

```cpp
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);
```

这里的 `width` 是列数，`height` 是行数。常见错误是把 `row/col` 写反，或者展平时写成 `col * height + row`，导致矩阵看起来像被转置或乱写。

### block、thread、warp 不要混在一起

Week 1 先记这三个层级：

| 概念 | 你能控制吗 | 主要作用 |
|---|---|---|
| `thread` | 间接控制数量 | 真正执行 kernel 代码的最小单位 |
| `block` | launch 时控制 | 线程协作边界，可用 shared memory 和 `__syncthreads()` |
| `warp` | 不直接在 launch 里写 | 硬件实际成组执行线程 |

CUDA 里常见 warp size 是 32；MUSA 上不要照搬 CUDA 的 warp 假设，应该从设备属性里查：

```cpp
musaDeviceProp prop;
musaGetDeviceProperties(&prop, 0);
printf("warpSize = %d\n", prop.warpSize);
```

Week 1 的普通索引公式不依赖 warp size；后面写 shuffle、warp reduce、occupancy 调参时才会真正踩到这个差异。

### Week 1 高频混淆清单

- **`<<<2, 4>>>` 是 8 个线程**: 2 是 block 数, 4 是每个 block 的 thread 数。
- **`threadIdx` 会在每个 block 内重新从 0 开始**: 多 block 时必须加上 `blockIdx * blockDim`。
- **host 代码里没有 `threadIdx`**: 这些内置变量只存在于 kernel/device 代码。
- **device pointer 不是普通 C 指针**: host 不能直接读写 `d_ptr[0]`, 要先 `musaMemcpy`。
- **`musaMemset` 按字节填充**: 设 0 常用且安全, 设 1 不等于 float/int 的数值 1。
- **launch 错误和执行错误暴露时机不同**: launch 配置错通常 `musaGetLastError()` 抓; kernel 内越界通常同步点才暴露。
- **GPU `printf` 顺序不可靠**: 输出顺序不是线程执行顺序, 只能用来辅助看索引。
- **MUSA 和 CUDA 语法像, 性能假设不一定像**: 普通索引写法能迁移, warp size、架构参数、库和 profiler 不能盲搬。

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
