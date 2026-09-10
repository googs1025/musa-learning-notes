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

### 写 kernel 时，一般怎么思考 grid / block

先记住一个最常见的思路：

```text
先决定 1 个 thread 负责什么
    ↓
再决定 1 个 block 放多少个 thread
    ↓
最后算需要多少个 block 去覆盖全部数据
```

以最常见的一维数组为例，通常会这样写：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    ...
}
```

对应的 launch 形状一般这样定：

- `threadsPerBlock`：先选一个常见经验值，比如 `128 / 256 / 512`，其中 `256` 最常见。
- `blocksPerGrid`：用向上取整把 `N` 个元素全部覆盖住。

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
```

为什么这样写：

- `threadsPerBlock` 不是“越大越好”，而是先选一个容易调度、又常见的 block 大小。
- `blocksPerGrid` 必须保证总线程数不少于 `N`，不然有些元素根本没人处理。
- 因为是“向上取整”，最后一个 block 往往会多出一些线程，所以 kernel 里要写 `if (idx < N)` 防止越界。

一个直观例子：

```text
N = 1000, threadsPerBlock = 256
blocksPerGrid = (1000 + 256 - 1) / 256 = 4

总线程数 = 4 * 256 = 1024
前 1000 个线程处理有效元素
后 24 个线程没有对应元素，要靠 if (idx < N) 拦住
```

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

把它代入一个具体例子就更直观了：

```text
假设 blockDim.x = 8，blockIdx.x = 2

global idx = blockIdx.x * blockDim.x + threadIdx.x
           = 2 * 8 + threadIdx.x
           = 16 + threadIdx.x
```

所以这个 block 里的 8 个线程会依次得到：

```text
threadIdx.x:  0  1  2  3  4  5  6  7
global idx:  16 17 18 19 20 21 22 23
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

这里的 `width` 是列数，`height` 是行数。

原因是 row-major 的内存布局是“先把一整行放完，再放下一行”：

```text
row 0:  [0] [1] [2] [3] [4] [5] [6] [7]
row 1:  [8] [9] [10][11][12][13][14][15]
row 2:  [16][17][18][19][20][21][22][23]
```

所以：

- `row` 决定你跳过多少“完整的行”；
- 每一行有 `width` 个元素；
- 因此 `row` 前面要乘 `width`；
- `col` 是当前行里的偏移，直接加上去。

如果误写成 `col * height + row`，就等于把“列优先”当成“行优先”了。这样会把一维下标的增长方向弄反，常见结果就是图像看起来像被转置，或者写回的位置错位。

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

再看坐标对应关系：

```text
(row, col) = (4, 6)

行方向先走 4 行，每行 8 个元素：
4 * 8 = 32

再在第 4 行里往右走 6 个：
32 + 6 = 38
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

### Grid、Block、Warp、Thread 的层级关系

可以把一次 kernel 的执行层级记成：

```text
一次 kernel launch
└── Grid：整个任务
    ├── Block：任务分组
    │   ├── Warp：硬件执行小组
    │   │   └── Thread
    │   └── Warp
    └── Block
```

它们分别处在不同层面：

| 层级 | 含义 | 主要职责 |
|---|---|---|
| Grid | 一次 kernel launch 产生的全部 block | 描述整个任务的规模 |
| Block | 一组可以协作的 thread | 作为 SM 的资源分配和驻留单位 |
| Warp | block 内由硬件成组执行的 thread | 作为硬件发射和执行指令的基本小组 |
| Thread | 逻辑上的单个线程 | 处理一个或多个数据元素 |

例如：

```cpp
kernel<<<4, 256>>>();
```

表示 1 个 grid 中有 4 个 block，每个 block 有 256 个 thread。假设设备的 warp size 是 32，那么每个 block 会被拆成 8 个 warp，整个 kernel 共包含 32 个 warp。实际 warp 数量要根据设备的 `warpSize` 计算，不能默认所有 MUSA 设备都和常见 CUDA 设备一样使用 32。

这里“block 是基本调度单元”和“warp 是实际执行单元”并不矛盾：

- block 是资源分配单位：一个 block 通常整体放到一个 SM 上，不能拆到多个 SM；它占用的寄存器、shared memory 和线程数量决定一个 SM 能同时驻留多少个 block。
- warp 是指令执行单位：SM 发射指令时，通常以 warp 为小组让多条线程一起执行同一条指令。
- thread 是编程时的逻辑视角：源码用 `threadIdx` 描述每个线程负责的元素，但硬件会把这些 thread 按 warp 组织起来执行。

因此，kernel 启动语法只写 grid 和 block：

```cpp
kernel<<<grid, block>>>(args);
```

grid 和 block 是程序员需要定义的逻辑工作划分；warp 是运行时根据设备硬件自动从 block 中切分出来的执行组织。如果 block 中的线程在分支上走了不同路径，同一个 warp 可能需要依次执行不同路径，这就是 warp divergence。后续的 shuffle、warp reduce 和 lane 操作才会直接使用 warp 概念。

### MUSA 硬件层级：MPC、MPX、MP 与线程层级的关系

上面的 Grid/Block/Warp/Thread 是编程模型；MUSA 硬件还可以从物理组织角度理解为：

```text
MUSA GPU
└── MPC：MUSA Processor Cluster，处理器簇
    └── MPX：MUSA Processor eXecution engine，处理器执行引擎
        └── MP：MUSA Processor，MUSA 处理器
            └── warp：由 MP 调度执行的线程束
                └── thread：线程
```

在官方描述的 MP10/MP21 架构模型中，一个 MPC 包含多个 MPX，一个 MPX 又包含多个 MP；MP 负责 SIMT warp 的创建、管理、调度和执行。不同架构代际的数量、缓存和计算单元可能不同，所以这里应该把它理解成“硬件层级关系”，不要把某一代的具体数量写死到通用代码里。

| 编程模型 | 硬件视角 | 关系 |
|---|---|---|
| Grid | 整个 GPU 上的一次任务 | 由很多 block 组成，运行时分配到可用的 MP 上 |
| Block | MP 上的驻留/资源分配对象 | block 内线程共享资源，并被划分成多个 warp |
| Warp | MP 上的 SIMT 执行小组 | 由 warp 调度器选择并发射指令 |
| Thread | warp 中的逻辑线程 | 每个线程拥有自己的寄存器状态和数据索引 |
| MPC / MPX / MP | GPU 的物理组织层级 | 决定计算单元、缓存和调度资源如何组合 |

这也解释了为什么源码通常只写：

```cpp
kernel<<<grid, block>>>(args);
```

程序员定义 grid 和 block，运行时负责把 block 分配到 MP，并在 MP 内把 block 切分成 warp 执行。程序一般不直接指定“把这个 block 放到某个 MPC/MPX/MP”，这样同一份 kernel 才能在不同规模和不同代际的 MUSA GPU 上自动扩展。

需要特别区分两个同名缩写：

- 硬件架构语境中的 **MPC** 是 MUSA Processor Cluster，表示处理器簇。
- 设备管理/虚拟化语境中的 **MPC** 也可能表示 Multiple Primary Core，用于把一块物理 GPU 切分成多个资源隔离的逻辑实例；它不是上面硬件层级中的处理器簇。
- MUSA 官方硬件架构资料中，常用的层级名称是 `MPC`、`MPX`、`MP`。`MPE` 不是本节采用的标准硬件层级名称；遇到具体 SDK、芯片白皮书或工具输出中的 `MPE`，应以对应版本文档定义为准，不能直接当成 MP 的同义词。

参考：[
MUSA 硬件架构](https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/history_version/rc4.3/programming_guide/Chapter02/)、[
MUSA 线程层次结构](https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/programming_guide/programming_model/thread_hierarchy/)。

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

## 逐示例课文

### `01_hello_world.mu`：第一次启动 GPU 工作

#### 示例目标

这个示例回答“CPU 如何让 GPU 执行一段函数”这一最小问题。它不处理输入输出数据，而是让 CPU 打印一行，再启动 GPU kernel，让 GPU 上的五个线程各打印一行，从而把 host、device 和一次 kernel launch 的边界变成可观察的输出。

#### 代码结构

文件由一个 `__global__` 修饰的 `hello_from_gpu` kernel 和 `main` 组成。kernel 读取 block 内的 `threadIdx.x` 并打印线程号；`main` 先输出 CPU 文本，再以 `<<<1, 5>>>` 启动一个 block 的五个线程，最后调用 `musaDeviceSynchronize()`。

#### 核心知识点

`__global__` 表示函数在 device 上执行、由 host 调用且返回 `void`。`<<<grid, block>>>` 是执行配置：这里是 1 个 block、每个 block 5 个 thread，总共 5 个线程。GPU `printf` 先进入设备端缓冲区，结束处的同步既等待 kernel 完成，也让这批输出被 flush；各线程的输出顺序不能当作调度顺序。

#### 执行流程

程序先在 host 打印 `CPU: Hello world!`，然后把 kernel 请求放入默认执行队列。五个线程并行执行同一份 kernel，各自得到本 block 内的 `tid=0..4` 并写入 printf 缓冲区；host 随后在 `musaDeviceSynchronize()` 处等待，输出缓冲区才可靠地显示。

#### 常见错误与实验

把同步删掉，可能在进程退出前看不到 GPU 输出；把 `<<<1, 5>>>` 改为 `<<<2, 4>>>` 或 `<<<4, 8>>>`，可验证输出线程数分别变为 8 和 32。观察这些输出时要注意：只打印 `threadIdx.x` 无法区分不同 block 中重复出现的 `tid`，而且 GPU 行顺序不保证固定。

#### 与本周其他示例的关系

这是 Week 1 的入口：它只展示启动和完成，不展示数据索引。`02_thread_index.mu` 接着解决多 block 下如何得到唯一全局编号；`04_memory_basics.mu` 展示 kernel 结果如何通过显式内存拷贝传回 host；`06_async_kernel.mu` 则把这里隐含的异步 launch 用计时放大出来。

### `02_thread_index.mu`：从局部线程号得到全局位置

#### 示例目标

当 grid 中有多个 block 时，单独使用 `threadIdx.x` 会让不同 block 得到相同的线程号。这个示例用 2 个 block、每个 4 个线程，说明如何把 block 坐标和 block 内线程坐标组合成覆盖整个 grid 的全局线程编号。

#### 代码结构

`print_index` kernel 读取 `threadIdx.x`、`blockIdx.x` 和 `blockDim.x`，计算 `global = bid * bdim + tid`，并把四个相关值一起打印。host 端用 `dim3 grid(2)` 和 `dim3 block(4)` 配置 launch，再同步等待输出。

#### 核心知识点

一维全局索引公式是 `blockIdx.x * blockDim.x + threadIdx.x`：前面 block 的线程数加上当前 block 内的偏移。`threadIdx` 和 `blockIdx` 是当前层级内的坐标，`blockDim` 表示每个 block 的形状，`gridDim` 表示 grid 的形状；`dim3` 未显式给出的维度默认为 1。二维或三维数据则分别沿 x、y、z 维套用同样的组合关系。

#### 执行流程

launch 展开为 2 个 block，每个 block 内的 `threadIdx.x` 都从 0 到 3 重新开始。第 0 个 block 产生 global 0 到 3，第 1 个 block 产生 global 4 到 7；每个线程将自己的局部坐标、所属 block 和全局编号写入输出，host 在同步点等待全部线程完成。

#### 常见错误与实验

实验可把 grid 改成 `(3, 2)`、block 改成 `(4, 2)`，同时打印 `.x/.y`，验证全局坐标范围为 `gx=0..11`、`gy=0..3`，总线程数为 48。常见错误是继续只用 `threadIdx.x`，或把二维的行列方向混用；面对真实数组，还要在全局索引后增加边界判断，避免向上取整后的多余线程访问无效数据。

#### 与本周其他示例的关系

它把 `01_hello_world.mu` 中“每个线程都执行同一 kernel”推进为“每个线程负责不同位置”。`04_memory_basics.mu` 直接复用这一维索引公式填充数组；`03_device_info.mu` 提供决定 block 形状时需要查阅的硬件上限；后续矩阵类示例则把这个公式扩展到二维。

### `03_device_info.mu`：让 launch 配置面对真实硬件

#### 示例目标

grid 和 block 不是脱离设备的抽象数字。这个示例解决“当前机器有几张 MUSA GPU、每张卡能承受怎样的并行配置”这一探查问题，为后续选择 block 大小、估算资源和判断 launch 合法性提供运行时依据。

#### 代码结构

程序定义统一返回错误的 `CHECK` 宏。`main` 先调用 `musaGetDeviceCount` 得到设备数量，再循环调用 `musaGetDeviceProperties`，将每张设备的名称、SM 数量、warp 大小、每 block 最大线程数、shared memory、总显存和 compute capability 打印出来。属性对象使用值初始化后再传给 Runtime API。

#### 核心知识点

设备属性是运行时事实，不应把 warp size、`maxThreadsPerBlock` 或显存容量永久写死。`warpSize` 和 `multiProcessorCount` 可帮助理解调度与 occupancy，`sharedMemPerBlock` 表示 block 级资源约束，`totalGlobalMem` 反映容量；`musaGetDeviceCount` 与按索引查询的组合也建立了多设备遍历模型。示例中的 API 调用都经 `CHECK` 检查返回的 `musaError_t`。

#### 执行流程

host 先获取 count；对每个设备索引 `i` 查询属性并按格式输出，循环结束后退出。文件本身不启动 kernel，也不分配显存，因此它的输出主要是机器相关的规格清单，而不是计算结果。若要在某张卡上执行工作，练习中指出应在对应工作前使用 `musaSetDevice(i)`。

#### 常见错误与实验

可尝试把一个 kernel 配置成 `<<<1, 4096>>>`，对照查询到的 `maxThreadsPerBlock` 理解为什么 launch 会因单 block 线程过多而失败，并观察错误检查的重要性。还可以扩展打印 `regsPerBlock`、`memoryClockRate`、`memoryBusWidth` 和 `l2CacheSize`，再按文件给出的公式估算理论带宽；不要把 CUDA 常见的 warp size 直接当成所有 MUSA 设备的值。

#### 与本周其他示例的关系

这是对 `01_hello_world.mu` 和 `02_thread_index.mu` 中启动形状的硬件侧补充：前两个示例说明“怎么写”，本例说明“设备允许什么”。它也为 `05_error_check.mu` 中非法配置的实验提供判断依据，并为后续带宽、occupancy 等主题提供属性来源。

### `04_memory_basics.mu`：让数据跨过 host/device 边界

#### 示例目标

启动 kernel 只是让 GPU 做事；真正的程序还要分配数据、把结果带回来并释放资源。本例用长度为 16 的 float 数组，把 host 内存和 device 显存之间的边界、显式拷贝以及生命周期压缩成一条可验证的最小数据流。

#### 代码结构

`fill_const` kernel 用全局索引定位元素，并在 `i < n` 时写入常数。host 端计算 `BYTES`，用 `malloc` 建立结果缓冲区，用 `musaMalloc` 建立 device 缓冲区，随后 `musaMemset` 清零、launch kernel、检查并同步，再通过 D2H `musaMemcpy` 拷回，逐元素验证 `3.14f`，最后分别 `musaFree` 和 `free`。

#### 核心知识点

host 指针和 device 指针属于不同地址空间，不能互相直接解引用；两侧之间要通过 `musaMemcpy`，并用方向参数说明 H2D、D2H 等关系。`musaMemset` 是按字节填充，清零适合初始化零值，但填充字节 1 不等于把 float 元素设成 1.0。向上取整得到的 block 数可能多于数据需要，因此 kernel 的边界判断是正确性条件。

#### 执行流程

本例没有输入数组：先分配 16 个 float 的两侧空间，把显存清零；`threads=64`、`blocks=(N+threads-1)/threads` 得到一个 block，kernel 的前 16 个有效线程写入 3.14，其余线程因边界判断不写；同步后把 device 结果复制到 host，host 检查并打印前四项，最后释放两侧内存。

#### 常见错误与实验

删除 D2H 拷贝并在 host 直接读取 `d[0]`，会触发非法访问；把 `musaMemset(d, 0, BYTES)` 改成字节值 1，可观察它得到的是 `0x01010101` 这样的 float 位模式，而不是 1.0。还可以将 `VAL` 改为别的常数验证 kernel 填充逻辑，并检查错误路径上资源释放的问题。

#### 与本周其他示例的关系

本例把 `02_thread_index.mu` 的全局索引用于真实数组，把 `01_hello_world.mu` 的“同步后观察 GPU 行为”变成“同步后拷回并验证结果”。它也为 `05_error_check.mu` 的非法地址实验提供了 device pointer 背景，并和 `06_async_kernel.mu` 共享 `musaMalloc`、同步及释放的生命周期概念。

### `05_error_check.mu`：区分 launch 错误和执行错误

#### 示例目标

GPU 程序的错误不一定在发生的那一行返回。这个示例故意制造正常 launch、非法 block 配置、kernel 写 null 指针和超大显存申请四种情形，说明应该在什么时机检查错误，避免错误状态被后续 API 掩盖。

#### 代码结构

文件用不退出程序的 `CHECK_SOFT` 宏记录错误并继续演示。`noop` 作为正常 kernel，`write_bad` 将 `p[0]` 写入传入指针。`main` 依次执行正常 case、`noop<<<1, 99999>>>`、把 null 指针传给 `write_bad`，以及申请 100 TiB 的 `musaMalloc`，分别在关键位置调用错误检查。

#### 核心知识点

直接返回值可以检查 `musaMalloc` 等 API；kernel launch 没有普通返回值，所以先用 `musaGetLastError()` 检查配置等同步错误，再用 `musaDeviceSynchronize()` 检查执行期间的异步错误。合法的 launch 只说明请求能入队，不说明 kernel 内的指针访问一定合法；null 指针写入通常要到同步点才报告 illegal address。

#### 执行流程

正常 case 中两个检查都应显示成功。非法配置在 `musaGetLastError()` 处被发现，kernel 实际不会运行；null 指针 case 的 launch 形状仍合法，因此第一次检查通常成功，GPU 执行到写操作时才在同步处暴露错误；最后的超大分配直接检查 `musaMalloc` 返回值，并按结果打印预期失败或异常成功。

#### 常见错误与实验

可分别注释掉 `musaGetLastError()` 或 `musaDeviceSynchronize()`，观察 launch 配置错误和执行期错误各自如何被遗漏；还可以用 `musaGetErrorString` 或 `musaGetErrorName` 将错误码翻译成可读文本。示例中的错误码和最终输出随 SDK、设备而变，不应把某一次运行的数字当成跨环境保证。

#### 与本周其他示例的关系

它把 `03_device_info.mu` 查询到的 block 上限转化为失败实验，把 `04_memory_basics.mu` 中“device 指针不能由 host 直接当普通指针使用”推进为 kernel 内非法访问诊断。`06_async_kernel.mu` 继续说明为什么同步点既是等待位置，也是发现执行错误的重要位置。

### `06_async_kernel.mu`：把异步 launch 变成可测的时间差

#### 示例目标

普通 CPU 计时包住 `kernel<<<>>>` 时，测到的通常只是提交开销，不是 GPU 真正执行时间。本例用一个故意很慢的 kernel，把“launch 立即返回”和“同步等待计算完成”拆成两个时间段，建立正确的异步执行直觉。

#### 代码结构

`busy_kernel` 的每个线程执行 `LOOPS=5'000'000` 次累加，并用 `volatile` 与 `sink[0]` 的副作用阻止循环和结果被优化掉；只有第一个 block 的第一个线程写 sink。host 端先分配一个 int 的 device sink，使用 `std::chrono::high_resolution_clock` 分别测 launch 返回时间 `t_launch` 和同步等待阶段耗时 `t_wait`；后者包含 host 等待设备完成的阶段，并不是纯 GPU kernel 时间。两者相加得到 `t_total`，最后释放 sink。

#### 核心知识点

kernel launch 是把工作放入队列，host 不会自动等待 GPU 完成；`musaDeviceSynchronize()` 等待所有流上的任务，D2H `musaMemcpy` 也可在拷贝前隐式等待。CPU wall clock 直接包住 launch 只能得到提交阶段，必须把同步包含进测量，或使用后续示例介绍的 GPU event 做更准确的设备侧计时。

#### 执行流程

程序先在 device 上分配 sink，按 `blocks=64`、`threads=128` 启动 8192 个线程。CPU 记录 launch 前后时间并立即继续；GPU 在线程中执行长循环。随后 CPU 在同步点阻塞，直到默认流中的 kernel 完成，再输出 `t_launch`、`t_wait` 和两者之和，并释放显存。循环足够长时，`t_wait` 应明显大于提交开销，但具体数值取决于设备和负载。

#### 常见错误与实验

把 `LOOPS` 改成 100，对比 `t_launch` 基本不随 kernel 内循环改变而 `t_wait` 明显缩短；删除显式同步后不要直接把 `d_sink` 当 host 数据读取，若后续接 D2H 拷贝则拷贝本身会等待。还可以连续提交多个 kernel，再用一次同步观察默认流中的排队关系；真正的并发 stream 属于后续 Week 2 内容。

#### 与本周其他示例的关系

它解释了 `01_hello_world.mu` 为什么需要同步才能稳定看到 GPU 输出，也解释了 `04_memory_basics.mu` 中同步后再 D2H 的时序。`05_error_check.mu` 展示同步如何暴露异步执行错误；本例则把同一个时序机制用于性能测量，并为 Week 2 的 stream、event 和 graph 铺路。

## CUDA_Freshman 对照

优先看这些 CUDA 主题，然后回到 MUSA 版本改写：

- `0_hello_world`: 对照 `01_hello_world.mu`
- `1_check_dimension` / `2_grid_block` / `5_thread_index`: 对照 `02_thread_index.mu`
- `7_device_information`: 对照 `03_device_info.mu`

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
