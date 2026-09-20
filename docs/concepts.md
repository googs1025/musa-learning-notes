# MUSA 基础概念

> 本文配合 [`code/week1/`](../code/week1/) 阅读，补充代码注释里没有展开的概念和执行模型。

---

## 1. SIMT 模型 vs CPU 多线程

| | CPU 多线程 | GPU SIMT |
|---|---|---|
| 线程数量级 | 几个 ~ 几十 | 几千 ~ 几百万 |
| 调度单位 | 单个 thread | **warp**(MUSA = 128 线程,NVIDIA = 32) |
| 指令视角 | 每个线程跑自己的指令流 | warp 内**锁步执行同一条指令** |
| 切换代价 | 高(寄存器保存/恢复)| 极低(寄存器都在 SM 里,切换就是改个指针) |
| 适合任务 | 控制流复杂、分支多 | 数据并行,大量同样的小操作 |

**SIMT** 全称 Single Instruction, Multiple Thread。可以先记住两点：

- 一个 warp 内 128 线程**共享同一条指令的执行**;
- 如果 warp 内的线程走了 `if/else` 不同分支(**warp divergence**),硬件会**串行执行两个分支**(先跑 if 那部分,再跑 else 那部分),其他线程被 mask 掉,等于浪费算力;
- 写 kernel 时尽量让同一 warp 内的线程走同一条路径。

GPU 不需要像 CPU 那样保存和恢复完整的线程上下文。一个 warp 等待内存访问时，SM 可以立刻让另一个已就绪的 warp 执行，寄存器和 shared memory 仍留在 SM 上。GPU 用这种方式隐藏内存延迟。

SM 上有更多驻留 warp 时，通常更容易隐藏延迟。Week 4 会继续调整 occupancy 并观察它对性能的影响。

---

## 2. 硬件层级:SM → Warp → Block → Thread

```
GPU
 ├── SM 0  (Streaming Multiprocessor)        ← 物理调度单位
 │    ├── Warp scheduler × N                  
 │    ├── Register file (~64 KB)              
 │    ├── Shared memory / L1 (~96 KB)         ← 同 block 共享
 │    └── ALU/FPU/SFU/Tensor 等执行单元        
 ├── SM 1
 ├── ...
 └── SM 47                                     ← MTT S4000 大致这个量级
 
 全局共享:
 ├── L2 Cache (~6 MB)                          
 └── Global Memory (HBM/GDDR, ~48 GB)          
```

软件视角的对应:

| 软件概念 | 硬件对应 |
|---|---|
| `kernel<<<grid, block>>>` | grid 上所有 block 会被分配到各 SM |
| 一个 `block` | 整个 block 调度到**同一个 SM**(不会跨 SM)|
| 一个 `warp`(128 thread) | warp scheduler 调度的最小单位 |
| 一个 `thread` | 占用若干 register,在 ALU 上跑一条 SIMT 指令 |

block 是调度边界，warp 是执行边界。

> 这里用 CUDA 常见的 SM 心智模型做快速入门。MUSA 的 `MPC → MPX → MP` 物理层级、CUDA 近似对照和完整执行路径见 [`gpu-hierarchy.md`](gpu-hierarchy.md)。

---

## 3. 线程层级与全局索引

四个内置变量,在 kernel 内可以直接用:

| 变量 | 含义 | 启动时谁决定 |
|---|---|---|
| `threadIdx` | 线程在 block 内的 (x,y,z) 编号 | 硬件自动分配 |
| `blockIdx`  | block 在 grid 内的 (x,y,z) 编号 | 硬件自动分配 |
| `blockDim`  | 一个 block 的 (x,y,z) 维度 | `<<<grid, block>>>` 第二参数 |
| `gridDim`   | 一个 grid 的 (x,y,z) 维度 | `<<<grid, block>>>` 第一参数 |

### 一维全局索引

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) { ... }    // 边界保护,N 不是 blockDim 整数倍时必备
```

### 二维全局索引

```cpp
int gx = blockIdx.x * blockDim.x + threadIdx.x;
int gy = blockIdx.y * blockDim.y + threadIdx.y;
if (gx < W && gy < H) {
    matrix[gy * W + gx] = ...;   // 行优先存储
}
```

### grid size 的计算(向上取整)

```cpp
int threads = 256;
int blocks  = (N + threads - 1) / threads;
kernel<<<blocks, threads>>>(...);
```

→ 配套代码:[`code/week1/02_thread_index.mu`](../code/week1/02_thread_index.mu)

### 为什么要分 grid + block 两层?

block 内的线程具备三个关键条件：

- 共享 **shared memory**(快得多,~30 周期 vs global 的几百周期);
- 可以 `__syncthreads()` 互相等;
- 一定调度到**同一个 SM**。

block 之间没有这些保证：它们不能直接同步，也可能被调度到不同 SM 上。

> **block size 决定协作粒度,grid size 决定总并行度。**

---

## 4. 内存层级

```
速度    │  容量    │  作用域
──────────┼──────────┼──────────────────────
Register │  几十 KB │  单个 thread        ← 最快,~1 周期
Shared   │  ~48 KB  │  block 内共享       ← ~30 周期,要避 bank conflict
L1 Cache │  与 shared 共用                  ↓
L2 Cache │  ~6 MB   │  全 GPU 共享        ← ~200 周期
Constant │  64 KB   │  只读,有专门 cache  ↓
Global   │  ~48 GB  │  全 GPU 可读写      ← ~400-800 周期,要 coalesced
Local    │  在 global 上的"线程私有"        ↓
Host     │  CPU RAM │  通过 musaMemcpy 搬   ← PCIe 慢得多
```

常见误区：

- "**Local memory**" 不是真的快内存,它就是给寄存器装不下的局部变量在 global 上开辟的空间,慢。
- "**Constant memory**" 中，同一地址读取可利用广播和 constant cache；读取不同地址会失去广播优势，性能取决于实际访问模式和设备，不能一概等同于 global memory。
- 优化访存时主要检查三件事：
  1. **Coalesced**:同 warp 相邻线程访问相邻地址(week4)
  2. **Shared 替代 Global**:把多次复用的数据搬进 shared(week5 GEMM)
  3. **避免 Bank Conflict**:shared memory 分 32 个 bank,同 warp 内同 bank 不同地址会串行化

### 全局、共享与常量内存：先判断数据该放哪里

先从数据的读写者和生命周期出发，再考虑优化。下面的对比描述的是作用域与访问方式；具体设备和 SDK 的实现细节会有差异，不能据此推导固定容量、延迟或性能结论。

| 内存 | 谁能访问 | 读写行为 | 适合的数据 | 首先检查什么 |
| --- | --- | --- | --- | --- |
| Global（全局内存） | kernel 中的线程 | 可读、可写 | 输入、输出、跨 block 的中间结果 | 相邻线程是否访问相邻的 global 地址 |
| Shared（共享内存） | 同一 block 的线程 | 可读、可写 | block 内反复使用的 tile、部分和 | 屏障是否安全、是否有 bank conflict |
| Constant（常量内存） | kernel 中的线程 | 设备端只读，由主机更新 | 小型只读数据，且同一批线程经常读取同一地址 | 是否呈现广播式（同地址）访问 |

选择顺序可以固定为四步：

1. 跨 block 共享或一般可写的中间结果 → global；需要跨 kernel 存活的数据通常也放在 global，除非它是适合 constant 的小型只读常量符号。
2. 小型、只读、适合广播的参数 → constant。
3. 同一 block 内会被反复消费的数据 → shared。
4. 其他情况先从 global 开始，再测量。

下面三个 MUSA/CUDA 风格片段只说明作用域和访问模式；它们不构成可移植的性能结论，也不替代 Week 4、Week 5 中可运行的示例。

```cpp
__global__ void global_add(float* out, const float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
```

`global_add` 通过全局索引读写 global 指针指向的元素，并用 `if (i < n)` 保护尾部线程。

```cpp
__global__ void block_sum(const float* x, float* partial, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) return;
  int i = blockIdx.x * 256 + t;
  s[t] = (i < n) ? x[i] : 0.0f;
  __syncthreads();
  if (t == 0) {
    float sum = 0.0f;
    for (int j = 0; j < 256; ++j) sum += s[j];
    partial[blockIdx.x] = sum;
  }
}
```

`block_sum` 要求以一维、恰好 256 线程的 block 启动；若 `blockDim.x != 256`、`blockDim.y != 1` 或 `blockDim.z != 1`，所有线程会在访问 `s` 前一致返回。正常路径中，每个线程先写入 shared 数组，越界元素填零，所有线程无条件到达 `__syncthreads()`，再由 thread 0 对恰好 256 个元素求和并写出每个 block 的部分和。

```cpp
__constant__ float c_scale;

__global__ void scale(float* out, const float* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * c_scale;
}
```

`scale` 从设备端只读的 `c_scale` 读取缩放参数；主机负责在启动 kernel 前更新该常量。

读代码时，可以逐项检查：

1. 指针实际驻留在哪一种内存中？
2. 谁会读取、谁会写入这份数据？
3. 每个 block 的每个线程是否都能到达屏障？
4. global 访问是否呈现相邻地址模式？
5. 尾部线程是否受到边界保护？

继续学习时，可先阅读 [Week 3：同步前提](../code/week3/learning-notes.md)，再结合 [Week 4：global memory 优化](../code/week4/learning-notes.md) 和 [Week 5：shared/constant 与复用](../code/week5/learning-notes.md) 的可运行示例理解这些选择。

---

## 5. 执行模型与同步

### kernel 启动是异步的

```cpp
kernel<<<g, b>>>(...);      // 立即返回,不等 GPU 算完
// CPU 此时可以做别的事
musaDeviceSynchronize();    // 强制等
```

→ 配套代码:[`code/week1/06_async_kernel.mu`](../code/week1/06_async_kernel.mu) 直观看到 launch 时间 ≪ kernel 时间。

### 同步点:把 GPU 的进度拉回 CPU

| 同步方式 | 等什么 | 用法 |
|---|---|---|
| `musaDeviceSynchronize()` | 当前设备所有流上所有任务 | 调试期"暴力同步" |
| `musaStreamSynchronize(s)` | 单个 stream | 多流并发时精确控制 |
| `musaEventSynchronize(e)` | 单个 event 触发 | 计时 + 跨流依赖 |
| `musaMemcpy(D2H)` | **隐式同步**当前流 | 拷数据顺便等 kernel 跑完 |

没有经过同步点时，host 不能假设 kernel 已经完成。

### Stream:多任务并发

默认 stream(stream 0)上的任务**串行执行**。要让 H2D / Kernel / D2H 真正重叠,需要多个 stream:

```cpp
musaStream_t s1, s2;
musaStreamCreate(&s1);
musaStreamCreate(&s2);

musaMemcpyAsync(d_in1, h_in1, ..., s1);    // 走 s1
kernel<<<g, b, 0, s2>>>(...);              // 走 s2,可与 s1 并行
```

Week 2 会继续练习多 stream 并发。

---

## 6. 错误处理:同步 vs 异步

错误分两种,**必须分别抓**:

| 类型 | 例子 | 抓的方式 |
|---|---|---|
| 同步错误 | block 超过 1024 / 参数非法 / 显存不够 | 函数返回值 / `musaGetLastError()` |
| 异步错误 | kernel 内越界 / 写空指针 | 下一个**同步点**才会暴露 |

→ 标准范式:

```cpp
kernel<<<g, b>>>(...);
MUSA_CHECK(musaGetLastError());        // 同步错误
MUSA_CHECK(musaDeviceSynchronize());   // 异步错误
```

- 只 sync 不 GetLastError → launch 失败 kernel 根本没跑,sync 啥也不报
- 只 GetLastError 不 sync → kernel 越界 → 报错点会出现在下次 musaMemcpy

→ 配套代码:[`code/week1/05_error_check.mu`](../code/week1/05_error_check.mu) 主动触发 4 种错误。

### 常见错误码

| 码 | 名 | 含义 |
|---|---|---|
| 0 | `musaSuccess` | 一切正常 |
| 1 | `musaErrorInvalidValue` | 参数非法 |
| 2 | `musaErrorMemoryAllocation` | 显存不够 |
| 9 | `musaErrorInvalidConfiguration` | block/grid 超限 |
| 700 | `musaErrorIllegalAddress` | kernel 越界(异步)|

完整列表见官方编程指南附录。

---

## 7. Host vs Device:两片独立内存

> 这一节单独说明 host 和 device 指针的使用边界。

```cpp
float *h_ptr = (float*)malloc(N * sizeof(float));   // host RAM
float *d_ptr;
musaMalloc(&d_ptr, N * sizeof(float));               // device VRAM

d_ptr[0] = 1.0f;        // ✗ Segfault(host 解引用 device 地址)
kernel<<<g, b>>>(h_ptr); // ✗ illegal address(kernel 用 host 地址)
```

规则:**host 指针只在 host 用,device 指针只在 device 用,搬数据必须 musaMemcpy。**

命名建议:`h_` 前缀给 host 指针,`d_` 前缀给 device 指针,从一开始就养习惯。

→ 配套代码:[`code/week1/04_memory_basics.mu`](../code/week1/04_memory_basics.mu)

---

## 延伸阅读

- [`gpu-hierarchy.md`](gpu-hierarchy.md)：MUSA 的 MPC / MPX / MP 硬件层级与 CUDA 近似对照
- [`cuda-vs-musa.md`](cuda-vs-musa.md)：CUDA / MUSA 命名对照与差异
- [`glossary.md`](glossary.md)：术语小词典
- [官方编程指南 Ch1 到 Ch4](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/)：官方参考文档
- [《Programming Massively Parallel Processors》](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)：GPU 编程参考书
