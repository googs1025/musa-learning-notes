# MUSA 学习笔记 (一) · 我是怎么把第一段代码跑起来的

> 这是我学摩尔线程 MUSA SDK 的学习记录,不是教程。
> 写下来一是为了强迫自己理清思路,二是攒一份"未来的我会感谢现在的我"的笔记。
> 配套代码:`code/week1/`,官方指南:<https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/>

---

## 为什么开始学 MUSA

平时主要写 Go 和云原生方向的东西,异构这块一直是知识盲区。最近几个原因凑到一起决定开个坑:

- 国产算力的故事在身边越来越多,真去落地的时候 CUDA 那套不一定能用,得有备选;
- 摩尔线程的 MUSA SDK 跟 CUDA 几乎一对一映射,学一份等于顺带把 CUDA 也补了;
- 给自己定个节奏:每周读一章官方指南 + 跑通对应代码 + 写一篇笔记,逼着输出。

第一周的目标很简单:**把 Hello World 跑起来,搞清楚 MUSA 的软件栈分层。** 对应官方指南 Ch1–4。

---

## 我对 MUSA 软件栈的理解

官方文档第 3 章把软件架构讲得挺清楚,但我自己更习惯画成这样一张图:

```
┌─────────────────────────────────────────┐
│     应用 (PyTorch / 自写 kernel)         │
├─────────────────────────────────────────┤
│   加速库  muBLAS / muDNN / MCCL ...      │
├─────────────────────────────────────────┤
│   Runtime API  (musaMalloc / Memcpy)     │  ← 我现在在这层
├─────────────────────────────────────────┤
│   Driver API   (muLaunchKernel ...)      │  ← 框架/JIT 用的更底层 API
├─────────────────────────────────────────┤
│   MUSA Toolkit (mcc 编译器 + 头文件)     │
├─────────────────────────────────────────┤
│   驱动 + 内核模块 (mt-driver)            │
└─────────────────────────────────────────┘
```

我学到的几个关键点:

- **Runtime API 是 99% 应用要打交道的层**,Driver API 主要是 JIT、动态加载 kernel 模块的场景才用。第一周完全不用碰 Driver API。
- 跟 CUDA 几乎是一对一映射:`cuda*` 改 `musa*`,`nvcc` 改 `mcc`,`nvidia-smi` 改 `mthreads-gmi`,基本就完成大半翻译。
- **有一个坑容易踩**:CUDA 的 warp 是 32 线程,MUSA 是 **128 线程**(摩尔线程内部叫 MTT, Multi-Thread-Tile)。涉及 warp-level 操作时坐标系会变,我标记下来,等到 Week 4 性能优化再回头看。

---

## 环境搭建

我没有本地卡,直接用了 AutoDL:

1. <https://www.autodl.com/> 注册;
2. 算力市场筛 GPU = MTT S4000;
3. 镜像选预装 MUSA Toolkit + torch_musa 的版本;
4. SSH 进去,`mcc --version` + `mthreads-gmi` 验证。

整个流程没踩坑,镜像里啥都有。后续如果要本地装驱动,参考[官方安装指导](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/install_guide/),坑应该不少,先不折腾。

---

## 6 个最小示例,我的笔记

我给自己定的规矩:每个示例不仅要能编过、跑过,还要自己能用一两句话讲清楚"为什么是这样写"。下面是这 6 个示例我学到的关键点。完整代码在 `code/week1/`。

### Hello World

```cpp
#include <musa_runtime.h>
#include <cstdio>

__global__ void hello_from_gpu() {
    int tid = threadIdx.x;
    printf("GPU: tid=%d Hello world!\n", tid);
}

int main() {
    printf("CPU: Hello world!\n");
    hello_from_gpu<<<1, 5>>>();
    musaDeviceSynchronize();
    return 0;
}
```

第一次写就被三件事拌了一下:

- `__global__` 这个修饰符的语义:CPU 调用,GPU 执行,**返回值必须 void**。我一开始想返回个 int 看看效果,直接编不过,这才意识到 kernel 是异步启动的,语义上根本拿不到返回值。
- `<<<1, 5>>>` 这个语法第一次见挺反直觉,但其实就是"启动 1 个 block × 5 个线程"的简写。
- 没写 `musaDeviceSynchronize()` 前我看不到 GPU 的输出 —— 后来才知道 GPU 上的 printf 是写到环形缓冲区的,程序不同步直接退出,缓冲区就被丢了。

跑出来 5 行 GPU 输出**顺序不固定**,这点我开始觉得是 bug,后来反应过来:GPU 是 SIMT 模型,不同 warp 调度顺序由硬件决定,不能依赖输出顺序判断逻辑。

### 线程索引:全局编号

这是最高频的一行公式,几乎所有 1D kernel 都从这开始:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

我一开始问的问题是:**为什么要分 grid + block 两层,不能直接一层吗?**

后来想通了,因为 block 内的线程是有"特权"的:

- 共享 shared memory(很快,~几十周期);
- 可以 `__syncthreads()` 互相等;
- 一定调度到同一个 SM 上。

而 block 之间是独立的,不能直接同步,可以被调到不同 SM 跑。所以 **block size 决定协作粒度,grid size 决定总并行度** —— 这个划分是物理硬件结构的反映,不是为了凑数。

### 设备查询:让"硬件"具体起来

`musaGetDeviceProperties` 能问到一堆字段,我跑出来记录了下:

```
SM count            : ?       (我那台 S4000 跑出来是 ?,你跑了告诉我)
Warp size           : 128     ← MUSA 这里和 CUDA 不一样
Max threads/block   : 1024
Shared mem/block    : 48 KB   (常见值)
```

这些数字平时记不住,但写优化代码时处处依赖:

- block size 应该是 warpSize 的倍数 → MUSA 上就是 128 / 256 / 512 / 1024;
- shared memory 用量受限 → 决定 tile 大小;
- 算理论显存带宽:`bw = 2 × memoryClockRate × memoryBusWidth / 8 / 1e6`(系数 2 是 DDR,每周期传两次)。

下一周做带宽测试的时候这个公式会用上。

### 显存四件套:踩了一个新手坑

写第一版的时候我想直接 `printf("%f\n", d[0])`,以为 `musaMalloc` 返回的指针就是个普通指针。**Segfault 直接送我回家。**

理解之后才知道:host 和 device 是两片完全独立的内存,各自有各自的地址空间。host 解引用 device 指针就是访问非法地址。

修正姿势:

| 动作 | API | 我容易忘的事 |
|---|---|---|
| 分配 | `musaMalloc` | 第一个参数是 `void**`(双指针) |
| 置值 | `musaMemset` | **按字节填**,不是按元素 |
| 拷贝 | `musaMemcpy` | 4 个方向,kind 参数别传错 |
| 释放 | `musaFree` | 长任务忘记一定 OOM |

`musaMemset` 这个坑我专门验证了一下:`musaMemset(d, 1, N*4)` 之后 `d[0]` 不是 1.0f,而是 `0x01010101` 解释成 float ≈ 2.36e-38。要按元素初始化只能自己写 kernel,这是为什么我在 `04_memory_basics.mu` 里写了个 `fill_const` kernel 演示。

我给自己定的命名规范:host 指针 `h_` 前缀,device 指针 `d_` 前缀。多写几次就再也不会混。

### 错误检查:同步 vs 异步两个时机

这块我刚开始觉得"不就是查个错吗",写多了才意识到这是个真问题。

错误分两类,**必须分别抓**:

```cpp
kernel<<<g, b>>>(...);
CHECK(musaGetLastError());        // ← 同步错误:launch 配置非法
CHECK(musaDeviceSynchronize());   // ← 异步错误:执行期崩溃
```

我自己测了下,只查任意一个会怎样:

- 只 sync 不 GetLastError:launch 配置非法时(比如 `<<<1, 99999>>>`),kernel 根本没跑,sync 啥也不报。
- 只 GetLastError 不 sync:kernel 内部越界写指针,launch 时是合法配置,GetLastError 显示 ok,**直到下一次 musaMemcpy 才挂**。报错点离真正的 bug 隔了十万八千里。

这就是为什么我现在写 MUSA 代码时养成了反射:每次 `<<<>>>` 后面接两行 CHECK。

CHECK 宏长这样,准备复制到所有 MUSA 项目:

```cpp
#define CHECK(call) do {                                       \
    musaError_t _e = (call);                                   \
    if (_e != musaSuccess) {                                   \
        fprintf(stderr, "MUSA error %d at %s:%d\n",            \
                (int)_e, __FILE__, __LINE__);                  \
        return 1;                                              \
    }                                                          \
} while (0)
```

### 异步 kernel:第一次"看见"它

知道 kernel 是异步的是一回事,**亲眼看见**又是一回事。我写了个故意慢的 kernel,分别测了两段时间:

```cpp
auto t0 = chrono::now();
busy_kernel<<<64, 128>>>(LOOPS, d_sink);
double t_launch = ms_since(t0);     // launch 这一行多久

auto t1 = chrono::now();
musaDeviceSynchronize();
double t_wait = ms_since(t1);       // GPU 实际跑多久
```

跑出来大概是这个比例:

```
launch ≈ 0.08 ms   ← 几乎为 0,GPU 才刚开始算
wait   ≈ 42 ms     ← 这才是 kernel 真正在 GPU 上跑的时间
```

这个比例让我直观地理解了几件事:

- **想测 kernel 时间必须先 sync**,否则你测的是 launch 开销;
- launch 本身有固定开销(微秒级),所以**反复 launch 极小 kernel** 时它会变成瓶颈 —— 这就是 MUSA Graph 要解决的问题(下周的内容);
- `musaMemcpy(D2H)` 内部会**隐式同步**,所以日常代码里很多时候不用显式 sync,但前提是后面紧跟 D2H 拷贝。

---

## CUDA → MUSA 速查(我自己用的版本)

写代码时反射不过来,贴这里供自己查:

```
cudaMalloc            →  musaMalloc
cudaMemcpy            →  musaMemcpy
cudaMemcpyHostToDevice→  musaMemcpyHostToDevice
cudaMemcpyDeviceToHost→  musaMemcpyDeviceToHost
cudaMemset            →  musaMemset
cudaFree              →  musaFree
cudaDeviceSynchronize →  musaDeviceSynchronize
cudaGetLastError      →  musaGetLastError
cudaGetDeviceCount    →  musaGetDeviceCount
cudaGetDeviceProperties→ musaGetDeviceProperties
cudaStreamCreate      →  musaStreamCreate
cudaEventCreate       →  musaEventCreate
cudaEventElapsedTime  →  musaEventElapsedTime
cudaMallocManaged     →  musaMallocManaged
nvcc                  →  mcc
nvidia-smi            →  mthreads-gmi
```

经验:基本"想到 cuda 就敲 musa",敲错了编译器会提醒你。

---

## 这周我没搞懂的几个问题

留个清单,后面学到的时候回来填:

1. **MTT (warp) = 128 在 occupancy 计算里是怎么影响 block size 选择的?** —— Week 3 看 Ch9 的时候应该会清楚。
2. **musaMallocManaged(统一内存)和普通 musaMalloc 的真实代价差多少?** —— 准备拿同一个 vectorAdd 跑两版对比。
3. **Driver API 到底什么场景才值得用?** —— 下周用 Driver API 重写一遍 vectorAdd,看代码量差异。
4. **MUSA 的 mcc 编译流程跟 nvcc 是不是一样分 host/device 两路?** —— 想 dump 中间产物看一下。

---

## 下周计划

跟着官方 Ch5 走,目标是把 vectorAdd 这条路走通:

- [ ] Runtime API 写完整 vectorAdd(7 步骨架);
- [ ] Driver API 重写一遍,做对比;
- [ ] Multi-Stream 三阶段流水线,看 H2D / Kernel / D2H 重叠效果;
- [ ] MUSA Graph,把今天异步 kernel 那个 launch overhead 实际打下去。

---

## 今天的产出

- 代码:`code/week1/01..06_*.mu`,6 个示例 + Makefile + 10 道习题
- 笔记:本文
- 仓库:[github.com/googs1025/musa-learning-notes](https://github.com/googs1025/musa-learning-notes)

下周见。
