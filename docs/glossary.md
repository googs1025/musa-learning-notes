# GPU / MUSA 术语小词典

> 按字母序,看代码或文档遇到不认识的词来这里查。
> 想看完整心智模型,看 [`concepts.md`](concepts.md);想查 API,看 [`cuda-vs-musa.md`](cuda-vs-musa.md)。

---

## A

**AoS (Array of Structures)** — 数据布局形式之一,每个元素是个结构体,结构体连续存放:`struct{x,y,z}[N]`。GPU 上常因访存不合并而慢,对比 SoA。

**Asynchronous (异步)** — kernel launch 和 `*Async` API 调用立刻返回,GPU 在后台执行。需要同步点(`musaDeviceSynchronize` / `musaMemcpy(D2H)`)才能确认完成。

**Atomic (原子操作)** — `atomicAdd / atomicCAS` 等。多个线程并发更新同一地址时保证不丢更新。代价是序列化,慎用在热路径上。

## B

**Bank Conflict** — Shared memory 分 32 个 bank。同 warp 内多个线程访问**同一 bank 不同地址**时,硬件会串行化 → 性能下降。同地址(广播)和不同 bank 都没事。

**Block (线程块)** — 启动配置 `<<<grid, block>>>` 的第二个参数。block 内线程共享 shared memory + 可 `__syncthreads()` 同步。一个 block 整体调度到一个 SM。

**blockDim** — 内置变量,kernel 内可直接用,表示当前 block 的 (x,y,z) 维度。

**blockIdx** — 内置变量,当前 block 在 grid 内的 (x,y,z) 编号。

## C

**Coalesced Access (合并访存)** — 同 warp 内连续线程访问连续地址时,硬件可以合成一次 wide load。这是 GPU 内存优化的第一原则。

**Constant Memory** — 64 KB 的只读内存,有专用 cache。所有线程读同一地址时极快;读不同地址退化成 global。`__constant__` 修饰。

**Compute Capability** — NVIDIA GPU 的架构版本号(如 sm_80)。MUSA 用自己的架构号体系。

## D

**Device** — GPU 端。`__device__` 修饰的函数在 device 执行;device 指针(`musaMalloc` 返回)只能在 device 用。

**dim3** — 启动配置用的三维向量类型。`dim3(16, 16)` 等价于 `(16, 16, 1)`。

**Driver API** — 比 Runtime API 更底层的一套 API,前缀是 `mu`(不是 `musa`)。需要手动管理 context / module / kernel,适合 JIT 场景。日常用 Runtime 即可。

**Dynamic Parallelism (动态并行)** — kernel 内部启动子 kernel。

## E

**Event** — `musaEvent_t`,标记 stream 中的某个时间点。用途:GPU 计时(配合 `musaEventElapsedTime`)、跨 stream 同步。

**Execution Configuration** — `<<<grid, block, smem, stream>>>` 这一坨,控制 kernel 怎么启动。

## G

**Global Memory** — 显存。所有线程可读写,容量大(MTT S4000 = 48 GB),但慢(几百周期)。优化目标就是让 global 访问变 coalesced。

**GPU printf** — `printf` 在 kernel 内可用,但写到环形缓冲区,要 `musaDeviceSynchronize` 才 flush。

**Grid (网格)** — kernel 启动时的最外层维度,由若干 block 组成。

**gridDim** — 内置变量,当前 grid 的 (x,y,z) 维度。

## H

**HBM (High Bandwidth Memory)** — 一种高带宽显存(堆叠 DRAM),数据中心 GPU 常用。MTT S4000 用的就是 HBM。

**Host** — CPU 端。`__host__` 修饰的函数在 host 执行。host 指针(`malloc / new`)只能在 host 用。

## K

**Kernel** — `__global__` 修饰的函数,host 调用,device 执行,**返回值必须 void**。

## L

**L1 / L2 Cache** — 缓存层级。L1 通常和 shared memory 共享存储,L2 全 GPU 共享(~6 MB 量级)。

**Latency Hiding (延迟隐藏)** — GPU 不靠"快",靠"切到别的 warp 来藏延迟"。所以 occupancy(每 SM 驻留多少 warp)是性能关键。

**Launch Overhead** — kernel 启动本身有固定开销(微秒级)。极小 kernel 反复启动会被这个吃掉,解药是 MUSA Graph。

**Local Memory** — 名字骗人 —— 它**不是快内存**,只是 register 装不下时给单线程在 global 上开的私有空间,慢。

## M

**Managed Memory / Unified Memory** — `musaMallocManaged` 分配的内存,host / device 都能用同一指针,运行时自动迁移。简单但有性能代价。

**mcc** — MUSA 编译器,对应 NVIDIA 的 `nvcc`。

**MCCL** — MUSA Collective Communications Library,对应 NCCL。多卡 AllReduce / Broadcast 等。

**Memcpy** — `musaMemcpy(dst, src, bytes, kind)`。kind 有 4 种:H2D / D2H / D2D / H2H。**默认同步**,会等数据搬完才返回。

**MTLink** — 摩尔线程的多卡互连技术,对应 NVLink。

**MTT** — Multi-Thread-Tile,**MUSA 的 warp 单位 = 128 线程**。这是和 CUDA 最显眼的差异。

**muBLAS / muDNN / muFFT** — MUSA 加速库,对应 cuBLAS / cuDNN / cuFFT。

**musaError_t** — 错误类型,几乎所有 MUSA API 返回它。`musaSuccess = 0`。

## N

**Nested Kernel** — 见 Dynamic Parallelism。

## O

**Occupancy (占用率)** — 每 SM 实际驻留 warp 数 / 理论最大 warp 数。高 occupancy 利于隐藏延迟。但**不是越高越好**,寄存器/shared 用得多了,虽然 occupancy 低,单线程可能更快。

## P

**Pinned Memory / Page-Locked** — `musaHostAlloc` 分配的 host 内存,不会被 OS 换页。是 `musaMemcpyAsync` 真正异步的前提。

**Pitch** — `musaMallocPitch` 给 2D 数组分配时,实际行宽会向上对齐到某个值(pitch ≥ width)。访问时用 pitch 步长才对。

**P2P (Peer-to-Peer)** — 两张 GPU 之间直接互访显存,不经过 host。需要 `musaDeviceEnablePeerAccess`。

## R

**Register** — 最快的存储,每线程私有,几十 KB 量级。kernel 用太多 register 会减少 occupancy。

**Runtime API** — 高层 API,`musa*` 前缀,99% 应用用这套。本仓库的所有示例都是 Runtime API。

## S

**SAXPY** — Single-precision A*X + Y,经典的访存型基准。带宽测试常用。

**Shared Memory** — block 内共享的快速内存(~30 周期),`__shared__` 修饰。优化大头(week5 GEMM 的核心)。

**Shuffle (shfl)** — warp 内线程之间直接交换寄存器值的指令(`__shfl_sync` 等),不经过 shared memory。MUSA 上 mask 是 128 bit。

**SIMT (Single Instruction, Multiple Thread)** — GPU 执行模型。同 warp 内的线程锁步执行同一条指令。

**SM (Streaming Multiprocessor)** — GPU 的物理调度单位,内含 warp scheduler / 寄存器堆 / shared memory / 各类执行单元。

**SoA (Structure of Arrays)** — 与 AoS 相反,每个字段单独一个数组。GPU 上访存通常更友好。

**Stream** — `musaStream_t`,任务执行的"通道"。同一 stream 内 FIFO 串行,不同 stream 之间可并行。默认 stream 0 会和所有 stream 同步。

## T

**threadIdx** — 内置变量,当前线程在 block 内的 (x,y,z) 编号。

**Tile / Tiling** — 把大问题切成 block 能装下的"小块",在 shared memory 里做。Tiled GEMM 是经典案例。

**torch_musa** — PyTorch 在 MUSA 上的后端,对应 `torch.cuda` 那一套。

## U

**Unified Memory** — 见 Managed Memory。

## W

**Warp** — GPU 调度的最小单位。**MUSA = 128 线程,CUDA = 32 线程**。同 warp 内线程指令同步执行。

**Warp Divergence** — 同 warp 内线程走了不同的 if/else 分支,硬件串行执行各分支 → 性能下降。

**Warp Size** — 一个 warp 的线程数。`warpSize` 是内置变量,运行时用它而不是写死。

## Z

**Zero-Copy** — `musaHostAlloc(..., musaHostAllocMapped)` + `musaHostGetDevicePointer`,GPU 直接访问 host 内存,不经过显式拷贝。适合数据只读一次的场景,绕开 PCIe 显存往返。

---

## 还想看?

- 入门概念心智模型 → [`concepts.md`](concepts.md)
- API 名怎么从 CUDA 翻译过来 → [`cuda-vs-musa.md`](cuda-vs-musa.md)
- 安装环境 → [`setup.md`](setup.md)