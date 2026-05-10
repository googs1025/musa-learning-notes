# CUDA → MUSA 对照与迁移

> MUSA SDK 设计上几乎一对一映射 CUDA Runtime API。
> 这篇文档列**完整命名映射 + 真正不同的地方 + 实战迁移步骤**。

---

## 一句话总结

> **把 `cuda` 全局替换成 `musa`,把 `nvcc` 替换成 `mcc`,90% 的 CUDA 代码就能编**。
> 剩下 10% 集中在:`warp = 128` 引发的同步原语、专有库名(muBLAS / muDNN)、特定调优参数。

---

## 工具链对照

| CUDA | MUSA | 说明 |
|---|---|---|
| `nvcc` | `mcc` | 编译器 |
| `nvidia-smi` | `mthreads-gmi` | 设备状态查询 |
| `cuda-gdb` | `mcc-gdb`(部分版本) | kernel 单步调试 |
| `cuobjdump` | `mcc-objdump` | 反汇编(看 SASS / PTX 对应) |
| `nsys` / `nsight` | 官方 profiler(参考 Ch9 文档) | 性能分析 |

---

## API 命名映射(Runtime)

### 设备管理

| CUDA | MUSA |
|---|---|
| `cudaGetDeviceCount` | `musaGetDeviceCount` |
| `cudaGetDeviceProperties` | `musaGetDeviceProperties` |
| `cudaSetDevice` | `musaSetDevice` |
| `cudaGetDevice` | `musaGetDevice` |
| `cudaDeviceProp` | `musaDeviceProp` |
| `cudaDeviceSynchronize` | `musaDeviceSynchronize` |
| `cudaDeviceReset` | `musaDeviceReset` |

### 内存管理

| CUDA | MUSA |
|---|---|
| `cudaMalloc` | `musaMalloc` |
| `cudaFree` | `musaFree` |
| `cudaMemset` | `musaMemset` |
| `cudaMemcpy` | `musaMemcpy` |
| `cudaMemcpyAsync` | `musaMemcpyAsync` |
| `cudaMemcpyHostToDevice` | `musaMemcpyHostToDevice` |
| `cudaMemcpyDeviceToHost` | `musaMemcpyDeviceToHost` |
| `cudaMemcpyDeviceToDevice` | `musaMemcpyDeviceToDevice` |
| `cudaMallocPitch` | `musaMallocPitch` |
| `cudaMallocHost` / `cudaHostAlloc` | `musaMallocHost` / `musaHostAlloc` |
| `cudaFreeHost` | `musaFreeHost` |
| `cudaMallocManaged` | `musaMallocManaged` |
| `cudaMemPrefetchAsync` | `musaMemPrefetchAsync` |
| `cudaMemAdvise` | `musaMemAdvise` |
| `cudaMemGetInfo` | `musaMemGetInfo` |

### 错误处理

| CUDA | MUSA |
|---|---|
| `cudaError_t` | `musaError_t` |
| `cudaSuccess` | `musaSuccess` |
| `cudaGetLastError` | `musaGetLastError` |
| `cudaPeekAtLastError` | `musaPeekAtLastError` |
| `cudaGetErrorString` | `musaGetErrorString` |
| `cudaGetErrorName` | `musaGetErrorName` |

### Stream / Event

| CUDA | MUSA |
|---|---|
| `cudaStream_t` | `musaStream_t` |
| `cudaStreamCreate` | `musaStreamCreate` |
| `cudaStreamDestroy` | `musaStreamDestroy` |
| `cudaStreamSynchronize` | `musaStreamSynchronize` |
| `cudaStreamWaitEvent` | `musaStreamWaitEvent` |
| `cudaEvent_t` | `musaEvent_t` |
| `cudaEventCreate` | `musaEventCreate` |
| `cudaEventDestroy` | `musaEventDestroy` |
| `cudaEventRecord` | `musaEventRecord` |
| `cudaEventSynchronize` | `musaEventSynchronize` |
| `cudaEventElapsedTime` | `musaEventElapsedTime` |

### Graph(API 名直接替换)

| CUDA | MUSA |
|---|---|
| `cudaGraph_t` | `musaGraph_t` |
| `cudaGraphExec_t` | `musaGraphExec_t` |
| `cudaStreamBeginCapture` | `musaStreamBeginCapture` |
| `cudaStreamEndCapture` | `musaStreamEndCapture` |
| `cudaGraphInstantiate` | `musaGraphInstantiate` |
| `cudaGraphLaunch` | `musaGraphLaunch` |

### Kernel 修饰符与内置变量

| CUDA / MUSA(完全相同) | 含义 |
|---|---|
| `__global__` | kernel,host 调用,device 执行 |
| `__device__` | device 调用,device 执行 |
| `__host__` | host 调用,host 执行(可与 `__device__` 共用) |
| `__shared__` | block 内共享 |
| `__constant__` | constant memory |
| `threadIdx` / `blockIdx` / `blockDim` / `gridDim` | 内置维度变量 |
| `__syncthreads()` | block 内同步 |
| `__syncwarp()` | warp 内同步 |

---

## 加速库命名

| CUDA 库 | MUSA 库 | 用途 |
|---|---|---|
| cuBLAS | **muBLAS** | 线性代数(GEMM、AXPY...) |
| cuDNN | **muDNN** | DNN 算子(conv / pooling / norm...) |
| cuFFT | **muFFT** | FFT |
| cuRAND | **muRAND** | 随机数 |
| cuSPARSE | **muSparse** | 稀疏矩阵 |
| cuSOLVER | **muSolver** | 线性方程组 |
| NCCL | **MCCL** | 多卡通信(AllReduce / Broadcast...) |
| Thrust | (类似封装,具体名以官方为准) | 高级容器 / 算法 |

---

## 真正的差异(踩坑警告)

### 1. Warp size:128 vs 32 ⚠️

```
CUDA  warp = 32  线程
MUSA  warp = 128 线程  (摩尔线程内部叫 MTT, Multi-Thread-Tile)
```

影响:

- **Warp shuffle / shfl 指令**:`__shfl_sync(mask, ...)` 的 mask 含义会变。CUDA 上 `0xffffffff` 表示 32 线程全活,MUSA 上要用 `0xffffffffffffffffffffffffffffffff`(128 bit)。
- **Reduce / Scan 算法**:跨 warp 那一层的循环边界从 32 变成 128。
- **Occupancy 计算**:每 SM 能驻留多少 warp,公式里的 32 换成 128。
- **Block size 推荐值**:`128 / 256 / 512 / 1024` 仍然好,因为它们都是 128 的倍数;`64 / 96` 这种**在 MUSA 上不再是整 warp 倍数**,要避免。

### 2. Compute Capability vs MTT 架构号

CUDA 用 `sm_70` / `sm_80` / `sm_86` 这种字符串指定架构;MUSA 用自己的架构号(具体见官方 mcc --help 输出和发布说明)。**直接搬 CUDA 的 `-arch=sm_xx` 参数是不行的**,要查 mcc 当前版本支持的架构选项。

### 3. PTX → 中间表示

CUDA 的中间表示叫 **PTX**(可读汇编);MUSA 也有自己的中间 IR,具体名字以 SDK 版本为准。`cuobjdump` 和 `mcc-objdump` 看到的输出格式不完全一样。

### 4. Device 数量与拓扑 API

`cudaDeviceCanAccessPeer` / NVLink 相关 API 在 MUSA 对应的是 MTLink。**多卡 P2P / 拓扑发现 API 名相似但参数语义可能不同**,涉及多卡时务必查 MUSA 官方手册。

### 5. Driver API 前缀

CUDA Driver API 用 `cu` 前缀(`cuLaunchKernel`、`cuModuleLoad` 等);MUSA Driver API 用 **`mu`** 前缀(`muLaunchKernel`、`muModuleLoad`)。**注意不是 `musa` 前缀**,Driver API 单独是 `mu`。

---

## 实战迁移步骤

如果你手里有一份 CUDA 项目想跑在 MUSA 上,推荐顺序:

```bash
# 1. 大批量替换 API 前缀(用 sed 或 IDE 全局替换)
sed -i 's/\bcuda/musa/g' src/*.cu src/*.h

# 2. 重命名文件后缀(可选)
for f in src/*.cu; do mv "$f" "${f%.cu}.mu"; done

# 3. 替换工具链
sed -i 's/nvcc/mcc/g' Makefile CMakeLists.txt

# 4. 替换库名
sed -i 's/cublas/mublas/g; s/cudnn/mudnn/g' src/*.h

# 5. 编一遍,看错误信息修剩下的 5%
mcc -O2 src/main.mu -o main -lmusart
```

剩下的会编不过 / 运行报错的,大概率落在:

1. warp size 写死了 32 的地方(查所有 `32` / `0xffffffff`)
2. PTX 内联汇编(MUSA IR 不一样,要重写)
3. 用了 CUDA 独有库(cuTENSOR / cuGraphics / OptiX,MUSA 暂无对等)
4. 调用了 `sm_xx` 这种 NVIDIA 架构字符串

---

## 不要自动化的地方

- **`-arch=sm_xx` 不要无脑替换** —— 必须用 MUSA 真实存在的架构号
- **warp 边界 32 不要替换 128 替换** —— 算法层面要重新 review,不仅仅是数字
- **`cuda` 出现在字符串里** —— 比如错误日志、cmake 变量名,有些是无害的不要改

---

## 参考

- [`concepts.md`](concepts.md) — 基础概念(SIMT / 硬件 / 内存)
- [`glossary.md`](glossary.md) — 术语小词典
- [官方编程指南 Ch11 附录](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/) — 错误码完整表
- [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — 学完 MUSA 反过来看 CUDA 文档很顺