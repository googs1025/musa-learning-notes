# CUDA → MUSA 对照与迁移

> MUSA SDK 的 Runtime API 大多可以与 CUDA 一一对照。
> 本文整理命名映射、主要差异和迁移步骤。

---

## 迁移范围

API 前缀和工具链名称通常可以直接替换。warp 宽度、同步原语、专有库名（muBLAS / muDNN）和调优参数需要按目标设备与 SDK 单独检查。

---

## 工具链对照

| CUDA | MUSA | 说明 |
|---|---|---|
| `nvcc` | `mcc` | 编译器 |
| `nvidia-smi` | `mthreads-gmi` | 设备状态查询 |
| `cuda-gdb` | MUSA SDK 调试器(以安装包实际命令为准) | kernel 单步调试 |
| `cuobjdump` | MUSA SDK 反汇编工具(以安装包实际命令为准) | 反汇编(看 SASS / PTX 对应) |
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

## 主要差异

### 1. Warp size 必须在目标设备查询 ⚠️

CUDA 常见设备的 warp 是 32。MUSA 不能用一个厂商级常数概括：旧版官方 S3000 示例报告 128，本仓库的 S4000/MUSA SDK 3.1.0 实测为 32，官方 S5000 示例也报告 32。迁移时应读取 `musaDeviceProp.warpSize`，device 代码使用内置 `warpSize`。

影响:

- **Warp shuffle / shfl 指令**：mask、lane 范围和可选 width 都必须与当前 SDK 的 API 定义和实际 `warpSize` 一致。
- **Reduce / Scan 算法**：循环边界、warp 数量和 shared partial 数量从 `warpSize` 推导，不能把 32 或 128 散落在代码中。
- **Occupancy 计算**：使用设备属性里的 warp 宽度、最大驻留线程数、寄存器和 shared memory 限制。
- **Block size 推荐值**：先选择 `warpSize` 的整数倍，再用实际测量决定 128、256 或其他配置。

### 2. Compute Capability vs MTT 架构号

CUDA 用 `sm_70` / `sm_80` / `sm_86` 这种字符串指定架构;MUSA 用自己的架构号(具体见官方 mcc --help 输出和发布说明)。**直接搬 CUDA 的 `-arch=sm_xx` 参数是不行的**,要查 mcc 当前版本支持的架构选项。

### 3. PTX → 中间表示

CUDA 的中间表示叫 **PTX**(可读汇编);MUSA 也有自己的中间 IR,具体名字以 SDK 版本为准。CUDA 和 MUSA 工具链反汇编输出格式不完全一样。

### 4. Device 数量与拓扑 API

`cudaDeviceCanAccessPeer` / NVLink 相关 API 在 MUSA 对应的是 MTLink。**多卡 P2P / 拓扑发现 API 名相似但参数语义可能不同**,涉及多卡时务必查 MUSA 官方手册。

### 5. Driver API 前缀

CUDA Driver API 用 `cu` 前缀(`cuLaunchKernel`、`cuModuleLoad` 等);MUSA Driver API 用 **`mu`** 前缀(`muLaunchKernel`、`muModuleLoad`)。**注意不是 `musa` 前缀**,Driver API 单独是 `mu`。

---

## 实战迁移步骤

把 CUDA 项目迁移到 MUSA 时，可以按下面的顺序处理：

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

1. warp size 写死为 32 或 128 的地方（查所有常数和固定 mask）
2. PTX 内联汇编(MUSA IR 不一样,要重写)
3. 用了 CUDA 独有库(cuTENSOR / cuGraphics / OptiX,MUSA 暂无对等)
4. 调用了 `sm_xx` 这种 NVIDIA 架构字符串

---

## 不要自动化的地方

- `-arch=sm_xx` 必须换成 MUSA 工具链实际支持的架构号。
- warp 宽度变化会改变算法语义，需要从目标设备查询并重新检查算法，不能机械替换数字。
- 错误日志和 CMake 变量名中的 `cuda` 可能不需要替换，应先确认它的用途。

---

## 参考

- [`concepts.md`](concepts.md)：基础概念（SIMT / 硬件 / 内存）
- [`glossary.md`](glossary.md)：术语小词典
- [官方编程指南 Ch11 附录](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/)：错误码完整表
- [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：CUDA 编程参考
