# GPU Architecture and Programming Practice

> 外部实践案例集，来源目录原名 `GPU-Architecture-and-Programming-Practice-main`。
> 这里先作为 `code/` 下的补充案例库保留，不默认接入顶层 CMake 全量构建。

这些例子覆盖 MUSA Runtime、数学库、随机数、稀疏计算、多卡通信、设备属性、虚拟内存、图像处理和 GEMM/GEMV。它们更像“按主题查阅的案例集”，不是本仓库 week 主线的线性课程。

## 和现有 week 路线的关系

| 目录 | 主题 | 建议搭配 |
|---|---|---|
| `Chapter3/` | Stream 与矩阵乘法入门 | `code/week2/` Stream、`code/week5/` GEMM |
| `Chapter4/` | muBLAS / muRAND / muSPARSE / muSOLVER / muDNN / MCCL | `code/week5/` 库调用，`code/week6/` 多卡 |
| `Chapter6/` | 设备属性、虚拟内存、图像旋转、双缓冲 Stream | `code/week1/03_device_info.mu`、`code/week2/`、`code/week4/` |
| `Chapter8/` | GEMV / GEMM 性能案例 | `code/week5/05_naive_gemm.mu`、`code/week5/06_tiled_gemm.mu` |
| `Chapter9/` | 数学函数示例 | `docs/glossary.md`、`docs/cuda-vs-musa.md` |

## 知识点索引

### Chapter3: Runtime 基础和并发

- `3_1_musaStream.cpp`: Stream 创建、异步执行、同步边界。读这个例子时重点看两点：kernel launch 本身通常异步返回；真正的耗时统计要放在 Stream 同步或 Event 之后。
- `3_2_matrixMultiply.cpp`: 基础矩阵乘。它适合拿来和 `week5` 的 naive GEMM / tiled GEMM 对照：同样是 `C = A * B`，性能差异主要来自访存复用、线程块划分和 shared memory 使用。

### Chapter4: MUSA 生态库

- `4_1_callmuBLAS.cpp`: muBLAS 调用入口。关注 handle 生命周期、矩阵布局、leading dimension 和 alpha/beta 参数。
- `4_2_hostSideRNG.c` / `4_3_deviceSideRNG.c`: 随机数生成。host side 适合批量生成再拷贝，device side 适合每个线程独立采样；关键是 seed、sequence、offset 的可复现性。
- `4_4_iLUDecomposition.c`: 稀疏预条件分解。它属于线性求解器前处理，读代码时先确认矩阵格式和稀疏结构，再看调用链。
- `4_5_sparseDenseMatvec.c`: 稀疏矩阵乘稠密向量，典型 SpMV。重点是 CSR/COO 等格式如何影响访存连续性。
- `4_6_sparseSparseMatmul.c`: 稀疏矩阵乘稀疏矩阵。相比 SpMV，输出稀疏结构本身也需要计算或预估。
- `4_7_sparseTriMatDenseVecSolver.c`: 稀疏三角求解。常见于 LU/ILU 分解后的前代/回代。
- `4_8_sparseVecDot.c`: 稀疏向量点积。适合观察索引匹配和规约的库封装。
- `4_9_callmuSOLVER.cpp` / `4_10_QRDecompLSSolver.cpp`: muSOLVER 入口和 QR 最小二乘。关注 workspace 查询、info 返回值和数值稳定性。
- `4_11_callmuDNN.cpp`: muDNN 调用入口。关注 descriptor、tensor layout、workspace 和算法选择。
- `4_12_singleThreadedMcclMultiGpuComp.cpp`: 单线程控制多 GPU 的 MCCL 示例。重点是每个 device 的上下文切换和 communicator 管理。
- `4-13-oneDevicePerProcThreadComp.cpp`: 每个进程线程绑定一个 GPU。适合和单线程多卡写法比较：控制流更清晰，但线程同步更复杂。
- `4_14_multipleDevicesPerProcThreadComp.cpp`: 每个进程线程管理多个 GPU。适合观察更复杂的 device/thread 归属关系。

### Chapter6: 设备、内存和 Pipeline

- `6_1_musaDevicePropDefinition.cpp`: 设备属性结构体字段。建议和 `code/week1/03_device_info.mu` 对照，理解哪些字段影响 kernel 配置。
- `6_2_getDeviceBasicProperties.cpp.cpp`: 查询设备基础属性。重点看 SM 数、最大线程数、shared memory 上限、内存带宽相关字段。
- `6_3_musaVirtualMemoryManagement.cpp.cpp`: 虚拟内存管理。它比 `musaMalloc` 更底层，适合理解地址空间、物理内存映射和大块内存管理。
- `6_4_imageRotation.cpp`: 图像旋转。关注二维线程索引、边界判断和全局内存访问模式。
- `6_5_doubleBufferedStreamOptimization.cpp`: 双缓冲 Stream 优化。核心思想是让 H2D 拷贝、kernel、D2H 拷贝尽量流水化重叠。

### Chapter8: GEMV / GEMM

- `8_1_GEMV.cpp`: 矩阵向量乘。GEMV 算术强度通常低于 GEMM，更容易受内存带宽限制。
- `8_2_GEMM.cpp`: 矩阵矩阵乘。适合和 `week5` 的 naive/tiled/muBLAS 三种层次对照：手写 kernel 用来学原理，库调用用来建立性能基线。

### Chapter9: 数学函数

- `9_1_powDemo.c`: `pow` 类数学函数示例。读这类例子时要关注精度类型、函数开销，以及是否存在更便宜的等价写法，比如平方用乘法替代通用幂函数。

## 阅读顺序

1. 先看 `Chapter3/3_1_musaStream.cpp`，确认异步执行和同步边界。
2. 再看 `Chapter3/3_2_matrixMultiply.cpp` 和 `Chapter8/8_2_GEMM.cpp`，把矩阵乘作为性能主线。
3. 需要库能力时读 `Chapter4/`，按 muBLAS、muSPARSE、muSOLVER、muDNN、MCCL 分块。
4. 需要设备和内存细节时读 `Chapter6/`。

## 构建说明

这些文件暂未纳入 `code/CMakeLists.txt` 的默认构建，原因是依赖并不一致：

- Runtime 示例只需要 MUSA Runtime。
- muBLAS / muDNN / muSOLVER / MCCL 示例需要对应 SDK 组件和链接库。
- 多卡示例依赖实际设备数量和运行环境。

后续如果要把某个示例转成主线教材，建议复制到对应 `weekN/`，改成 `.mu` 三段式注释，并单独加入该周 `CMakeLists.txt`。
