# Week 5 学习材料

Week 5 进入片上内存和 GEMM。主线是从 global memory 直接访问，逐步引入 shared memory、constant memory 和 tiled GEMM，再用 muBLAS 建立库函数性能基准。

## 阅读顺序

1. `01_shared_basics.mu`: 先学 static/dynamic shared memory 写法。
2. `02_reduce_shared.mu`: 用 shared memory 改写 reduce。
3. `03_transpose_shared.mu`: 用 shared tile 处理 transpose。
4. `04_stencil_constant.mu`: 用 constant memory 放小型只读参数。
5. `05_naive_gemm.mu`: 建立 GEMM 正确性和性能基线。
6. `06_tiled_gemm.mu`: 用 shared tile 降低 global memory 访问。
7. `07_mublas_sgemm.mu`: 和库实现对照。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_shared_basics.mu` | `__shared__`、动态 shared、block 内可见性 | 把 shared memory 当全局缓存 |
| `02_reduce_shared.mu` | shared reduce、`__syncthreads()` | 忘记同步导致读到旧值 |
| `03_transpose_shared.mu` | tile 读写、padding | shared 访问也可能 bank conflict |
| `04_stencil_constant.mu` | constant memory 适合小型广播只读数据 | 把任意大数组塞进 constant |
| `05_naive_gemm.mu` | 每线程算一个 C 元素 | 只追求正确不算 GFLOPS |
| `06_tiled_gemm.mu` | A/B tile 复用、同步边界 | tile 边界没有处理非整除尺寸 |
| `07_mublas_sgemm.mu` | 自写版和库版对比 | 期望入门 tiled GEMM 接近库性能 |

## 代码阅读抓手

GEMM 阅读重点：

- 每个 thread 负责哪个 `C[row, col]`。
- 每轮 tile 从 global memory 读了哪些 A/B 元素。
- `__syncthreads()` 放在 tile 加载后和下一轮覆盖 shared 前。
- 边界判断是否覆盖非 16/32 整除的矩阵尺寸。

## CUDA_Freshman 对照

- `24_shared_memory_read_data`: shared memory 基础。
- `25_reduce_integer_shared_memory`: shared reduce。
- `26_transform_shared_memory`: shared transpose。
- `27_stencil_1d_constant_read_only`: constant/read-only cache。

GEMM 进阶可参考 SGEMM_CUDA: <https://github.com/siboehm/SGEMM_CUDA>。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
