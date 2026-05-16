# LeetGPU Easy · MUSA 移植版

把 [LeetGPU](https://leetgpu.com/challenges) Easy 难度 19 题(去掉 41 PyTorch-only 1 题)的 CUDA starter 移植为 MUSA,每题给一份"能跑通过"的基线实现。**不追求性能极限**,只演示题目本身要点和 MUSA 与 CUDA 的差异。

## 用法

```bash
# 远端 AutoDL 实例上(MUSA SDK 已装):
cd code
cmake -B build -DMUSA_PATH=/usr/local/musa
cmake --build build -j

# 跑某一道
./build/leetgpu/easy/01_vector_add
./build/leetgpu/easy/02_matrix_multiplication
# ...
```

每个 `.mu` 都自带 `main()`,用题目 Example 1 做最小 smoke test。
**只要打印的 "expect:" 行能对上,就说明 kernel 正确**。

## 题目清单(18 题)

| # | 题目 | 知识点 | roadmap |
|---|------|--------|---------|
| 01 | [Vector Addition](https://leetgpu.com/challenges/vector-addition) | 1D 索引模板 | W1 |
| 02 | [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) | 2D 索引 / naive GEMM | W3 |
| 03 | [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose) | 2D / 读写不对称 | W3 |
| 07 | [Color Inversion](https://leetgpu.com/challenges/color-inversion) | uchar / 像素级并行 | W1 |
| 08 | [Matrix Addition](https://leetgpu.com/challenges/matrix-addition) | 2D 当 1D 处理 | W1 |
| 09 | [1D Convolution](https://leetgpu.com/challenges/1d-convolution) | 滑窗 / 邻域重叠 | W4 |
| 19 | [Reverse Array](https://leetgpu.com/challenges/reverse-array) | in-place 配对 swap | W1 |
| 21 | [ReLU](https://leetgpu.com/challenges/relu) | 激活 / fmaxf | W1 |
| 23 | [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu) | 分段函数 / α=0.01 | W1 |
| 24 | [Rainbow Table](https://leetgpu.com/challenges/rainbow-table) | `__device__` 函数 / 迭代 | W1 |
| 31 | [Matrix Copy](https://leetgpu.com/challenges/matrix-copy) | 显存带宽基线 | W1 |
| 52 | [SiLU](https://leetgpu.com/challenges/silu) | sigmoid * x | W1 |
| 54 | [SwiGLU](https://leetgpu.com/challenges/swish-gated-linear-unit) | 切半 + SiLU(x1)*x2 | W2 |
| 62 | [Value Clipping](https://leetgpu.com/challenges/value-clipping) | fminf/fmaxf | W1 |
| 63 | [Interleave](https://leetgpu.com/challenges/interleave-two-arrays) | 输出 stride=2 | W1 |
| 65 | [GEGLU](https://leetgpu.com/challenges/geglu) | erff / GELU 精确版 | W2 |
| 66 | [RGB → Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale) | 输入 stride=3 | W1 |
| 68 | [Sigmoid](https://leetgpu.com/challenges/sigmoid) | expf 激活函数 | W1 |

> #41 Simple Inference 是 PyTorch only(没有 CUDA kernel),不在本目录内。

## 文件结构约定

每个 `.mu` 文件分三段(注释里标出):

```
PART I    题面(中文摘录 + 约束)
PART II   MUSA kernel + extern "C" solve()
PART III  MUSA vs CUDA 的差异点 / 易踩的坑 / 后续优化空间
+ main()  本地 smoke test(用 Example 1 的输入)
```

## 提交到 LeetGPU 的方法

`solve()` 的签名跟 LeetGPU 官方 starter 完全一致,只需要做两处替换:

```diff
- #include <musa_runtime.h>
+ #include <cuda_runtime.h>

- musaDeviceSynchronize();
+ cudaDeviceSynchronize();
```

然后把 `__global__` kernel + `solve()` 两块代码复制到 LeetGPU 的 CUDA 编辑器里即可。**main() 不用提交**(LeetGPU judge 自己有 driver)。

## 通用移植规则(CUDA → MUSA)

| CUDA | MUSA | 说明 |
|------|------|------|
| `<cuda_runtime.h>` | `<musa_runtime.h>` | 头文件 |
| `cudaMalloc` / `cudaFree` | `musaMalloc` / `musaFree` | 显存 |
| `cudaMemcpy` / `cudaMemcpyKind` | `musaMemcpy` / `musaMemcpyKind` | 拷贝 |
| `cudaDeviceSynchronize` | `musaDeviceSynchronize` | 同步 |
| `cudaError_t` / `cudaGetErrorString` | `musaError_t` / `musaGetErrorString` | 错误 |
| `cudaEvent_t` 系列 | `musaEvent_t` 系列 | 事件计时 |
| `__global__` / `__device__` / `__host__` | 完全相同 | 编译标记 |
| `<<<g, b>>>` 启动语法 | 完全相同 | 启动 |
| `threadIdx` / `blockIdx` / `blockDim` / `gridDim` | 完全相同 | 索引 |
| `fmaxf` / `expf` / `erff` / `fminf` | 完全相同 | 数学库 |

**warp size 差异**(本目录 18 题不涉及,先记着):

- CUDA: warp = 32
- MUSA: warp = 128
- 一旦你的 kernel 用到 `__shfl_*` / `__ballot_*` / warp-level 归约,要重新算
  warp 数量、shared mem padding 大小,以及 reduction tree 的深度。

## 难度梯度建议

学习时按这个顺序刷,从最 trivial 的并行开始,逐步上 2D 索引:

1. **W1 elementwise 基本功**(把 1D 模板写顺手):
   `01 → 21 → 68 → 23 → 31 → 08 → 62`
2. **W1 索引变形题**(读写不对称):
   `19 → 07 → 63 → 66`
3. **W1 进阶 / 数学函数**:
   `52 → 24`
4. **W2 LLM 算子家族**:
   `54 → 65`
5. **W3 2D + GEMM 前夜**:
   `03 → 02`
6. **W4 共享内存优化候选**:
   `09`(naive 版在这,tiled 版后续放 Week 4)

## 性能?

这个目录所有解法都是 **naive 基线**,没有用 shared memory / vector load /
warp-level intrinsic。要做性能优化:

- 在本目录 `.mu` 里加 `_v2`、`_v3` 不同实现,用 `GpuTimer` 对比
- 真正性能调优放到 `code/week4/`,跟着 roadmap 章节一起做
