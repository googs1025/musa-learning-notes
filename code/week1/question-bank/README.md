# Week 1 题库

本目录收 Week 1 之后可以继续刷的 kernel 基础题。主线习题仍看
[`../exercises.md`](../exercises.md)；这里偏向补手感：1D 索引、边界判断、device
memory、简单数学函数和 stride 访存。

题目代码已在 [`../../leetgpu/easy/`](../../leetgpu/easy/) 给出 MUSA 移植版。建议先自己写
`kernel + solve()`，再对照仓库里的基线实现。

## 必做：elementwise 基本功

| 题号 | 题目 | 本地文件 | 练习重点 |
|---|---|---|---|
| 01 | Vector Addition | [`../../leetgpu/easy/01_vector_add.mu`](../../leetgpu/easy/01_vector_add.mu) | 1D 全局索引、越界保护、H2D/D2H |
| 21 | ReLU | [`../../leetgpu/easy/21_relu.mu`](../../leetgpu/easy/21_relu.mu) | 单输入单输出 elementwise |
| 68 | Sigmoid | [`../../leetgpu/easy/68_sigmoid.mu`](../../leetgpu/easy/68_sigmoid.mu) | `expf` 和浮点数学函数 |
| 23 | Leaky ReLU | [`../../leetgpu/easy/23_leaky_relu.mu`](../../leetgpu/easy/23_leaky_relu.mu) | 分支表达式、负数路径 |
| 62 | Value Clipping | [`../../leetgpu/easy/62_value_clipping.mu`](../../leetgpu/easy/62_value_clipping.mu) | `fminf` / `fmaxf` 组合 |
| 52 | SiLU | [`../../leetgpu/easy/52_silu.mu`](../../leetgpu/easy/52_silu.mu) | 复合 elementwise：`x * sigmoid(x)` |

## 进阶：索引和访存变形

| 题号 | 题目 | 本地文件 | 练习重点 |
|---|---|---|---|
| 31 | Matrix Copy | [`../../leetgpu/easy/31_matrix_copy.mu`](../../leetgpu/easy/31_matrix_copy.mu) | 显存读写基线、吞吐直觉 |
| 08 | Matrix Addition | [`../../leetgpu/easy/08_matrix_addition.mu`](../../leetgpu/easy/08_matrix_addition.mu) | 2D 数据按 1D flatten 处理 |
| 19 | Reverse Array | [`../../leetgpu/easy/19_reverse_array.mu`](../../leetgpu/easy/19_reverse_array.mu) | in-place 配对访问，只启动半数有效线程 |
| 63 | Interleave | [`../../leetgpu/easy/63_interleave.mu`](../../leetgpu/easy/63_interleave.mu) | 输出 stride=2 的写入模式 |
| 66 | RGB to Grayscale | [`../../leetgpu/easy/66_rgb_to_grayscale.mu`](../../leetgpu/easy/66_rgb_to_grayscale.mu) | 输入 stride=3 的读取模式 |
| 07 | Color Inversion | [`../../leetgpu/easy/07_color_inversion.mu`](../../leetgpu/easy/07_color_inversion.mu) | `uchar` 数据、像素级并行 |

## 选做：Week 1 上限题

| 题号 | 题目 | 本地文件 | 练习重点 |
|---|---|---|---|
| 24 | Rainbow Table | [`../../leetgpu/easy/24_rainbow_table.mu`](../../leetgpu/easy/24_rainbow_table.mu) | `__device__` 函数、寄存器内迭代 |

## 刷题顺序

1. `01 -> 21 -> 68 -> 23 -> 62 -> 52`
2. `31 -> 08 -> 19 -> 63 -> 66 -> 07`
3. `24`

## 收录边界

暂不放入 Week 1：

| 题号 | 原因 |
|---|---|
| 54 SwiGLU / 65 GEGLU | 更适合 Week 2 的 LLM 算子家族 |
| 03 Matrix Transpose / 02 Matrix Multiplication | 更适合 Week 3 的 2D 索引和 naive GEMM |
| 09 1D Convolution | 后续可以接 Week 4 shared memory tile 优化 |
