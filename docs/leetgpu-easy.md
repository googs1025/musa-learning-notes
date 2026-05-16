# 练习题:LeetGPU Easy(MUSA 移植版)

[LeetGPU](https://leetgpu.com) 是个 GPU 编程在线 judge,但只支持 CUDA。
本目录把 Easy 难度的题目 **移植成 MUSA** 后整理为练习集,跟仓库的 6 周路线图穿插使用。

代码在 [`code/leetgpu/easy/`](../code/leetgpu/easy/),编译运行说明见
[`code/leetgpu/easy/README.md`](../code/leetgpu/easy/README.md)。

## 为什么用 LeetGPU 当练习

- **题面有梯度**:从 vector add → GEMM → softmax/attention,刚好覆盖路线图前 4 周要练的 kernel pattern
- **judge 给基线**:在 LeetGPU 上能看 reference 实现和性能榜单,有"挑战目标"
- **官方仓库公开题面和 starter**:[AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges),不会突然失效
- **CUDA → MUSA 移植规则非常机械**:这批 Easy 题没用到 warp-level 特性,
  绝大多数移植就是 `cuda*` → `musa*` 替换,练手最舒服

## 题目 ↔ 路线图对应表

刷题顺序按"复杂度递增、对路线图章节有助益"排:

### Week 1 · 线程索引 + 显存(elementwise 基本功)

| # | 题目 | 重点 |
|---|------|------|
| 01 | Vector Addition | 1D 索引模板 |
| 21 | ReLU | 激活函数最简版 |
| 68 | Sigmoid | expf / 浮点数学 |
| 23 | Leaky ReLU | 分段函数 |
| 31 | Matrix Copy | 显存带宽极限基线 |
| 08 | Matrix Addition | 2D 当 1D 处理 |
| 62 | Value Clipping | fminf/fmaxf 组合 |
| 52 | SiLU | sigmoid * x |

### Week 1 进阶 · 不对称访存

| # | 题目 | 重点 |
|---|------|------|
| 19 | Reverse Array | in-place 配对 swap,只让 N/2 线程动手 |
| 07 | Color Inversion | uchar / 像素级 / R G B 分别处理 |
| 63 | Interleave | 输出 stride=2 的写入模式 |
| 66 | RGB to Grayscale | 输入 stride=3 的读取模式 |
| 24 | Rainbow Table | `__device__` 函数 + 寄存器内迭代 |

### Week 2 · LLM 算子家族(给 torch_musa 章节打基础)

| # | 题目 | 重点 |
|---|------|------|
| 54 | SwiGLU | 输入切两半,SiLU(x1) * x2 |
| 65 | GEGLU | erff 实现精确 GELU(非 tanh 近似) |

### Week 3 · 2D 索引 + GEMM 前夜

| # | 题目 | 重点 |
|---|------|------|
| 03 | Matrix Transpose | 2D 索引 / 写入 stride 大 → coalesce 直觉 |
| 02 | Matrix Multiplication | naive GEMM,后续 Week 4 改 tiled 版做对比 |

### Week 4 · 共享内存优化候选

| # | 题目 | 重点 |
|---|------|------|
| 09 | 1D Convolution | naive 版本带宽浪费,改 shared memory tile 经典 |

## CUDA → MUSA 速查

跟 [`cuda-vs-musa.md`](cuda-vs-musa.md) 的对照表一致。本批 18 题用到的接口:

```
cudaMalloc           → musaMalloc
cudaFree             → musaFree
cudaMemcpy           → musaMemcpy
cudaMemcpyHostToDevice → musaMemcpyHostToDevice (枚举值)
cudaDeviceSynchronize → musaDeviceSynchronize
cudaError_t          → musaError_t
<cuda_runtime.h>     → <musa_runtime.h>

__global__ __device__ __host__         不变
<<<grid, block>>> 启动语法              不变
threadIdx blockIdx blockDim gridDim    不变
fmaxf fminf expf erff                  不变(device 数学库同名)
```

## 提交回 LeetGPU 的工作流

写好 MUSA 版本本地跑通后,要在 LeetGPU 上 judge:

1. 在 .mu 文件里抽出 `__global__ kernel` + `extern "C" solve()` 两块
2. 把 `musa_runtime.h` 改回 `cuda_runtime.h`、`musaDeviceSynchronize` 改回 `cudaDeviceSynchronize`
3. 复制粘贴到 LeetGPU 网页编辑器,Submit

我自己一般本地先用 main() 验功能,提交主要看是否能拿到性能榜单上一个位置。

## 下一步

- Medium / Hard 题目会再开一个章节,涉及 shared memory tile、warp-level intrinsic,
  那时候 MUSA warp=128 跟 CUDA warp=32 的差异就要显式处理了
- 这个 Easy 章节定位是"会写 kernel 的最低门槛",刷完 18 题对照 roadmap 就有了 ~50% 的肌肉记忆
