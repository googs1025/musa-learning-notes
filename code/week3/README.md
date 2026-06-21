# Week 3 · 执行模型

对应官方指南 Ch5 + Ch9。本周用 reduce 串起 warp divergence、循环展开、warp shuffle、动态并行和 2D grid。

| 文件 | 主题 |
|---|---|
| `01_warp_divergence.mu` | coherent vs divergent 分支耗时对比 |
| `02_reduce_naive.mu` | global memory + host final reduce 基线 |
| `03_reduce_unrolling.mu` | 每线程处理 2 个元素的展开归约 |
| `04_reduce_shfl.mu` | warp shuffle 归约骨架，MUSA warp=128 时需按 SDK 调整 mask |
| `05_nested_hello.mu` | 动态并行骨架：device kernel 启动子 kernel |
| `06_sum_matrix_2d.mu` | 2D grid 矩阵求和 |

```bash
cd code/week3
make
./01_warp_divergence
./02_reduce_naive
./03_reduce_unrolling
./04_reduce_shfl
./05_nested_hello
./06_sum_matrix_2d
```

结果记录到 `../../notes/week3.md`。
