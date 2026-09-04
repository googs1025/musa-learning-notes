# Week 4 学习材料

Week 4 专注全局内存访问。GPU kernel 的算术经常不是瓶颈，访存地址是否连续、是否合并、数据布局是否适合线程访问，才决定吞吐上限。

## 阅读顺序

1. `01_saxpy_bandwidth.mu`: 建立带宽利用率基线。
2. `02_offset_access.mu`: 观察 offset 如何破坏合并访存。
3. `03_offset_unrolling.mu`: 看展开能否弥补访存损失。
4. `04_aos_vs_soa.mu`: 对比 AoS 和 SoA 对连续访问的影响。
5. `05_transpose_naive.mu`: 观察转置中的读写方向冲突。
6. `06_transpose_padded.mu`: 用 shared tile 和 padding 改善转置。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_saxpy_bandwidth.mu` | effective bandwidth、理论带宽对照 | 只看 kernel 时间不算吞吐 |
| `02_offset_access.mu` | 合并访存和 cache line 对齐 | 以为 offset 只是多加一个整数 |
| `03_offset_unrolling.mu` | unroll 与访存模式的交互 | 以为 unroll 一定提速 |
| `04_aos_vs_soa.mu` | 数据布局决定线程读到的地址序列 | 用 CPU 结构体直觉写 GPU 数据 |
| `05_transpose_naive.mu` | 转置的读连续/写跨步问题 | 只检查结果正确不检查带宽 |
| `06_transpose_padded.mu` | shared tile、padding、bank conflict | 以为 shared memory 自动更快 |

## 代码阅读抓手

每个 kernel 都问同一个问题：相邻线程访问的地址是否相邻。

如果相邻线程访问的是 `a[i]`、`a[i+1]`、`a[i+2]`，通常更容易合并；如果访问的是 `a[i * stride]` 或结构体中的分散字段，就要警惕吞吐下降。

## 高频混淆点

- **结果正确不代表性能正确**: offset、stride、AoS 代码可能结果完全正确, 但相邻线程访问地址不连续, 带宽会明显下降。
- **合并访存看的是线程组访问序列**: 不要只看单个线程访问了什么, 要看相邻线程在同一条 load/store 指令上访问的地址是否相邻。
- **AoS 是 CPU 友好, 不一定 GPU 友好**: GPU 上一组线程常常只读同一个字段, SoA 更容易让地址连续。
- **转置有读写两个方向**: naive transpose 往往读连续但写跨步, 或反过来。优化时要分别看 load 和 store。
- **unroll 不是万能加速**: unroll 能减少循环/调度开销, 但如果访存模式差或寄存器压力变大, 可能收益很小甚至变慢。

## CUDA_Freshman 对照

- `18_sum_array_offset`: offset 访问。
- `19_AoS`、`20_SoA`: 数据布局。
- `21_sum_array_offset_unrolling`: offset + unrolling。
- `22_transform_matrix2D`: matrix transpose。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
