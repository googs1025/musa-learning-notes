# Week 3 学习材料

Week 3 的主线是执行模型。Reduce 是最适合入门的实验对象：它既能暴露线程索引、访存、同步和分支问题，又能逐步演进到 unrolling 和 shuffle。

## 阅读顺序

1. `01_warp_divergence.mu`: 先观察分支分化的代价。
2. `02_reduce_naive.mu`: 建立 naive reduce 基线。
3. `03_reduce_unrolling.mu`: 看每个线程多处理元素如何减少开销。
4. `04_reduce_shfl.mu`: 进入 warp-level reduce。
5. `05_nested_hello.mu`: 理解动态并行的语义和限制。
6. `06_sum_matrix_2d.mu`: 把一维索引扩展到二维数据。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_warp_divergence.mu` | 同一 warp 内分支不一致会串行化路径 | 把 branch 数量等同于性能损失 |
| `02_reduce_naive.mu` | 分块归约、host final reduce | 忽略 block 间不能直接同步 |
| `03_reduce_unrolling.mu` | 每线程多元素、减少 block 数和循环开销 | 只看算术操作不看访存模式 |
| `04_reduce_shfl.mu` | warp shuffle、warp size 差异 | 直接照搬 CUDA 的 32-wide 假设 |
| `05_nested_hello.mu` | device 端 launch 子 kernel | 把动态并行当普通函数调用 |
| `06_sum_matrix_2d.mu` | 2D grid、row-major 展开 | 混淆 `(x,y)` 和 `row/col` |

## 代码阅读抓手

Reduce 代码重点看三个边界：

- block 内如何同步。
- block 间结果如何汇总。
- 最后一段不足一个 block 或一个 warp 时如何处理。

MUSA 的 warp size 和 CUDA 常见值不同，所有 shuffle 或 warp reduce 都必须先确认设备能力和 SDK intrinsic 行为。

## CUDA_Freshman 对照

- `8_divergence`: 分支分化。
- `10_reduceInteger`、`12_reduce_unrolling`、`29_reduce_shfl`: reduce 三阶演进。
- `28_shfl_test`: shuffle API 探测。
- `13_nested_hello_world`: 动态并行。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
