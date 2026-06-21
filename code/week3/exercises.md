# Week 3 习题

> 把运行数据写入 `../../notes/week3.md`，不要填伪造性能数字。

1. 对比 `01_warp_divergence` 中 coherent / divergent 的耗时。
2. 比较 `02_reduce_naive` 与 `03_reduce_unrolling` 的带宽和耗时。
3. 确认 `04_reduce_shfl` 在当前 SDK 下的 shuffle mask 写法。
4. 运行 `05_nested_hello`，记录当前 SDK 是否支持动态并行。
5. 修改 `06_sum_matrix_2d` 的 block 形状，观察 16x16 与 32x8 的差异。
