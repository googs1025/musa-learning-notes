# Week 5 习题

> 把运行数据写入 `../../notes/week5.md`。

1. 对比 `05_naive_gemm` 和 `06_tiled_gemm` 的 GFLOPS。
2. 扫描 tiled GEMM 的 `TS`，记录启动失败和性能变化。
3. 补齐 `07_mublas_sgemm` 的本地 SDK 调用，和自写 GEMM 对比。
4. 解释 constant memory 适合 stencil 权重但不适合大数组的原因。
