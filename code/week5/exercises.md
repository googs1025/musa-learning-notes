# Week 5 习题

> 把运行数据写入 `../../notes/week5.md`。

1. 对比 `05_naive_gemm` 和 `06_tiled_gemm` 的 GFLOPS。

   预计输出 / 预期现象：

   ```text
   Naive GEMM  M=N=K=...  time=... ms  perf=... GFLOPS  C[0]=... (expect ...)
   Tiled GEMM  M=N=K=...  TS=...  time=... ms  perf=... GFLOPS  C[0]=...
   ```

   两个版本的 `C[0]` 应正确；tiled 版本通常显著高于 naive，因为 A/B tile 被放进 shared memory 后复用。

2. 扫描 tiled GEMM 的 `TS`，记录启动失败和性能变化。

   预计输出 / 预期现象：

   ```text
   Tiled GEMM  M=N=K=...  TS=8   time=... ms  perf=... GFLOPS
   Tiled GEMM  M=N=K=...  TS=16  time=... ms  perf=... GFLOPS
   Tiled GEMM  M=N=K=...  TS=32  time=... ms  perf=... GFLOPS
   ```

   `TS` 太小会复用不足；太大可能因 threads/block、shared memory 或寄存器压力导致性能下降，甚至 launch 失败。

3. 补齐 `07_mublas_sgemm` 的本地 SDK 调用，和自写 GEMM 对比。

   预计输出 / 预期现象：

   ```text
   Week 5 muBLAS SGEMM comparison skeleton
   Shape: M=... N=... K=...
   muBLAS header found.
   Record: naive GFLOPS, tiled GFLOPS, muBLAS GFLOPS, ratio to tiled.
   ```

   如果没有 muBLAS 头文件，会看到 `muBLAS header not found` 和后续补齐提示。补齐后，muBLAS 通常应快于手写 naive/tiled 基线。

4. 解释 constant memory 适合 stencil 权重但不适合大数组的原因。

   预计输出 / 预期现象：

   ```text
   constant stencil done
   ```

   constant memory 适合小而只读、所有线程反复访问的权重，例如 stencil filter；不适合大数组流式访问，因为容量有限且访问模式分散时缓存收益低。
