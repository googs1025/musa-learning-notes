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

5. 在 `code/week5/cuda-reference` 中执行两种 backend 的构建预览：

   ```bash
   make -n BACKEND=cuda
   make -n BACKEND=musa MUSA_ARCH=mp_31
   ```

   对比输出中的编译器、`-arch`/`--offload-arch` 和 include 路径；再在有 SDK 的
   环境显式构建 `chapter07__my-atomic-add`。记录实际编译结果，不能把 dry-run
   当成 MUSA 已兼容。

6. 修改 `checkSmemSquare.cu` 或 `checkSmemRectangle.cu` 的 block 形状、
   `IPAD` 或 shared 配置，验证输出索引关系和错误码；对每个配置记录 block
   线程数、shared 用量和 kernel 时间。若配置触发 launch 错误，保留错误记录。

7. 改变 `reduceInteger.cu` 的输入规模和 block size，分别检查 CPU/GPU sum，
   并记录 shared、unroll 版本的误差与耗时。耗时必须注明是否包含拷贝和 host
   partial 合并；`reduceIntegerShfl.cu` 只在确认 warp/lane 语义后运行。

8. 改变 `floating-point-accuracy.cu` 和 `fmad.cu` 的优化/架构参数，比较输出
   误差；改变 `floating-point-perf.cu` 的 threads/block 与 blocks，记录
   float/double 的三段时间。只报告本机实测值，不填写固定的“double 慢多少”。

9. 重复运行 `atomic-ordering.cu` 并改变 block 数，检查 atomic 版本最终计数，
   记录非原子版本的结果分布；解释 atomic ordering、执行顺序和内存可见性的
   区别。一次运行不能证明固定调度顺序。
