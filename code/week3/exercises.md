# Week 3 习题

## E3.1 双后端与 warp 差异实验

在 `code/week3/cuda-reference` 分别执行 `make -n BACKEND=cuda TARGET=chapter03__reduceIntegerShfl` 与 `make -n BACKEND=musa MUSA_ARCH=mp_31 TARGET=chapter03__reduceIntegerShfl`，记录编译器、架构参数和源码中的 `warpSize`/mask 假设。若有设备，再分别运行 `simpleShfl`，不要把 CUDA 32-lane 输出直接当作 MUSA 正确性证明。

## E3.2 默认 all 与可选目标

解释为什么动态并行和旧 shuffle 目标不应默认执行；分别显式构建 `chapter03__nestedHelloWorld`，记录失败位置、SDK 版本和替代的 host-side 两阶段实现。

> 把运行数据写入 `../../notes/week3.md`，不要填伪造性能数字。

1. 对比 `01_warp_divergence` 中 coherent / divergent 的耗时。

   预计输出 / 预期现象：

   ```text
   coherent=... ms divergent=... ms slowdown=...x
   ```

   `divergent` 通常慢于 `coherent`，因为同一 warp 内不同分支需要被拆开执行。具体 slowdown 取决于数据规模、编译优化和硬件调度。

2. 比较 `02_reduce_naive` 与 `03_reduce_unrolling` 的带宽和耗时。

   预计输出 / 预期现象：

   ```text
   sum=4194304 expected=4194304 kernel=... ms partial_blocks=...
   ```

   两个版本的 `sum` 都应等于 `expected`；unrolling 版本通常更快，因为每个线程处理更多元素，减少 block 数和部分全局内存访问开销。

3. 确认 `04_reduce_shfl` 在当前 SDK 下的 shuffle mask 写法。

   预计输出 / 预期现象：

   ```text
   sum=1048576 expected=1048576 warpSize=128
   ```

   正确时 `sum == expected`。如果 mask 或 warp size 假设不匹配，可能编译失败，或运行结果小于 expected。

4. 运行 `05_nested_hello`，记录当前 SDK 是否支持动态并行。

   预计输出 / 预期现象：

   ```text
   parent launches child from block=0
   child block=0
   child block=1
   ...
   If this fails to compile or launch, dynamic parallelism is not enabled in this SDK/config.
   ```

   如果 SDK / 编译选项不支持动态并行，可能在编译、链接或 kernel launch 阶段失败；把具体错误记录下来。

5. 修改 `06_sum_matrix_2d` 的 block 形状，观察 16x16 与 32x8 的差异。

   预计输出 / 预期现象：

   ```text
   row0=... expected=... row_last=...
   ```

   正确时 `row0` 和 `row_last` 都应等于 `expected`。`16x16` 与 `32x8` 总线程数相同，但访存形状和调度细节不同，耗时可能略有差异。
