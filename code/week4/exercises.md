# Week 4 习题

> 把运行数据写入 `../../notes/week4.md`。

1. 记录 `01_saxpy_bandwidth` 的 GB/s，与理论带宽比值。

   预计输出 / 预期现象：

   ```text
   SAXPY ... ms ... GB/s
   ```

   GB/s 应明显低于理论峰值，因为 SAXPY 受全局内存带宽、launch 开销和访存效率共同影响。

2. 扫描 `02_offset_access` 的 offset 0/1/2/4/8/16/31。

   预计输出 / 预期现象：

   ```text
   offset= 0 time=... ms
   offset= 1 time=... ms
   offset= 2 time=... ms
   offset= 4 time=... ms
   offset= 8 time=... ms
   offset=16 time=... ms
   offset=31 time=... ms
   ```

   `offset=0` 通常最好；非 0 offset 可能破坏对齐和合并访存，耗时上升。

3. 比较 `03_offset_unrolling` 是否改善 offset 版本。

   预计输出 / 预期现象：

   ```text
   offset= 0 unroll4=... ms
   offset= 1 unroll4=... ms
   ...
   ```

   unroll 可能降低循环和调度开销，但不能从根本上修复非对齐访存；如果瓶颈在内存 transaction，改善会有限。

4. 对比 AoS 和 SoA，解释哪种更适合 GPU 合并访存。

   预计输出 / 预期现象：

   ```text
   AoS=... ms SoA=... ms speedup=...x
   ```

   SoA 通常更快，因为相邻线程访问相邻字段数组，合并访存更自然；AoS 容易让线程跨结构体取字段，访存跨度更大。

5. 对比朴素转置和 padded shared 转置。

   预计输出 / 预期现象：

   ```text
   transpose naive ... ms
   transpose padded ... ms
   ```

   padded shared 版本通常快于朴素转置，因为它把非合并写入转成 shared tile 内重排，并通过 padding 降低 shared memory bank conflict。
