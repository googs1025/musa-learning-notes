# Week 4 习题

> 把运行数据写入 `../../notes/week4.md`。

1. 记录 `01_saxpy_bandwidth` 的 GB/s，与理论带宽比值。
2. 扫描 `02_offset_access` 的 offset 0/1/2/4/8/16/31。
3. 比较 `03_offset_unrolling` 是否改善 offset 版本。
4. 对比 AoS 和 SoA，解释哪种更适合 GPU 合并访存。
5. 对比朴素转置和 padded shared 转置。
