# Week 4 · 全局内存与访存

对应官方指南 Ch9。本周所有示例都围绕一个问题：线程访问地址是否连续，决定吞吐上限。

| 文件 | 主题 |
|---|---|
| `01_saxpy_bandwidth.mu` | SAXPY 带宽利用率 |
| `02_offset_access.mu` | offset 对合并访存的影响 |
| `03_offset_unrolling.mu` | offset + unroll 是否能抵消开销 |
| `04_aos_vs_soa.mu` | AoS / SoA 数据布局对比 |
| `05_transpose_naive.mu` | 朴素转置 |
| `06_transpose_padded.mu` | shared tile + padding 转置 |

结果记录到 `../../notes/week4.md`。
