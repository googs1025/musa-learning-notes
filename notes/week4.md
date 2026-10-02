# Week 4 记录

## 2026-10-02 · MTT S4000 · MUSA SDK 3.1.0

| 目标 | 结果 | 关键观察 | 证据 |
|---|---|---|---|
| `01_saxpy_bandwidth` | PASS | 0.353 ms，569.82 GB/s | [日志](../validation/raw/2026-10-02-s4000/week4/01_saxpy_bandwidth.log) |
| `02_offset_access` | PASS | 所有配置完成；offset 16/31 均约 0.317 ms | [日志](../validation/raw/2026-10-02-s4000/week4/02_offset_access.log) |
| `03_offset_unrolling` | PASS | 所有配置完成；结果依当前输入与设备解释 | [日志](../validation/raw/2026-10-02-s4000/week4/03_offset_unrolling.log) |
| `04_aos_vs_soa` | PASS | AoS 0.487 ms，SoA 0.318 ms，本次约 1.53x | [日志](../validation/raw/2026-10-02-s4000/week4/04_aos_vs_soa.log) |
| `05_transpose_naive` | PASS | 朴素转置约 0.097 ms | [日志](../validation/raw/2026-10-02-s4000/week4/05_transpose_naive.log) |
| `06_transpose_padded` | PASS | padded 转置约 0.099 ms；本次未显示加速 | [日志](../validation/raw/2026-10-02-s4000/week4/06_transpose_padded.log) |

## 统一运行记录模板

本模板只填写真实执行结果；没有 SDK、编译器或设备时填写“未运行”，不填估计性能。

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc；版本 | sm_XX / mp_XX | cuda / musa | `chapterXX__name` | N、stride、tile 等 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：`make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>`，随后记录 `./build/<target>` 的完整输出。
