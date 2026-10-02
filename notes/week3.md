# Week 3 记录

## 2026-10-02 · MTT S4000 · MUSA SDK 3.1.0

| 目标 | 结果 | 关键观察 | 证据 |
|---|---|---|---|
| `01_warp_divergence` | PASS | coherent 0.322 ms，divergent 0.325 ms；仅代表本次输入 | [日志](../validation/raw/2026-10-02-s4000/week3/01_warp_divergence.log) |
| `02_reduce_naive` | PASS | sum=4,194,304，kernel 1.484 ms | [日志](../validation/raw/2026-10-02-s4000/week3/02_reduce_naive.log) |
| `03_reduce_unrolling` | PASS | sum=4,194,304，kernel 0.562 ms | [日志](../validation/raw/2026-10-02-s4000/week3/03_reduce_unrolling.log) |
| `04_reduce_shfl` | NOT_RUN（修复后） | 旧版得到正确 sum 并报告 `warpSize=32`，但发现 shared 分配写死 128；修复后未复测 | [旧版日志](../validation/raw/2026-10-02-s4000/week3/04_reduce_shfl.log) |
| `05_nested_hello` | PASS | 当前 SDK/设备支持示例中的动态并行 | [日志](../validation/raw/2026-10-02-s4000/week3/05_nested_hello.log) |
| `06_sum_matrix_2d` | PASS | CPU/GPU 总和一致，最大行误差为 0 | [日志](../validation/raw/2026-10-02-s4000/week3/06_sum_matrix_2d.log) |
| `07_sum_matrix_1d` | PASS | CPU/GPU 总和一致，最大行误差为 0 | [日志](../validation/raw/2026-10-02-s4000/week3/07_sum_matrix_1d.log) |

## 统一运行记录模板

本模板只填写真实执行结果；没有 SDK、编译器或设备时填写“未运行”，不填估计性能。

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc；版本 | sm_XX / mp_XX | cuda / musa | `chapterXX__name` | N、grid/block、warp 等 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：`make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>`，随后记录 `./build/<target>` 的完整输出。
