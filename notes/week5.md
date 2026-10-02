# Week 5 记录

## 2026-10-02 · MTT S4000 · MUSA SDK 3.1.0

| 目标 | 结果 | 关键观察 | 证据 |
|---|---|---|---|
| `01_shared_basics` | PASS | shared memory 基础示例完成 | [日志](../validation/raw/2026-10-02-s4000/week5/01_shared_basics.log) |
| `02_reduce_shared` | PASS | sum=4,194,304，kernel 0.418 ms | [日志](../validation/raw/2026-10-02-s4000/week5/02_reduce_shared.log) |
| `03_transpose_shared` | PASS | padded shared transpose 约 0.317 ms | [日志](../validation/raw/2026-10-02-s4000/week5/03_transpose_shared.log) |
| `04_stencil_constant` | PASS | constant stencil 完成 | [日志](../validation/raw/2026-10-02-s4000/week5/04_stencil_constant.log) |
| `05_naive_gemm` | PASS | 1024³，7.69 ms，279.1 GFLOPS，结果正确 | [日志](../validation/raw/2026-10-02-s4000/week5/05_naive_gemm.log) |
| `06_tiled_gemm` | PASS | 1024³、TS=16，1.35 ms，1590.6 GFLOPS，结果正确 | [日志](../validation/raw/2026-10-02-s4000/week5/06_tiled_gemm.log) |
| `07_mublas_sgemm` | ENV_LIMITED | 当前仍是调用骨架，没有执行真实 muBLAS SGEMM | [日志](../validation/raw/2026-10-02-s4000/week5/07_mublas_sgemm.log) |

## 统一运行记录模板

本模板只填写真实执行结果；没有 SDK、编译器或设备时填写“未运行”，不填估计性能。

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc；版本 | sm_XX / mp_XX | cuda / musa | `chapterXX__name` | M/N/K、dtype、重复次数 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：`make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>`，库或 optional target 需另记依赖和完整输出。
