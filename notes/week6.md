# Week 6 记录

## 2026-10-02 · MTT S4000 · MUSA SDK 3.1.0

| 目标 | 结果 | 关键观察 | 证据 |
|---|---|---|---|
| `01_mccl_allreduce` | ENV_LIMITED | 编译成功；单卡环境不具备多 rank collective 验证条件 | [日志](../validation/raw/2026-10-02-s4000/week6/01_mccl_allreduce.log) |
| `02_musa_gdb_demo` | NOT_RUN | 编译成功；故障注入未在无人值守批处理中执行 | [日志](../validation/raw/2026-10-02-s4000/week6/02_musa_gdb_demo.log) |
| `03_error_dump` | NOT_RUN | 编译成功；需要显式配置 Error Dump 后交互执行 | [日志](../validation/raw/2026-10-02-s4000/week6/03_error_dump.log) |
| `04_torch_musa_minimal` | PASS | PyTorch 2.2.0，MUSA available，sample=0.0 | [日志](../validation/raw/2026-10-02-s4000/week6/04_torch_musa_minimal.log) |
| `05_torch_musa_custom_op` | BUILD_FAIL | Makefile 未提供所需 PyTorch C++ extension 头文件路径 | [日志](../validation/raw/2026-10-02-s4000/week6/05_torch_musa_custom_op.log) |

## 统一运行记录模板

本模板只填写真实执行结果；没有 SDK、编译器或设备时填写“未运行”，不填估计性能。`simpleP2P.c` 为 source-only，若用 MPI 单独验证，应记录 `mpicc`、MPI 版本和 `mpirun` 命令。

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号；GPU 数 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc / mpicc；版本 | sm_XX / mp_XX | cuda / musa / mpi | `chapterXX__name` 或 source-only | 消息大小、矩阵、rank 等 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：`make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>`；`simpleP2P.c` 不使用该 Makefile target，需单独记录 MPI 构建与运行命令。
