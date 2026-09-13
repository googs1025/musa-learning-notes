# Week 6 记录

| Topic | Command / Setup | Result | Log path |
|---|---|---|---|
| MUSA SDK 调试器 | | | |
| Error Dump | | | |
| MCCL AllReduce | | | |
| torch_musa minimal | | | |

## 统一运行记录模板

本模板只填写真实执行结果；没有 SDK、编译器或设备时填写“未运行”，不填估计性能。`simpleP2P.c` 为 source-only，若用 MPI 单独验证，应记录 `mpicc`、MPI 版本和 `mpirun` 命令。

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号；GPU 数 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc / mpicc；版本 | sm_XX / mp_XX | cuda / musa / mpi | `chapterXX__name` 或 source-only | 消息大小、矩阵、rank 等 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：`make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>`；`simpleP2P.c` 不使用该 Makefile target，需单独记录 MPI 构建与运行命令。
