# Week 6 · 多卡 + 调试 + 框架

对应官方指南 Ch6 + Ch8 + Ch10。本周示例偏环境相关，默认只构建调试 `.mu` 示例。

本周教材见 [`learning-notes.md`](learning-notes.md)：聚焦多卡、库调用、调试、error dump 和 torch_musa 的可复现记录。

| 文件 | 主题 |
|---|---|
| `01_mccl_allreduce.cpp` | MCCL AllReduce 骨架，`make mccl` |
| `02_musa_gdb_demo.mu` | 故意 illegal address，配合 MUSA SDK 调试器 |
| `03_error_dump.mu` | Error Dump 复现入口 |
| `04_torch_musa_minimal.py` | torch_musa 最小检查 |
| `05_torch_musa_custom_op.cpp` | custom op 注册骨架 |

真实错误码、MCCL 带宽、GDB 截图记录到 `../../notes/week6.md` 和 `../../notes/troubleshooting.md`。

CUDA / MUSA 调试和多卡对照案例计划见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。

## CUDA reference

`cuda-reference/` 保存固定 commit `63825d64683b644198dd9cb0d4d472d6914d4f72` 的 Ch8–Ch10 精选样例。
源码保留 `.cu` 和 CUDA API 语义；可复用的检查与计时入口统一来自 `code/week1/cuda-reference/common/common.h`。

| 文件 | 主题 | target | CUDA 状态 | MUSA 状态 / 限制 |
|---|---|---|---|---|
| Ch8 `cublas.cu` | cuBLAS SGEMV | `chapter08__cublas` | CUDA library 候选，未实编 | MUSA Mapping / 对应库候选，未验证 |
| Ch8 `cusparse.cu` | cuSPARSE CSR SpMV | `chapter08__cusparse` | CUDA library 候选，未实编 | MUSA Mapping / 对应库候选，未验证 |
| Ch8 `cufft.cu` | cuFFT C2C FFT | `chapter08__cufft` | CUDA library 候选，未实编 | MUSA Mapping / 对应库候选，未验证 |
| Ch9 `simpleMultiGPU.cu` | 多卡分片、pinned host、stream overlap | `chapter09__simpleMultiGPU` | CUDA 多 GPU，未实编 | MUSA 多卡与 pinned memory 未验证 |
| Ch9 `simpleP2P.c` | MPI host staging | source-only（无 Make target） | CUDA + MPI C toolchain，需 `mpi.h`、两进程/两卡 | MUSA/MPI 互操作未验证 |
| Ch9 `simpleP2P_PingPong.cu` | GPU P2P 与双向异步拷贝 | `chapter09__simpleP2P_PingPong` | CUDA P2P，需两卡拓扑 | MUSA P2P API/拓扑未验证 |
| Ch10 `debug-hazards.cu` | 共享归约调试风险 | `chapter10__debug-hazards` | CUDA 调试候选，未实编 | MUSA GDB 行为未验证 |
| Ch10 `debug-segfault.cu` | 故意 device pointer 错误 | `chapter10__debug-segfault` | CUDA 故障样例 | MUSA 调试候选，未验证 |
| Ch10 `debug-segfault.fixed.cu` | device pointer table 修复 | `chapter10__debug-segfault.fixed` | CUDA 对照，未实编 | MUSA Mapping 候选，未验证 |
| Ch10 `sumMatrixGPU.cu` | 2D 矩阵加法 | `chapter10__sumMatrixGPU` | CUDA runtime，未实编 | MUSA Mapping 候选，未验证 |
| Ch10 `crypt.parallelized.cu` | IDEA 分块并行加密 | `chapter10__crypt.parallelized` | CUDA kernel，需输入/密钥文件 | MUSA kernel、文件 I/O 未验证 |
| Ch10 `crypt.overlap.cu` | 分块 stream overlap 加密 | `chapter10__crypt.overlap` | CUDA streams，需输入/密钥文件 | MUSA stream 语义未验证 |

默认 `make` 排除库、P2P、debug 和 crypt 等额外依赖或高风险目标；`simpleP2P.c` 是 source-only，
不进入 Makefile 的 `SOURCES`、`TARGETS` 或 `optional`。使用 `make optional`
或 `make BACKEND=cuda TARGET=chapter10__debug-segfault.fixed` 显式构建；`BACKEND=musa` 使用
`--offload-arch=$(MUSA_ARCH)`。当前无 CUDA/MUSA SDK，只能做 dry-run，不能声称真实编译通过。
