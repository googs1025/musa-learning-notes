# Week 6 CUDA reference

来源：kriegalex/wrox-pro-cuda-c，固定 commit
63825d64683b644198dd9cb0d4d472d6914d4f72。源码保留上游版权头和 CUDA 语义；
可复用的 .cu 示例统一使用 code/week1/cuda-reference/common/common.h。

## 文件清单

| Chapter / 文件 | 主题 | target / 状态 | CUDA 状态 | MUSA 状态 / 命令 |
|---|---|---|---|---|
| `chapter08/cublas.cu` | cuBLAS SGEMV | chapter08__cublas，optional | CUDA library，未实编 | MUSA Mapping/对应库候选，未验证 |
| `chapter08/cusparse.cu` | CSR SpMV | chapter08__cusparse，optional | CUDA library，未实编 | MUSA Mapping/对应库候选，未验证 |
| `chapter08/cufft.cu` | C2C FFT | chapter08__cufft，optional | CUDA library，未实编 | MUSA Mapping/对应库候选，未验证 |
| `chapter09/simpleMultiGPU.cu` | 多卡分片与 stream overlap | chapter09__simpleMultiGPU，默认 all | CUDA runtime，需多 GPU，未实编 | MUSA 多卡/pinned memory 未验证 |
| `chapter09/simpleP2P.c` | MPI host staging | source-only，不可构建 | CUDA + MPI C toolchain，需 mpi.h、两进程/两卡 | 不进入 MUSA Make target，需外部 MPI C toolchain |
| `chapter09/simpleP2P_PingPong.cu` | GPU P2P ping-pong | chapter09__simpleP2P_PingPong，optional | CUDA P2P，需两卡拓扑，未实编 | MUSA P2P API/拓扑未验证 |
| `chapter10/debug-hazards.cu` | 共享归约调试风险 | chapter10__debug-hazards，optional | CUDA 调试候选，未实编 | MUSA GDB/错误行为未验证 |
| `chapter10/debug-segfault.cu` | 故意 device pointer 错误 | chapter10__debug-segfault，optional | CUDA 故障样例 | MUSA 调试器未验证，隔离运行 |
| `chapter10/debug-segfault.fixed.cu` | pointer table 修复 | chapter10__debug-segfault.fixed，optional | CUDA 对照，未实编 | MUSA Mapping 候选，未验证 |
| `chapter10/sumMatrixGPU.cu` | 2D 矩阵加法 | chapter10__sumMatrixGPU，默认 all | CUDA runtime，未实编 | MUSA Mapping 候选，未验证 |
| `chapter10/crypt.parallelized.cu` | IDEA 分块并行加密 | chapter10__crypt.parallelized，optional | CUDA kernel，需输入/密钥文件 | MUSA kernel/文件 I/O 未验证 |
| `chapter10/crypt.overlap.cu` | IDEA stream overlap 加密 | chapter10__crypt.overlap，optional | CUDA streams，需输入/密钥文件 | MUSA stream 语义未验证 |

## 构建

    make BACKEND=cuda
    make BACKEND=musa MUSA_ARCH=mp_31
    make BACKEND=cuda optional
    make BACKEND=cuda TARGET=chapter10__sumMatrixGPU
    make clean

默认 all 只构建非 optional 的 .cu 目标；库、P2P、debug 和 crypt 通过 optional
或显式 TARGET 构建。simpleP2P.c 保留为 source-only，不会被发现，也不会出现在
TARGET 或 optional。clean 只删除当前目录的 ./build。

当前环境无 CUDA/MUSA SDK，以上命令仅用于 dry-run 验证。
