# Week 3 CUDA reference · execution model

来源：[`kriegalex/wrox-pro-cuda-c`](https://github.com/kriegalex/wrox-pro-cuda-c)，固定 commit `63825d64683b644198dd9cb0d4d472d6914d4f72`。源码保留 `.cu` 和版权头；共享头来自 `../../week1/cuda-reference/common/common.h`，没有复制第二份 `common.h`。

本目录的 target 由 Makefile 自动发现，路径 `/` 编码为 `__`。默认 `make all` 排除动态并行和旧 shuffle 目标；需要显式实验时再指定 `TARGET`。

| 上游 chapter / 文件 | target | 主题 | CUDA | MUSA |
|---|---|---|---|---|
| chapter03/simpleDivergence.cu | `chapter03__simpleDivergence` | divergence 与 warp 分支 | 预计可用，未实编 | `warpSize`/计时未验证 |
| chapter03/reduceInteger.cu | `chapter03__reduceInteger` | shared reduce、unroll | 预计可用，未实编 | 未验证，检查 block/warp 假设 |
| chapter03/reduceIntegerShfl.cu | `chapter03__reduceIntegerShfl` | shuffle reduce | 预计可用，未实编 | warp/mask 语义需 SDK 实测 |
| chapter03/simpleShfl.cu | `chapter03__simpleShfl` | shuffle API 探针 | 旧 intrinsic，未验证 | **可选**，旧 intrinsic/32-lane 假设 |
| chapter03/nestedHelloWorld.cu | `chapter03__nestedHelloWorld` | dynamic parallelism | **可选**，需架构支持 | **可选**，MUSA 支持未验证 |
| chapter03/nestedReduce.cu | `chapter03__nestedReduce` | device-side recursive launch | **可选**，需动态并行 | **可选**，不纳入默认 all |
| chapter03/sumMatrix.cu | `chapter03__sumMatrix` | 2D grid 矩阵加法 | 预计可用，未实编 | 未验证 |
| chapter05/checkSmemSquare.cu | `chapter05__checkSmemSquare` | shared shape / bank mapping | 预计可用，未实编 | 设备布局需实测 |
| chapter05/checkSmemRectangle.cu | `chapter05__checkSmemRectangle` | rectangular shared tile | 预计可用，未实编 | 设备布局需实测 |

```bash
cd code/week3/cuda-reference
make BACKEND=cuda
make BACKEND=musa MUSA_ARCH=mp_31
make BACKEND=cuda TARGET=chapter03__simpleDivergence
make BACKEND=cuda TARGET=chapter03__nestedHelloWorld  # 可选
make clean
```

`BACKEND=cuda` 使用 `nvcc`；`BACKEND=musa` 使用 `mcc -mtgpu --offload-arch=...`。当前环境无 CUDA/MUSA SDK，以上状态均为未验证，不代表真实编译通过。
