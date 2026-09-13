# Week 1 CUDA reference：基础执行模型

来源：[`kriegalex/wrox-pro-cuda-c`](https://github.com/kriegalex/wrox-pro-cuda-c)，固定 commit `63825d64683b644198dd9cb0d4d472d6914d4f72`，MIT license。源码保留 `.cu` 和 CUDA API；本目录的 Makefile 用 `BACKEND=cuda` 选择 `nvcc`，用 `BACKEND=musa` 选择 `mcc -mtgpu`。MUSA 状态均为“未在本机实编”，兼容性取决于 SDK、MUSA Mapping、架构和 API 支持。

| 文件 | 主题 / 来源 | target | CUDA 状态 | MUSA 状态 / 限制 |
|---|---|---|---|---|
| `chapter01/hello.cu` | kernel 与 GPU printf / Ch1 | `chapter01__hello` | 预计可用 `nvcc`；未实编 | 基础语法通常可映射；未实编 |
| `chapter02/checkDeviceInfor.cu` | 设备属性 / Ch2 | `chapter02__checkDeviceInfor` | 预计可用 `nvcc`；未实编 | 属性字段随 runtime 变化；未实编 |
| `chapter02/checkDimension.cu` | grid/block/thread 维度 / Ch2 | `chapter02__checkDimension` | 预计可用 `nvcc`；未实编 | 基础 API；未实编 |
| `chapter02/checkThreadIndex.cu` | 线程索引 / Ch2 | `chapter02__checkThreadIndex` | 预计可用 `nvcc`；未实编 | 基础 API；未实编 |
| `chapter02/defineGridBlock.cu` | launch 配置 / Ch2 | `chapter02__defineGridBlock` | 预计可用 `nvcc`；未实编 | 受设备最大 block 限制；未实编 |
| `chapter02/sumArraysOnGPU-small-case.cu` | 单 block 向量加 / Ch2 | `chapter02__sumArraysOnGPU-small-case` | 预计可用 `nvcc`；未实编 | 基础 kernel；未实编 |
| `chapter02/sumArraysOnGPU-timer.cu` | CPU/GPU 计时 / Ch2 | `chapter02__sumArraysOnGPU-timer` | 预计可用 `nvcc`；未实编 | 计时结果不能跨设备外推；未实编 |
| `chapter02/sumMatrixOnGPU-1D-grid-1D-block.cu` | 1D grid/block 矩阵加 / Ch2 | `chapter02__sumMatrixOnGPU-1D-grid-1D-block` | 预计可用 `nvcc`；未实编 | 观察 MUSA warp/block 差异；未实编 |
| `chapter02/sumMatrixOnGPU-2D-grid-1D-block.cu` | 2D grid/1D block / Ch2 | `chapter02__sumMatrixOnGPU-2D-grid-1D-block` | 预计可用 `nvcc`；未实编 | 设备限制依赖 SDK；未实编 |
| `chapter02/sumMatrixOnGPU-2D-grid-2D-block.cu` | 2D grid/2D block / Ch2 | `chapter02__sumMatrixOnGPU-2D-grid-2D-block` | 预计可用 `nvcc`；未实编 | 设备限制依赖 SDK；未实编 |

## 运行

```bash
cd code/week1/cuda-reference
make BACKEND=cuda TARGET=chapter01__hello
./build/chapter01__hello
make BACKEND=musa MUSA_ARCH=mp_31 TARGET=chapter01__hello
```

`make` 默认会尝试编译本目录全部 `.cu`；没有 CUDA/MUSA SDK 时只运行 `make -n` 做命令检查。公共头只有 `code/week1/cuda-reference/common/common.h` 一份，源码通过 `-Icommon` 使用。
