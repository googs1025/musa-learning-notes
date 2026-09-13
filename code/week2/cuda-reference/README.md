# Week 2 CUDA reference：内存与异步 API

来源：[`kriegalex/wrox-pro-cuda-c`](https://github.com/kriegalex/wrox-pro-cuda-c)，固定 commit `63825d64683b644198dd9cb0d4d472d6914d4f72`，MIT license。保留 `.cu` 与 CUDA API。Week 2 通过 `COMMON_DIR=../../week1/cuda-reference/common` 复用 Week 1 公共头，不复制第二份 `common.h`。

| 文件 | 主题 / 来源 | target | CUDA 状态 | MUSA 状态 / 限制 |
|---|---|---|---|---|
| `chapter04/memTransfer.cu` | pageable 显式拷贝 / Ch4 | `chapter04__memTransfer` | 预计可用 `nvcc`；未实编 | 基础路径；未实编 |
| `chapter04/pinMemTransfer.cu` | pinned host memory / Ch4 | `chapter04__pinMemTransfer` | 预计可用 `nvcc`；未实编 | 需 SDK 支持 pinned allocation；未实编 |
| `chapter04/sumArrayZerocpy.cu` | zero-copy mapped host memory / Ch4 | `chapter04__sumArrayZerocpy` | 需 mapped host/device 支持 | CUDA-only/硬件依赖，默认不保证 MUSA |
| `chapter04/sumMatrixGPUManaged.cu` | managed memory / Ch4 | `chapter04__sumMatrixGPUManaged` | 需合适 Toolkit/设备 | MUSA managed memory 能力与迁移语义需实测 |
| `chapter04/sumMatrixGPUManual.cu` | 显式矩阵拷贝 / Ch4 | `chapter04__sumMatrixGPUManual` | 预计可用 `nvcc`；未实编 | 基础对照；未实编 |
| `chapter06/asyncAPI.cu` | async copy + event / Ch6 | `chapter06__asyncAPI` | 预计可用 `nvcc`；未实编 | 需 pinned memory；未实编 |
| `chapter06/simpleCallback.cu` | stream callback / Ch6 | `chapter06__simpleCallback` | 依赖 callback API | MUSA callback 支持/回调线程语义需实测 |
| `chapter06/simpleHyperqBreadth.cu` | Hyper-Q breadth submission / Ch6 | `chapter06__simpleHyperqBreadth` | CUDA/特定硬件观察例 | MUSA 不承诺 Hyper-Q 等价语义，建议单独构建 |
| `chapter06/simpleHyperqDependence.cu` | 跨 stream event 依赖 / Ch6 | `chapter06__simpleHyperqDependence` | CUDA/特定硬件观察例 | Hyper-Q 与设备并发能力依赖，未实编 |

## 运行

```bash
cd code/week2/cuda-reference
make BACKEND=cuda TARGET=chapter04__memTransfer
./build/chapter04__memTransfer
make BACKEND=musa MUSA_ARCH=mp_31 TARGET=chapter06__asyncAPI
```

默认 `make` 会发现全部 `.cu`；zero-copy、managed memory、callback 和 Hyper-Q 不是“默认兼容”的证明，建议按 target 单独验证。没有 SDK 时使用 `make -n`。
