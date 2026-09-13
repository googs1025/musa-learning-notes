# Week 4 CUDA reference · memory access

来源：[`kriegalex/wrox-pro-cuda-c`](https://github.com/kriegalex/wrox-pro-cuda-c)，固定 commit `63825d64683b644198dd9cb0d4d472d6914d4f72`。源码保留 `.cu` 和版权头，并通过 `-I../../week1/cuda-reference/common` 使用共享 `common.h`。

| 上游 chapter / 文件 | target | 主题 | CUDA | MUSA |
|---|---|---|---|---|
| chapter04/readSegment.cu | `chapter04__readSegment` | 读 segment / 合并访存 | 预计可用，未实编 | 未验证 |
| chapter04/writeSegment.cu | `chapter04__writeSegment` | 写 segment / stride | 预计可用，未实编 | 未验证 |
| chapter04/readSegmentUnroll.cu | `chapter04__readSegmentUnroll` | offset + unroll | 预计可用，未实编 | 未验证，性能不推定 |
| chapter04/simpleMathAoS.cu | `chapter04__simpleMathAoS` | Array of Structs | 预计可用，未实编 | 未验证 |
| chapter04/simpleMathSoA.cu | `chapter04__simpleMathSoA` | Struct of Arrays | 预计可用，未实编 | 未验证 |
| chapter04/transpose.cu | `chapter04__transpose` | 转置读写方向 | 预计可用，未实编 | shared/bank 行为需实测 |
| chapter04/globalVariable.cu | `chapter04__globalVariable` | device/global symbol | 预计可用，未实编 | symbol 映射需 SDK 验证 |

```bash
cd code/week4/cuda-reference
make BACKEND=cuda
make BACKEND=musa MUSA_ARCH=mp_31
make BACKEND=cuda TARGET=chapter04__readSegment
make clean
```

功能正确不等于带宽提升；带宽数字必须记录数据规模、计时边界、CUDA Toolkit/MUSA SDK、设备和架构。当前未进行真实编译。
