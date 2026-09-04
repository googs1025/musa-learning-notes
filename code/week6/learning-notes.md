# Week 6 学习材料

Week 6 是 MUSA 特色收尾：多卡通信、调试、错误 dump、torch_musa 和 custom op。它和 CUDA_Freshman 的重合度不高，更适合以 MUSA 官方文档和本地环境实测为准。

## 阅读顺序

1. `01_mccl_allreduce.cpp`: 理解多卡通信的最小 AllReduce 骨架。
2. `02_musa_gdb_demo.mu`: 用故意 illegal address 学调试入口。
3. `03_error_dump.mu`: 学会复现、定位和记录 error dump。
4. `04_torch_musa_minimal.py`: 验证 PyTorch 到 MUSA 的最小链路。
5. `05_torch_musa_custom_op.cpp`: 看 custom op 注册边界。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_mccl_allreduce.cpp` | communicator、rank、device、collective | 单卡思维直接套多卡 |
| `02_musa_gdb_demo.mu` | 编译调试符号、断点、非法地址定位 | 只看最终错误码不看触发位置 |
| `03_error_dump.mu` | error dump 复现和归档 | 不记录环境导致问题不可复现 |
| `04_torch_musa_minimal.py` | `torch_musa` 设备检查、模型迁移 | 只测 import 不测实际算子 |
| `05_torch_musa_custom_op.cpp` | C++/PyTorch/MUSA 边界 | 把 custom op 当普通 Python 函数 |

## 代码阅读抓手

本周每个例子都要记录环境：

- MUSA SDK 版本。
- GPU 型号和卡数。
- 驱动/容器镜像。
- torch 和 torch_musa 版本。
- 触发命令和完整错误输出。

Week 6 的价值不只是跑通代码，而是形成可复现的调试记录。

## CUDA 对照

CUDA_Freshman 基本不覆盖本周主题。可以只把 NVIDIA cuda-samples 的多卡、调试、profiling 示例作为概念对照，MUSA 实现以官方文档和本仓库代码为准。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
