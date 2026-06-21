# Week 6 · 多卡 + 调试 + 框架

对应官方指南 Ch6 + Ch8 + Ch10。本周示例偏环境相关，默认只构建调试 `.mu` 示例。

| 文件 | 主题 |
|---|---|
| `01_mccl_allreduce.cpp` | MCCL AllReduce 骨架，`make mccl` |
| `02_musa_gdb_demo.mu` | 故意 illegal address，配合 MUSA SDK 调试器 |
| `03_error_dump.mu` | Error Dump 复现入口 |
| `04_torch_musa_minimal.py` | torch_musa 最小检查 |
| `05_torch_musa_custom_op.cpp` | custom op 注册骨架 |

真实错误码、MCCL 带宽、GDB 截图记录到 `../../notes/week6.md` 和 `../../notes/troubleshooting.md`。
