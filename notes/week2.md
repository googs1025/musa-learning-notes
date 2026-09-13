# Week 2 运行记录

本文件只记录真实执行结果。没有对应 SDK、编译器或设备时，明确填写“未运行”，不要填写估计性能。

## 统一运行记录模板

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc；版本 | sm_XX / mp_XX | cuda / musa | `chapterXX__name` | N、内存类型、stream 数等 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：

```text
make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>
./build/<target>
```
