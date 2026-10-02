# Week 2 运行记录

本文件只记录真实执行结果。没有对应 SDK、编译器或设备时，明确填写“未运行”，不要填写估计性能。

## 2026-10-02 · MTT S4000 · MUSA SDK 3.1.0

| 目标 | 结果 | 关键观察 | 证据 |
|---|---|---|---|
| `01_vector_add_runtime` | PASS | 1,048,576 个元素校验通过 | [日志](../validation/raw/2026-10-02-s4000/week2/01_vector_add_runtime.log) |
| `02_vector_add_pinned` | PASS | Pinned H2D 约 0.602 ms、25.95 GB/s | [日志](../validation/raw/2026-10-02-s4000/week2/02_vector_add_pinned.log) |
| `03_vector_add_timer` | PASS | Event 计时约 0.1243 ms | [日志](../validation/raw/2026-10-02-s4000/week2/03_vector_add_timer.log) |
| `04_vector_add_unified` | PASS | 计算校验通过；本设备不支持 prefetch，示例按设计跳过 | [日志](../validation/raw/2026-10-02-s4000/week2/04_vector_add_unified.log) |
| `05_multi_stream` | PASS | 四流流水线约 5.284 ms，结果正确 | [日志](../validation/raw/2026-10-02-s4000/week2/05_multi_stream.log) |
| `06_stream_event_dep` | PASS | 跨流依赖校验 `bad=0` | [日志](../validation/raw/2026-10-02-s4000/week2/06_stream_event_dep.log) |
| `07_musa_graph` | PASS | 5,000 次重放结果为预期 25,000 | [日志](../validation/raw/2026-10-02-s4000/week2/07_musa_graph.log) |
| `08_stream_callback` | PASS | 4 个 callback 全部完成 | [日志](../validation/raw/2026-10-02-s4000/week2/08_stream_callback.log) |

## 统一运行记录模板

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc；版本 | sm_XX / mp_XX | cuda / musa | `chapterXX__name` | N、内存类型、stream 数等 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：

```text
make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>
./build/<target>
```
