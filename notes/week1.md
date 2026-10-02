# Week 1 运行记录

本文件只记录真实执行结果。没有对应 SDK、编译器或设备时，明确填写“未运行”，不要填写估计性能。

## 2026-10-02 · MTT S4000 · MUSA SDK 3.1.0

| 目标 | 结果 | 关键观察 | 证据 |
|---|---|---|---|
| `01_hello_world` | PASS | CPU 与 5 个 GPU 线程均正常输出 | [日志](../validation/raw/2026-10-02-s4000/week1/01_hello_world.log) |
| `02_thread_index` | PASS | 两个 block 得到全局索引 0–7 | [日志](../validation/raw/2026-10-02-s4000/week1/02_thread_index.log) |
| `03_device_info` | PASS | 识别 MTT S4000；本机 `warpSize=32` | [日志](../validation/raw/2026-10-02-s4000/week1/03_device_info.log) |
| `04_memory_basics` | PASS | 显存分配、kernel 和回传校验通过 | [日志](../validation/raw/2026-10-02-s4000/week1/04_memory_basics.log) |
| `05_error_check` | PASS | 错误检查演示按预期完成 | [日志](../validation/raw/2026-10-02-s4000/week1/05_error_check.log) |
| `06_async_kernel` | PASS | 异步提交与同步等待均完成 | [日志](../validation/raw/2026-10-02-s4000/week1/06_async_kernel.log) |

## 统一运行记录模板

| 日期 | 主机 / 设备 | CUDA Toolkit 或 MUSA SDK | 编译器 | 架构 | backend | target | 输入规模 | 正确性 | 耗时 / 带宽 | 是否真实运行 |
|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | host；GPU 型号 | CUDA x.y / MUSA x.y；或未安装 | nvcc / mcc；版本 | sm_XX / mp_XX | cuda / musa | `chapterXX__name` | N、grid/block 等 | PASS/FAIL/未验证 | 实测值；无实测填“未运行” | 是 / 未运行 |

运行命令：

```text
make BACKEND=<cuda|musa> [CUDA_ARCH=sm_XX|MUSA_ARCH=mp_XX] TARGET=<target>
./build/<target>
```
