# Week 2 · 编程模型 + Stream + Graph

对应官方指南：Ch5 编程模型与 Stream

> Week 1 已经把"线程模型 / 显存 / 错误 / 异步"打地基。
> Week 2 把这些组合起来：先写第一个完整 vectorAdd，再升级到 pinned、统一内存、多流流水线、跨流依赖、Graph、Callback。
> 完成这周后，你应该能独立写出"H2D / Kernel / D2H 三件事重叠"的 GPU 程序。

## 示例

| 文件 | 内容 |
|---|---|
| `01_vector_add_runtime.mu` | Runtime API 的"7 步骨架"，第一个完整 vectorAdd |
| `02_vector_add_pinned.mu` | `musaMallocHost` vs `malloc`，H2D 加速比 |
| `03_vector_add_timer.mu` | `CpuTimer` vs `GpuTimer`：怎么才算量对了 kernel 时间 |
| `04_vector_add_unified.mu` | 统一内存 + `musaMemPrefetchAsync` |
| `05_multi_stream.mu` | 多流流水线：H2D / Kernel / D2H 三阶段重叠 |
| `06_stream_event_dep.mu` | event 让跨流形成 DAG 依赖 |
| `07_musa_graph.mu` | Stream Capture → Graph，10000 次小 kernel 看 launch overhead |
| `08_stream_callback.mu` | `musaStreamAddCallback`：GPU 完成时回调 host |

每个 `.mu` 都是 **三段式注释**：PART I 知识点 / PART II 代码 / PART III Q&A。

## 编译运行

### 方法 A：Makefile（单目录快速试）

```bash
cd code/week2
make             # 编全部 8 个
./01_vector_add_runtime
./02_vector_add_pinned
./03_vector_add_timer
./04_vector_add_unified
./05_multi_stream
./06_stream_event_dep
./07_musa_graph
./08_stream_callback

make clean
```

`MCC=/path/to/mcc MUSA_PATH=/usr/local/musa-3.1.0 make` 覆盖默认路径。

### 方法 B：CMake（IDE 友好，推荐）

从 `code/` 目录：

```bash
cd code
cmake -B build -DMUSA_PATH=/usr/local/musa
cmake --build build -j
./build/week2/01_vector_add_runtime
```

只编单个 target：

```bash
cmake --build build --target 05_multi_stream
```

### 方法 C：本地编辑 + 远程跑（Mac 用户）

```bash
./scripts/musa.sh run 05_multi_stream
```

完整说明见 [`../../docs/remote-dev.md`](../../docs/remote-dev.md)。

## 为什么没有 Driver API 示例

原 roadmap 里 #02 是 "Driver API 重写 vectorAdd"，写到一半改成了 pinned memory，原因：

- MUSA Driver API (`mu*` 前缀) 在 `<musa.h>` 里齐备，跟 CUDA Driver API 一对一对应；
- 但 Driver API 要把 kernel **单独编成 `.mubin` / fatbin 模块**，再用 `muModuleLoad` 加载，不是单 `.mu` 文件能完整演示的；
- Runtime API 已经覆盖了 99% 的应用层场景，新手 / 框架使用者基本不会写 Driver API。

留待将来单独写一篇 "Runtime vs Driver" 的对比笔记，本周聚焦更高频用到的 Stream / Graph。

## 实测结果（AutoDL · 2026-05-16）

机器：AutoDL MUSA 容器（`musa-3.1.0`），N=1M（vectorAdd），UM 示例 N=4M。

| 示例 | 关键数字 | 观察 |
|---|---|---|
| 02 pinned | pageable 1.966 ms（7.95 GB/s）/ pinned 0.516 ms（30.29 GB/s）→ **×3.81** | 符合预期，PCIe DMA 收益明显 |
| 03 timer | [A] 不 sync 0.155 ms（只是 launch 入队耗时）/ [B] CPU+sync 0.119 ms / [C] GpuTimer 0.122 ms | [B][C] 一致，证明 GpuTimer 是对的；[A] 没 sync 量到的是假数字 |
| 04 unified | `musaMemPrefetchAsync` **本设备不支持**（运行时打印 info 跳过）；冷启动 1.75 ms → 热启动 1.53 ms | MUSA 3.1.0 暂未实现 prefetch，UM 退化为普通访问 |
| 05 multi-stream | 单流串行 5.94 ms / 4-stream 流水线 5.21 ms（×1.14） | 重叠度有限，可能受 copy engine 数量或调度策略限制 |
| 07 graph | direct launch 25000 次 410 ms（16.4 µs/launch）/ graph 5000 step × 5 op 677 ms（135 µs/step） | **graph 反而比 direct 慢**，跟 CUDA 直觉相反——MUSA 当前 graph 实现 launch 开销没优化到位，或单 step 5 个轻量 op 摊销不出来 |
| 08 callback | 4 个 chunk 全部回调 fire，乱序完成（0→3→2→1） | 回调由 GPU 完成时间触发，跟提交顺序无关，符合预期 |

> 待跟进：
> - graph 慢 / prefetch 缺失先记录，不急着提 issue，等 week3/4 实际用到时再回头看；
> - 重做 05 多流时跑一次 `mprof`（如果有等价工具），看 copy / kernel 是否真的并行。

## 习题 与 文章

`exercises.md` 已补（10 道，全部围绕本周实测数字展开，见 [`exercises.md`](exercises.md)）。

各 `.mu` PART III Q&A 里的 `// TODO: AutoDL 跑通后回填实测数字` 标记已全部回填（01~08 共 7 处）。

公众号文章见 [`docs/articles/02-stream-graph.md`](../../docs/articles/02-stream-graph.md) 《Stream 与 Graph 把吞吐量挤到极限》。

> 实测时建议顺序：01 → 03 → 02 → 04 → 05 → 06 → 07 → 08。`01_vector_add_runtime` 先验证整条链路通；`03_vector_add_timer` 先把计时方法学固定下来，后面所有 µs 级实测才有意义。
