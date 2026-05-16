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

## 习题 与 文章

`exercises.md` 和 `docs/articles/02-stream-graph.md` **本轮没写**，等在 AutoDL 上跑通 8 个示例、拿到实测数据后补：

- 各示例 PART III Q&A 里凡是涉及 ms 数字的论断都标了 `// TODO: AutoDL 跑通后回填实测数字`
- 跑完后预期产物：`exercises.md`（10 道左右）+ 公众号文章《Stream 与 Graph 把吞吐量挤到极限》

> 实测时建议顺序：01 → 03 → 02 → 04 → 05 → 06 → 07 → 08。`01_vector_add_runtime` 先验证整条链路通；`03_vector_add_timer` 先把计时方法学固定下来，后面所有 µs 级实测才有意义。
