# Week 1 · 入门示例

对应官方指南：Ch1 GPU 并行计算 / Ch2 硬件架构 / Ch3 软件架构 / Ch4 环境搭建

> 环境配置见 [`../../docs/setup.md`](../../docs/setup.md)
> 代码按 MUSA 通用 API 命名写。如本地 SDK 版本有差异，按编译错误信息微调即可。

## 示例

| 文件 | 内容 |
|---|---|
| `01_hello_world.mu` | 最小 kernel：CPU 与 GPU 各打一句 hello |
| `02_thread_index.mu` | 单 block / 多 block 下 `threadIdx`、`blockIdx`、`blockDim` 的关系 |
| `03_device_info.mu` | 用 Runtime API 查设备名 / SM 数 / 显存 / warp 大小 |
| `04_memory_basics.mu` | 显存四件套：`musaMalloc` / `musaMemset` / `musaMemcpy` / `musaFree` |
| `05_error_check.mu` | 同步 vs 异步错误：`musaGetLastError` 与 `musaDeviceSynchronize` 的分工 |
| `06_async_kernel.mu` | 用 wall clock 直观观察 kernel 启动是异步的 |

> 1–3 建立"线程模型 + 硬件视角"，4–6 建立"显存 + 错误 + 异步"三大基础肌肉记忆。
> 跑完这 6 个示例就有了进入 Week 2 写 vectorAdd / 多 stream / Graph 的全部前置知识。

## 编译运行

### 方法 A：Makefile（快速、单目录）

```bash
cd code/week1
make             # 编译全部
./01_hello_world
./02_thread_index
./03_device_info
./04_memory_basics
./05_error_check
./06_async_kernel

make clean       # 清理
```

环境变量可覆盖：`MCC=/path/to/mcc MUSA_PATH=/usr/local/musa-3.1.0 make`。

### 方法 B：CMake（整库统一构建，IDE 友好）

从项目 `code/` 目录：

```bash
cd code
cmake -B build -DMUSA_PATH=/usr/local/musa
cmake --build build -j
./build/week1/01_hello_world
```

只编单个目标：

```bash
cmake --build build --target 01_hello_world
```

清理：`rm -rf build`。

> CMake 会生成 `compile_commands.json`，CLion / VS Code clangd 可用来索引。

### 方法 C：本地编辑 + 远程跑（Mac 用户）

Mac 上没 mcc，本地编辑、AutoDL 远程编译运行：

```bash
./scripts/musa.sh run 01_hello_world
```

完整说明见 [`../../docs/remote-dev.md`](../../docs/remote-dev.md)。

## 习题（Exercises）

见 [`exercises.md`](exercises.md)。本周共 10 题，从修改启动配置、到主动触发各类错误、到测 launch overhead。
