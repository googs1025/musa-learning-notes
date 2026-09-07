# MUSA Learning Notes

这是一个以“读官方文档 → 写最小示例 → 在 GPU 上验证 → 记录问题”为主线的 MUSA 学习仓库。

如果你刚开始接触 GPU 编程，建议按 Week 1 → Week 6 顺序学习；如果已经有 CUDA 基础，可以直接从 CUDA/MUSA 对照和对应周次的 `learning-notes.md` 开始。

## 六周学习地图

| 周次 | 要回答的问题 | 重点知识 | 入口 | 状态 |
|---|---|---|---|---|
| Week 1 | 一个 kernel 是怎样启动和完成的？ | `grid/block/thread`、索引、Host/Device、显存、错误、异步 | [`code/week1/README.md`](code/week1/README.md) | ✅ |
| Week 2 | 多个 GPU 操作怎样排队、计时和重放？ | pinned memory、统一内存、stream、event、graph、callback | [`code/week2/README.md`](code/week2/README.md) | 🧪 |
| Week 3 | 线程如何协作完成一个归约？ | warp divergence、reduce、unroll、shuffle、2D grid、动态并行 | [`code/week3/README.md`](code/week3/README.md) | ⏳ |
| Week 4 | 为什么结果正确但带宽利用率很低？ | coalesced access、offset、AoS/SoA、transpose、bank conflict | [`code/week4/README.md`](code/week4/README.md) | ⏳ |
| Week 5 | 怎样让数据在片上重复利用？ | shared/constant memory、naive GEMM、tiled GEMM、muBLAS | [`code/week5/README.md`](code/week5/README.md) | ⏳ |
| Week 6 | 怎样定位错误并扩展到多卡和框架？ | MUSA GDB、error dump、MCCL、torch_musa、自定义算子 | [`code/week6/README.md`](code/week6/README.md) | ⏳ |

完整路线和每周文件清单见 [`docs/roadmap.md`](docs/roadmap.md)。

## 推荐学习方法

每个周次都按下面的循环走：

1. 先读该周 `README.md`，了解主题和示例顺序。
2. 再读 `learning-notes.md`，搞清楚概念、数据流和常见误区。
3. 运行最小示例，确认环境和编译链路没有问题。
4. 修改一个参数或故意制造一个错误，观察结果和错误检查位置。
5. 把真实运行结果、截图和疑问记录到 `notes/` 对应文件。

Mac 用户可以在本地用 CLion 编辑，通过远程 Linux MUSA 环境编译运行，参见 [`docs/remote-dev.md`](docs/remote-dev.md)。

## 从哪里开始

### 1. 准备环境

先看 [`docs/setup.md`](docs/setup.md)。MUSA 编译器、运行库和 GPU 驱动需要在 Linux/MUSA 环境中准备好；Mac 更适合作为编辑端。

### 2. 跑通第一个 kernel

```bash
cd code/week1
make
./01_hello_world
```

也可以使用统一 CMake：

```bash
cd code
cmake -B build -DMUSA_PATH=/usr/local/musa
cmake --build build -j
./build/week1/01_hello_world
```

### 3. 先掌握这些概念

- [`docs/concepts.md`](docs/concepts.md)：SIMT、SM、grid/block/thread、内存层次、同步。
- [`docs/musa-runtime-api.md`](docs/musa-runtime-api.md)：Runtime API 速查。
- [`docs/musa-cuda-pitfalls.md`](docs/musa-cuda-pitfalls.md)：最容易写错的索引、内存、同步和 warp 问题。
- [`docs/cuda-vs-musa.md`](docs/cuda-vs-musa.md)：CUDA 迁移到 MUSA 时哪些地方不能机械替换。
- [`notes/musa-sdk-5.2.0.md`](notes/musa-sdk-5.2.0.md)：MUSA SDK 5.2.0 官方编程指南的重点摘录。

## 各周真正要抓住的重点

### Week 1：建立执行模型

不要只记 API 名字。重点是理解：一次 kernel launch 产生一个 grid，grid 由可独立调度的 block 组成，线程通过 `blockIdx`、`blockDim`、`threadIdx` 计算全局位置；Host 和 Device 有不同的内存空间，kernel launch 通常是异步的。

### Week 2：理解“提交”不等于“完成”

重点观察 H2D → kernel → D2H 的顺序、stream 内的依赖、不同 stream 的潜在并发，以及 event/同步对计时的影响。Graph 不是天然更快，必须用实际测量验证 launch overhead 是否值得优化。

### Week 3：从“一个线程一个结果”走向协作

重点是归约的演进：global memory 基线 → 循环展开 → shared memory/warp shuffle。特别注意分支分化、warp 宽度和同步边界，不能把 CUDA 的固定假设直接套到所有 MUSA 设备。

### Week 4：把访存当成性能主线

重点不是背“合并访存”四个字，而是观察相邻线程访问的地址是否连续、offset 如何改变事务数量、AoS/SoA 如何改变布局，以及 transpose 中 shared memory bank conflict 如何出现。

### Week 5：用片上存储提高数据复用

重点是 shared memory 的加载、同步和复用；从 naive GEMM 对照到 tiled GEMM，再和 muBLAS 建立性能基线。每次优化都要同时看正确性、访存模式、寄存器和 occupancy。

### Week 6：从单卡 kernel 进入工程系统

重点是错误定位、调试器、Error Dump、多卡中的 rank/device/stream/communicator 关系，以及 torch_musa 自定义算子的边界。这里更关注“如何验证和排错”，而不只是 API 调用。

## 练习与扩展材料

- 每周习题：`code/weekN/exercises.md`
- 每周学习材料：`code/weekN/learning-notes.md`
- LeetGPU MUSA 练习：[`docs/leetgpu-easy.md`](docs/leetgpu-easy.md)
- CUDA 对照案例：[`code/cuda-freshman/`](code/cuda-freshman/)
- CUDA Kernel / 面试复习参考：[`alexngng/CUDA-Learn-Note`](https://github.com/alexngng/CUDA-Learn-Note)，适合补充阅读 SGEMM、SGEMV、warp/block reduce、dot product、elementwise、softmax、LayerNorm/RMSNorm、histogram 等 kernel。
- GPU 架构与库调用案例：[`code/gpu-architecture-practice/`](code/gpu-architecture-practice/)
- 学习记录和实测结果：[`notes/`](notes/)
- 在线自测题库：[`docs/index.html`](docs/index.html)

## 仓库结构

```text
musa-learning-notes/
├── README.md                         # 学习者入口：路线、重点、入口
├── Agent.md                          # 代理/协作者工作约束
├── code/week1..week6/                # 主线示例、教材和习题
├── code/cuda-freshman/               # CUDA 对照材料
├── code/gpu-architecture-practice/   # 外部 GPU/MUSA 案例集
├── code/leetgpu/easy/                # LeetGPU MUSA 练习
├── docs/                             # 概念、路线、API、环境和文章
└── notes/                            # 实测结果、故障记录和官方文档摘录
```

## 官方资料

- [MUSA SDK v5.2.0 编程指南](https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/)
- [MUSA SDK 安装指南](https://docs.mthreads.com/musa-sdk/version-5.2.0/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA_Freshman](https://github.com/Tony-Tan/CUDA_Freshman)
- [CUDA-Learn-Note](https://github.com/alexngng/CUDA-Learn-Note)：CUDA kernel 和高频面试题复习参考

## 原则

1. 官方指南是事实基准；仓库笔记与官方内容冲突时，以当前 SDK 文档和实际测试为准。
2. MUSA 和 CUDA 的 API 很接近，但 warp 宽度、架构目标、工具链和性能行为不能默认相同。
3. 先跑通最小程序，再做性能优化；性能数字必须来自实际运行记录。

## License

MIT，代码和笔记随便用。
