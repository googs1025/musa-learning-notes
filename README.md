# MUSA Learning Notes

> 我学摩尔线程 [MUSA SDK](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/) 的学习记录。
> 不是教程,是公开的学习日志 —— 跟着官方编程指南一周一章,边学边写。

## 当前进度

| 周 | 主题 | 官方章节 | 状态 |
|---|---|---|---|
| Week 1 | Hello World / 线程索引 / 显存 / 错误处理 / 异步 | Ch1–4 | ✅ |
| Week 2 | vectorAdd · Stream · MUSA Graph | Ch5 | ⏳ |
| Week 3 | Tiled GEMM · 显存层级 | Ch5 + Ch9 | ⏳ |
| Week 4 | 性能优化(coalesced / bank conflict / occupancy) | Ch9 | ⏳ |
| Week 5 | muBLAS / muDNN / torch_musa | Ch7 + Ch10 | ⏳ |
| Week 6 | 调试 + 集群视角 | Ch6 + Ch11 | ⏳ |

完整 6 周路线见 [`docs/roadmap.md`](docs/roadmap.md)。

## Week 1 快速入门

环境配好之后(见 [`docs/setup.md`](docs/setup.md),AutoDL 摩尔线程实例最快):

**方式一 · Makefile(单周快速试)**

```bash
cd code/week1
make
./01_hello_world          # GPU printf
./02_thread_index         # threadIdx / blockIdx
./03_device_info          # 设备查询
./04_memory_basics        # 显存四件套
./05_error_check          # 错误检测两个时机
./06_async_kernel         # kernel 异步 + launch overhead
```

**方式二 · CMake(IDE 友好,推荐)**

```bash
cd code
cmake -B build -DMUSA_PATH=/usr/local/musa
cmake --build build -j
./build/week1/01_hello_world
```

CLion / VS Code 直接打开 `code/` 目录,会自动识别 CMakeLists,跳转 / 补全 / 单 target 编译都好用。

每个 `.mu` 文件都是 **三段式注释**:`PART I 知识点 / PART II 代码 / PART III Q&A`,直接当教材读就行。配套 10 道习题在 [`code/week1/exercises.md`](code/week1/exercises.md)。

学习笔记(对应公众号文章): [`docs/articles/01-first-musa-code.md`](docs/articles/01-first-musa-code.md)

## 仓库结构

```
musa-learning-notes/
├── README.md
├── LICENSE
├── code/
│   ├── CMakeLists.txt        ← 顶层 CMake,串各 week
│   ├── include/
│   │   └── musa_common.h     ← CHECK 宏 + CpuTimer + GpuTimer
│   └── week1/                ← 6 个示例 + Makefile + exercises.md
└── docs/
    ├── setup.md              ← 环境搭建
    ├── roadmap.md            ← 6 周路线(38 示例,对标 CUDA_Freshman 颗粒度)
    └── articles/             ← 学习笔记 / 公众号文章
```

> 路线图设计参考了 [Tony-Tan/CUDA_Freshman](https://github.com/Tony-Tan/CUDA_Freshman),但保留按 week 组织(学习日志的节奏)。

## 三个原则(给我自己也给读者)

1. **官方指南是 ground truth** —— 本仓库笔记若与官方矛盾以官方为准
2. **不要把 MUSA 当 CUDA 复读机** —— API 接近,但 warp = 128(MUSA) vs 32(CUDA),调优经验取决于硬件
3. **基础三章 + 编程基础先跑通** —— 这是 GPU 编程的通用能力,迁移到任何加速器都管用

## 快速参考

- [官方编程指南](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/)
- [官方安装指导](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/install_guide/)
- [AutoDL · 摩尔线程实例](https://www.autodl.com/)

## License

MIT,代码和笔记随便用。
