# 每周示例课文补充设计

## 背景

当前仓库的 `code/weekN/learning-notes.md` 已经包含学习顺序、核心知识点和常见误区，但各周的详细程度不一致：Week 1 和 Week 2 已有较完整的概念讲解，Week 3–6 主要是提纲式内容。学习者需要在源码、README 和笔记之间来回切换，才能理解每个示例到底解决什么问题、代码如何运行以及它与前后示例的关系。

本次工作把每周学习材料补充成可以直接配合源码阅读的课文，同时保留已有内容和已有实测记录。

## 目标与范围

### 目标

- 为每个 Week 1–6 的可运行示例增加一个独立课文小节。
- 每个小节都能说明示例的目的、代码结构、执行流程、关键知识点和常见错误。
- 让 `README.md` 的示例列表、`learning-notes.md` 的课文和实际源码文件一一对应。
- 保留现有学习顺序、概念总结、实测数据和 CUDA/MUSA 对照内容。

### 覆盖文件

- Week 1：6 个 `.mu` 示例。
- Week 2：8 个 `.mu` 示例。
- Week 3：6 个 `.mu` 示例。
- Week 4：6 个 `.mu` 示例。
- Week 5：7 个 `.mu` 示例。
- Week 6：`02_musa_gdb_demo.mu`、`03_error_dump.mu`、`01_mccl_allreduce.cpp`、`04_torch_musa_minimal.py`、`05_torch_musa_custom_op.cpp`。

Week 6 虽然不全是 `.mu`，但这些文件共同构成“多卡、调试、框架和 custom op”课程主线，因此一并纳入。

不在本次范围内：重写源码、改变编译行为、修改实验结果、补充外部案例目录的独立教材。

## 文档结构

每个 `learning-notes.md` 保留已有章节，并在核心知识点之后增加“逐示例课文”章节。每个示例使用以下结构，内容按示例复杂度伸缩：

```markdown
## 逐示例课文

### 1. `path/to/example`

#### 这个示例解决什么问题
#### 代码结构
#### 核心知识点
#### 执行流程
#### 常见错误与实验
#### 与本周其他示例的关系
```

简单示例可以合并“代码结构”和“执行流程”，但不能省略示例目标、核心知识点和至少一个可验证现象。课文解释设计意图和数据流，不逐行翻译源码，也不复制大段实现代码。

## 各周内容主线

### Week 1：从 kernel 到异步执行

按 `hello world → 线程索引 → 设备属性 → 显存管理 → 错误检查 → 异步 launch` 展开，重点解释 host/device 边界、索引计算、边界保护、同步错误与异步错误的区别。

### Week 2：从完整 vectorAdd 到异步任务图

按 `Runtime 骨架 → pinned memory → 计时方法 → unified memory → 多 stream → event 依赖 → Graph → callback` 展开，重点解释任务入队、同步点、数据依赖、流水线和计时边界。多 stream 课文明确说明 pinned host memory、chunk 不重叠以及 CPU wall-clock 与 GPU event 的适用范围。

### Week 3：从分支和 reduce 进入执行模型

按 `warp divergence → naive reduce → unrolling → shuffle → nested launch → 2D grid` 展开，重点解释 warp 内分支、block 内同步、block 间归约、MUSA warp size 差异、动态并行限制和二维索引。

### Week 4：从带宽基线到访存布局

按 `SAXPY → offset → unrolling → AoS/SoA → naive transpose → padded transpose` 展开，重点解释有效带宽、合并访存、对齐、stride、数据布局、shared tile 和 bank conflict。

### Week 5：从片上存储到 GEMM

按 `shared 基础 → shared reduce → shared transpose → constant stencil → naive GEMM → tiled GEMM → muBLAS SGEMM` 展开，重点解释 shared memory 生命周期、同步边界、constant broadcast、tile 复用、边界处理、布局和库函数基线。

### Week 6：从单卡调试到多卡和框架边界

按 `MCCL AllReduce → MUSA GDB → error dump → torch_musa 最小链路 → custom op` 展开，重点解释 rank/device/communicator 关系、错误复现、调试符号、环境记录、Python/C++/MUSA 边界和注册流程。

## 内容要求

每个示例小节至少回答以下问题：

1. 它在整周学习路径中解决什么问题？
2. 输入、输出和关键 buffer 如何流动？
3. 哪些 API、kernel 语法或同步原语是本示例的核心？
4. 哪个错误最容易发生，如何通过输出、验证或实验发现？
5. 它相比前一个示例增加了什么，相比后一个示例准备了什么？

涉及性能的示例必须区分“功能正确”和“性能提升”，并明确计时范围、同步点、数据规模和硬件依赖。涉及 MUSA 与 CUDA 差异的示例必须标明哪些结论来自源码、哪些结论依赖 SDK 或设备实测。

## 验收标准

- Week 1–6 的每个示例文件都能在对应 `learning-notes.md` 中找到唯一课文小节。
- 课文中的文件名、target 名称和 API 名称与源码一致。
- 原有章节、实测数据、链接和用户已有未提交修改不被覆盖。
- 每周课文前后顺序与对应 README 的示例顺序一致。
- Markdown 无明显断链、空章节、未完成标记或与源码矛盾的描述。
- 用 `rg` 检查所有示例是否都有课文入口，并用 Markdown 检查和 `git diff --check` 做静态验证。

## 实施策略

按 Week 1 → Week 6 分批修改，每次完成一到两周并做静态检查。先处理内容提纲最薄弱的 Week 3–6，再统一检查 Week 1–2 的新增章节是否与已有课文重复或矛盾。只修改对应的 `learning-notes.md`，不顺手重构源码或 README。
