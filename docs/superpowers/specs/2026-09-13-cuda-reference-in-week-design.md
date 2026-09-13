# Week 内 CUDA 参考示例整合设计

## 背景

仓库当前以 `code/week1`～`code/week6` 的 MUSA 示例为主线，已有 README、学习课文、习题和 Quiz。用户希望把 [wrox-pro-cuda-c](https://github.com/kriegalex/wrox-pro-cuda-c) 中《Professional CUDA C Programming》的高价值示例纳入每周学习路径，作为 CUDA 原始实现与 MUSA 兼容性的对照材料。

上游仓库按 `chapter01`～`chapter10`、`common` 和 `lectures_code` 组织，示例主要是 `.cu`，并带有每章 Makefile。该仓库最后一次版本说明在 2015 年，部分示例使用旧的 CUDA 架构参数，因此不能直接作为现代 CUDA 或 MUSA 的统一构建目录。

## 目标

- 将精选 CUDA 示例放到对应的 `code/weekN/cuda-reference/`，使参考代码与学习主题就近组织。
- 保留 `.cu` 源码和 CUDA API 语义，不强制迁移为 `.mu`。
- 提供 `BACKEND=cuda` 与 `BACKEND=musa` 两种编译入口。
- 在每个 Week 的 README、学习笔记和 exercises 中建立清晰入口，说明该示例补充了哪一个 MUSA 主线概念。
- 对 CUDA/MUSA 编译差异、库依赖、架构参数、warp 语义和未验证状态做显式记录。
- 同步增加 `docs/index.html` Quiz，覆盖新引入的关键概念与常见错误。

## 非目标

- 不完整镜像上游 10 章代码。
- 不把全部 CUDA 示例重写成 `.mu`。
- 不承诺所有 CUDA 专有库、OpenACC、OpenMP 或旧架构示例都能在 MUSA 上运行。
- 不伪造 MUSA 或 CUDA 的性能数字；没有硬件环境时只做静态检查和编译配置检查。
- 不改变现有 `code/weekN/` 主线示例的编号、语义和已记录实验结果。

## 目录结构

```text
code/weekN/
  cuda-reference/
    README.md
    common/
      common.h
    chapterXX/
      *.cu
      Makefile
```

实际只创建与该周主题相关的精选文件；不复制未纳入学习路径的上游文件。每个 `cuda-reference/README.md` 说明来源章节、选入理由、文件清单、运行命令、后端验证状态和已知限制。

## Week 映射与精选范围

### Week 1：编程模型与基础索引

从上游 `chapter01`、`chapter02` 选择：

- `hello.cu`：host/device 与最小 kernel。
- `checkDeviceInfor.cu`：设备属性查询。
- `checkDimension.cu`、`checkThreadIndex.cu`、`defineGridBlock.cu`：grid/block/thread 关系。
- `sumArraysOnGPU-small-case.cu`、`sumArraysOnGPU-timer.cu`：从最小 vector 加法进入完整数据流与计时。
- `sumMatrixOnGPU-1D-grid-1D-block.cu`、`sumMatrixOnGPU-2D-grid-1D-block.cu`、`sumMatrixOnGPU-2D-grid-2D-block.cu`：二维索引布局对照。

与现有 `code/week1` 的基础 kernel、内存和索引示例互补；矩阵求和与 Week 3 现有示例有重复时，以“CUDA 原始实现对照”标记，不新增第二套主线练习。

### Week 2：内存传输与异步执行

从上游 `chapter04`、`chapter06` 选择：

- `memTransfer.cu`、`pinMemTransfer.cu`、`sumArrayZerocpy.cu`：pageable、pinned 和 zero-copy。
- `sumMatrixGPUManaged.cu`、`sumMatrixGPUManual.cu`：managed memory 与手动显式传输。
- `asyncAPI.cu`、`simpleCallback.cu`：异步 API 与 host callback。
- `simpleHyperqBreadth.cu`、`simpleHyperqDependence.cu`：并发任务与依赖关系。

与现有 vectorAdd、stream、event、graph 示例形成 CUDA API 侧的同主题对照。Hyper-Q 相关名称和结论保留为 CUDA 语境，MUSA 后端若不支持则只保留源码阅读或标记未验证。

### Week 3：执行模型、分支、归约与动态并行

从上游 `chapter03`、`chapter05` 选择：

- `simpleDivergence.cu`：warp 分支发散。
- `reduceInteger.cu`、`reduceIntegerShfl.cu`：归约基线与 shuffle。
- `simpleShfl.cu`：shuffle 原语最小示例。
- `nestedHelloWorld.cu`、`nestedReduce.cu`：动态并行和设备端启动。
- `sumMatrix.cu`：矩阵求和综合示例。

重点补充 CUDA 的 32-thread warp 假设与 MUSA 设备实际 warp 语义之间的差异，不直接把 CUDA mask 或性能结论移植为 MUSA 结论。

### Week 4：全局访存、布局与转置

从上游 `chapter04` 选择：

- `readSegment.cu`、`writeSegment.cu`：读取与写入的访问段。
- `readSegmentUnroll.cu`：循环展开与访存吞吐。
- `simpleMathAoS.cu`、`simpleMathSoA.cu`：AoS/SoA 布局。
- `transpose.cu`：朴素转置与访存模式。
- `globalVariable.cu`：全局变量访问的补充案例。

这些示例与现有 offset、SAXPY、AoS/SoA、transpose 主线对应，主要作为 CUDA 参考实现和实验对照，不重复新增同名 MUSA 练习。

### Week 5：片上存储、原子与数值行为

从上游 `chapter05`、`chapter07` 选择：

- `checkSmemSquare.cu`、`checkSmemRectangle.cu`：shared memory 形状与索引。
- `constantReadOnly.cu`、`constantStencil.cu`：constant memory 与 stencil。
- `reduceInteger.cu`、`reduceIntegerShfl.cu`：shared/shuffle 归约对照。
- `my-atomic-add.cu`、`atomic-ordering.cu`：原子操作和顺序。
- `floating-point-accuracy.cu`、`floating-point-perf.cu`、`fmad.cu`：浮点精度与编译优化影响。

保留现有 naive/tiled GEMM 与 muBLAS 主线；上游库调用和 n-body 大案例不作为首批迁移内容。

### Week 6：库、多 GPU、调试与综合案例

从上游 `chapter08`、`chapter09`、`chapter10` 选择：

- `cublas.cu`、`cusparse.cu`、`cufft.cu`：CUDA 数学库入口，MUSA 后端标为库映射候选。
- `simpleMultiGPU.cu`、`simpleP2P.c`、`simpleP2P_PingPong.cu`：多 GPU 与 P2P。
- `debug-hazards.cu`、`debug-segfault.cu`、`debug-segfault.fixed.cu`：错误定位与修复对照。
- `sumMatrixGPU.cu`：综合性能/正确性案例。
- `crypt.parallelized.cu`、`crypt.overlap.cu`：把并行化与重叠组织成完整案例，默认仅作为进阶阅读。

OpenACC、OpenMP、CUDA-aware MPI 和专有 CUDA 库依赖不进入首批默认构建目标，但可以在 README 的“未纳入/后续候选”中列出。

## 双后端编译设计

每个 Week 的 `cuda-reference/Makefile` 共享一致的接口：

```bash
make BACKEND=cuda
make BACKEND=musa
make BACKEND=cuda TARGET=chapter02__checkThreadIndex
make clean
```

### CUDA 后端

- 编译器默认为 `nvcc`，可通过 `NVCC` 覆盖。
- 不固定上游过时的 `sm_20`；允许通过 `CUDA_ARCH` 指定当前设备架构。
- 仅对能在现代 CUDA Toolkit 编译的目标进入默认列表。

### MUSA 后端

- 优先使用 MUSA SDK 的 `mcc` 或 `mcc_wrapper` / MUSA Mapping，让 `.cu` 保持不变。
- 通过 `MUSA_HOME`、`MCC`、`MUSA_ARCH` 支持不同 SDK 安装路径和设备架构。
- 根据 SDK 版本选择 `.cu` 的识别选项（如 `-mtgpu`）或 Mapping 的 `-x musa` / 插件配置。
- 不能自动映射的 CUDA 库、头文件或 API，在目标级别明确失败原因，不用静默替换成不等价实现。

编译层只负责后端选择；学习文档负责解释 CUDA/MUSA 的语义差异。运行结果必须在 `notes/weekN.md` 中记录硬件、SDK/Toolkit 版本、编译器、架构和是否真实执行。

## 文档同步

每个 Week 同步修改：

- `code/weekN/cuda-reference/README.md`：精选清单、构建命令、来源和兼容性矩阵。
- `code/weekN/README.md`：新增“CUDA reference”入口。
- `code/weekN/learning-notes.md`：在对应主题后增加阅读顺序和 CUDA/MUSA 对照要点。
- `code/weekN/exercises.md`：增加至少一题编译后端或语义差异实验。
- `docs/index.html`：增加覆盖新知识点、易错点和来源路径的 Quiz 题。
- `docs/cuda-example-map.md`：登记上游章节到 Week/文件的映射，避免后续重复搬运。

## 验收标准

- 所有新增 `.cu` 文件都位于对应的 `code/weekN/cuda-reference/`，没有散落到主线目录。
- 每个新增文件在本周 `cuda-reference/README.md` 中有唯一入口、来源章节和状态。
- CUDA 后端和 MUSA 后端都有明确命令；不能编译的目标必须显示原因并标为 CUDA-only 或未验证。
- MUSA 编译不使用伪造的成功结果；无 MUSA SDK 的环境只报告静态检查。
- `code/weekN/README.md`、`learning-notes.md`、`exercises.md`、`docs/cuda-example-map.md` 和 Quiz 入口一致。
- 通过 `node scripts/check-docs.js`、`git diff --check` 和针对 Makefile 的静态检查。
- 不覆盖用户现有修改，不改变已有 Week 主线示例和实测记录。

## 来源与兼容性说明

- 上游样例仓库：[kriegalex/wrox-pro-cuda-c](https://github.com/kriegalex/wrox-pro-cuda-c)。
- MUSA SDK 的 MCC 文档说明 `mcc` 接受 `.cu`，但需要显式把它识别为 MUSA 源码；具体参数随 SDK 版本变化。
- MUSA Mapping 文档说明可以在编译期映射 CUDA 头文件、API 和宏，并提供 `mcc_wrapper` 兼容部分 nvcc/CMake 调用方式。
- 因此本设计采用“源码保留 `.cu` + 后端构建开关”，但不宣称 CUDA 专有生态和硬件语义完全兼容。
