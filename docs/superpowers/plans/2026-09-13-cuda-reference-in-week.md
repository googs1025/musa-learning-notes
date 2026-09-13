# Week 内 CUDA Reference 整合实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `wrox-pro-cuda-c` 中精选的 `.cu` 示例按主题纳入 `code/week1`～`code/week6`，并提供 CUDA/MUSA 双后端构建与完整学习入口。

**Architecture:** 每周新增独立的 `cuda-reference/`，保留上游 CUDA 风格 `.cu`，由每周 Makefile 通过 `BACKEND=cuda|musa` 选择 `nvcc` 或 `mcc`/MUSA Mapping。主线 `.mu` 不改编号；README、学习笔记、习题、CUDA 映射表和 Quiz 只增加引用与差异说明。

**Tech Stack:** CUDA `.cu`、`nvcc`、MUSA `mcc`/`mcc_wrapper`/MUSA Mapping、Makefile、Markdown、Node.js 静态文档检查。

---

## 文件边界

### 新增代码与构建文件

- `code/week1/cuda-reference/README.md`、`Makefile`、精选 `chapter01/` 与 `chapter02/` `.cu`。
- `code/week2/cuda-reference/README.md`、`Makefile`、精选 `chapter04/` 与 `chapter06/` `.cu`。
- `code/week3/cuda-reference/README.md`、`Makefile`、精选 `chapter03/` 与 `chapter05/` `.cu`。
- `code/week4/cuda-reference/README.md`、`Makefile`、精选 `chapter04/` `.cu`。
- `code/week5/cuda-reference/README.md`、`Makefile`、精选 `chapter05/` 与 `chapter07/` `.cu`。
- `code/week6/cuda-reference/README.md`、`Makefile`、精选 `chapter08/`、`chapter09/`、`chapter10/` `.cu`/`.c`。

每周只复制计划中明确列出的文件和它们实际需要的最小 `common/common.h`；不复制上游 Makefile 中与未纳入示例绑定的目标。

### 修改文档

- `code/weekN/README.md`：增加 CUDA reference 入口和后端命令。
- `code/weekN/learning-notes.md`：在相应主题加入阅读顺序、源码对照和兼容性警告。
- `code/weekN/exercises.md`：每周增加至少一道后端选择或 CUDA/MUSA 差异实验。
- `docs/cuda-example-map.md`：登记上游章节、Week、文件、状态和对应主线示例。
- `docs/index.html`：加入新主题 Quiz，题目含 `answer`、`pitfall`、`source`。
- `notes/weekN.md`：仅在实际运行后记录机器、Toolkit/SDK、架构和结果，不预填性能数字。

## Task 1: 建立上游快照与公共编译约定

**Files:**

- Create: `code/week1/cuda-reference/Makefile`
- Create: `code/week1/cuda-reference/README.md`
- Create: `code/week1/cuda-reference/common/common.h`
- Create: `code/week2/cuda-reference/Makefile`
- Create: `code/week3/cuda-reference/Makefile`
- Create: `code/week4/cuda-reference/Makefile`
- Create: `code/week5/cuda-reference/Makefile`
- Create: `code/week6/cuda-reference/Makefile`

- [ ] **Step 1: 固定上游版本与文件来源**

  使用 `gh api 'repos/kriegalex/wrox-pro-cuda-c/git/refs/heads/master'` 记录当前 commit SHA；对每个选入文件使用 GitHub raw URL 或 API 下载，保留文件头部版权信息，确认上游许可证为 MIT。

- [ ] **Step 2: 定义 Makefile 后端变量**

  每周 Makefile 都提供同一组变量：`BACKEND ?= cuda`、`NVCC ?= nvcc`、`MCC ?= mcc`、`MUSA_HOME ?= /usr/local/musa`、`CUDA_ARCH ?=`、`MUSA_ARCH ?=`、`TARGET ?= all`。未知 `BACKEND` 必须以非零状态退出；`make BACKEND=cuda TARGET=<name>` 只构建指定目标。

- [ ] **Step 3: 固定 CUDA 编译规则**

  CUDA 规则使用 `$(NVCC) -O2 -lineinfo`，只有 `CUDA_ARCH` 非空时才追加 `-arch=$(CUDA_ARCH)`；不写死上游的 `sm_20`。

- [ ] **Step 4: 固定 MUSA 编译规则**

  MUSA 规则优先调用 `$(MCC)`，传入 `--musa-path=$(MUSA_HOME)`、`-mtgpu`（适用于 `.cu` 输入）和用户指定的 `MUSA_ARCH`；如果当前 SDK 需要 Mapping，则在 README 给出 `mcc_wrapper`/`MUSA Mapping` 命令，不把未验证的参数伪装成通用成功路径。

- [ ] **Step 5: 做 Makefile 静态失败测试**

  在无 CUDA/MUSA SDK 的 Mac 环境执行 `make -n BACKEND=cuda`、`make -n BACKEND=musa` 和 `make BACKEND=invalid`。前两者应打印完整命令，后者应失败且说明可选后端；不得要求本地实际链接 GPU 库。

## Task 2: Week 1–2 基础、内存与异步参考

**Files:**

- Create: `code/week1/cuda-reference/chapter01/hello.cu`
- Create: `code/week1/cuda-reference/chapter02/checkDeviceInfor.cu`
- Create: `code/week1/cuda-reference/chapter02/checkDimension.cu`
- Create: `code/week1/cuda-reference/chapter02/checkThreadIndex.cu`
- Create: `code/week1/cuda-reference/chapter02/defineGridBlock.cu`
- Create: `code/week1/cuda-reference/chapter02/sumArraysOnGPU-small-case.cu`
- Create: `code/week1/cuda-reference/chapter02/sumArraysOnGPU-timer.cu`
- Create: `code/week1/cuda-reference/chapter02/sumMatrixOnGPU-1D-grid-1D-block.cu`
- Create: `code/week1/cuda-reference/chapter02/sumMatrixOnGPU-2D-grid-1D-block.cu`
- Create: `code/week1/cuda-reference/chapter02/sumMatrixOnGPU-2D-grid-2D-block.cu`
- Create: `code/week2/cuda-reference/chapter04/memTransfer.cu`
- Create: `code/week2/cuda-reference/chapter04/pinMemTransfer.cu`
- Create: `code/week2/cuda-reference/chapter04/sumArrayZerocpy.cu`
- Create: `code/week2/cuda-reference/chapter04/sumMatrixGPUManaged.cu`
- Create: `code/week2/cuda-reference/chapter04/sumMatrixGPUManual.cu`
- Create: `code/week2/cuda-reference/chapter06/asyncAPI.cu`
- Create: `code/week2/cuda-reference/chapter06/simpleCallback.cu`
- Create: `code/week2/cuda-reference/chapter06/simpleHyperqBreadth.cu`
- Create: `code/week2/cuda-reference/chapter06/simpleHyperqDependence.cu`
- Modify: `code/week1/cuda-reference/README.md`
- Create: `code/week2/cuda-reference/README.md`
- Modify: `code/week1/README.md`, `code/week1/learning-notes.md`, `code/week1/exercises.md`
- Modify: `code/week2/README.md`, `code/week2/learning-notes.md`, `code/week2/exercises.md`

- [ ] **Step 1: 复制并核对 Week 1 源码**

  对每个文件执行 `rg -n '__global__|threadIdx|blockIdx|blockDim|cudaMalloc|cudaMemcpy|cudaDeviceSynchronize'`，记录输入输出、边界判断和 CUDA API；只修复现代编译器显然阻止构建的过时参数或路径，不改变示例算法。

- [ ] **Step 2: 配置 Week 1 目标清单**

  将文件 basename 映射为 Make target；README 表格逐项列出“上游章节 / 当前 Week 对应 / CUDA 状态 / MUSA 状态”。矩阵求和标记为与 Week 3 现有示例的对照材料。

- [ ] **Step 3: 复制并核对 Week 2 源码**

  检查 pinned、zero-copy、managed memory、异步调用和 callback 的同步边界；Hyper-Q 示例的硬件前提写入 README，不把 CUDA 的并发结论推广为 MUSA 行为。

- [ ] **Step 4: 增加 Week 1–2 文档入口和实验题**

  每周 README 增加构建命令；学习笔记说明 CUDA `.cu` 与现有 `.mu` 的关系；每周 exercises 至少增加一道“同一源码切换 `BACKEND`，比较头文件/API/运行结果”的题目，要求记录 SDK 和设备信息。

- [ ] **Step 5: 做 Week 1–2 静态检查**

  执行 `make -n BACKEND=cuda`、`make -n BACKEND=musa`、`git diff --check`，并确认 README 中每个新增 `.cu` 文件都有唯一链接。

## Task 3: Week 3–4 执行模型与访存参考

**Files:**

- Create: `code/week3/cuda-reference/chapter03/simpleDivergence.cu`
- Create: `code/week3/cuda-reference/chapter03/reduceInteger.cu`
- Create: `code/week3/cuda-reference/chapter03/reduceIntegerShfl.cu`
- Create: `code/week3/cuda-reference/chapter03/simpleShfl.cu`
- Create: `code/week3/cuda-reference/chapter03/nestedHelloWorld.cu`
- Create: `code/week3/cuda-reference/chapter03/nestedReduce.cu`
- Create: `code/week3/cuda-reference/chapter03/sumMatrix.cu`
- Create: `code/week3/cuda-reference/chapter05/checkSmemSquare.cu`
- Create: `code/week3/cuda-reference/chapter05/checkSmemRectangle.cu`
- Create: `code/week4/cuda-reference/chapter04/readSegment.cu`
- Create: `code/week4/cuda-reference/chapter04/writeSegment.cu`
- Create: `code/week4/cuda-reference/chapter04/readSegmentUnroll.cu`
- Create: `code/week4/cuda-reference/chapter04/simpleMathAoS.cu`
- Create: `code/week4/cuda-reference/chapter04/simpleMathSoA.cu`
- Create: `code/week4/cuda-reference/chapter04/transpose.cu`
- Create: `code/week4/cuda-reference/chapter04/globalVariable.cu`
- Modify: `code/week3/cuda-reference/README.md`, `code/week4/cuda-reference/README.md`
- Modify: `code/week3/README.md`, `code/week3/learning-notes.md`, `code/week3/exercises.md`
- Modify: `code/week4/README.md`, `code/week4/learning-notes.md`, `code/week4/exercises.md`

- [ ] **Step 1: 核对执行模型源码**

  用 `rg -n '__global__|__shared__|__syncthreads|shuffle|warp|threadIdx|blockIdx|<<<'` 核对 divergence、归约、shuffle 和 nested launch；学习笔记同时写明 CUDA 32-thread warp 假设与 MUSA 实际设备参数的差异。

- [ ] **Step 2: 核对访存源码**

  用 `rg -n 'offset|stride|struct|shared|transpose|atomic|GpuTimer|GB/s'` 提取访问方向、stride、AoS/SoA 和转置的实验变量；性能说明必须区分带宽公式与实测数字。

- [ ] **Step 3: 完成 Week 3–4 README、课文与习题**

  每个示例说明它与现有 `.mu` 文件的对应关系、输入输出和最常见错误；CUDA-only 或 MUSA 未验证目标显示在目标表中而不是默认为可运行。

- [ ] **Step 4: 做 Week 3–4 静态检查**

  执行每周两个后端的 `make -n`、`git diff --check`，并用 `rg -n` 确认所有新增源文件名在对应 README 和学习笔记中出现。

## Task 4: Week 5 数值、Shared、Constant 与 Atomic 参考

**Files:**

- Create: `code/week5/cuda-reference/chapter05/checkSmemSquare.cu`
- Create: `code/week5/cuda-reference/chapter05/checkSmemRectangle.cu`
- Create: `code/week5/cuda-reference/chapter05/constantReadOnly.cu`
- Create: `code/week5/cuda-reference/chapter05/constantStencil.cu`
- Create: `code/week5/cuda-reference/chapter05/reduceInteger.cu`
- Create: `code/week5/cuda-reference/chapter05/reduceIntegerShfl.cu`
- Create: `code/week5/cuda-reference/chapter07/my-atomic-add.cu`
- Create: `code/week5/cuda-reference/chapter07/atomic-ordering.cu`
- Create: `code/week5/cuda-reference/chapter07/floating-point-accuracy.cu`
- Create: `code/week5/cuda-reference/chapter07/floating-point-perf.cu`
- Create: `code/week5/cuda-reference/chapter07/fmad.cu`
- Create: `code/week5/cuda-reference/README.md`
- Modify: `code/week5/README.md`, `code/week5/learning-notes.md`, `code/week5/exercises.md`

- [ ] **Step 1: 逐个确认 shared/constant/atomic 依赖**

  用 `rg -n '__shared__|__constant__|__syncthreads|atomic|volatile|fmaf|reduce|shuffle'` 检查每个目标的编译头、同步点和设备假设；不把 CUDA 原子顺序或浮点结果误写成跨设备恒等结论。

- [ ] **Step 2: 配置默认与可选目标**

  将无需额外库的目标纳入 `make all`；将依赖特定架构、旧 intrinsic 或未确认 MUSA 支持的目标列为 `OPTIONAL_TARGETS`，并让 README 显示精确的失败原因和替代阅读路径。

- [ ] **Step 3: 完成 Week 5 教材同步**

  把 reference 排在现有 shared → GEMM 主线对应位置，新增“功能正确性、数值稳定性、性能”三者的区分实验；题目要求改变 block size 或编译优化参数后重新验证误差与耗时。

- [ ] **Step 4: 做 Week 5 静态检查**

  执行 `make -n BACKEND=cuda`、`make -n BACKEND=musa`、`git diff --check`，并检查不存在未解释的 CUDA-only 目标。

## Task 5: Week 6 多 GPU、库与调试参考

**Files:**

- Create: `code/week6/cuda-reference/chapter08/cublas.cu`
- Create: `code/week6/cuda-reference/chapter08/cusparse.cu`
- Create: `code/week6/cuda-reference/chapter08/cufft.cu`
- Create: `code/week6/cuda-reference/chapter09/simpleMultiGPU.cu`
- Create: `code/week6/cuda-reference/chapter09/simpleP2P.c`
- Create: `code/week6/cuda-reference/chapter09/simpleP2P_PingPong.cu`
- Create: `code/week6/cuda-reference/chapter10/debug-hazards.cu`
- Create: `code/week6/cuda-reference/chapter10/debug-segfault.cu`
- Create: `code/week6/cuda-reference/chapter10/debug-segfault.fixed.cu`
- Create: `code/week6/cuda-reference/chapter10/sumMatrixGPU.cu`
- Create: `code/week6/cuda-reference/chapter10/crypt.parallelized.cu`
- Create: `code/week6/cuda-reference/chapter10/crypt.overlap.cu`
- Create: `code/week6/cuda-reference/README.md`
- Modify: `code/week6/README.md`, `code/week6/learning-notes.md`, `code/week6/exercises.md`

- [ ] **Step 1: 将库依赖分级**

  `cublas`、`cusparse`、`cufft` 默认标为 CUDA library targets；只有确认 MUSA Mapping 与对应 mu* 库均可用时才加入 MUSA 默认目标。README 必须列出所需库和替代的 MUSA 库名。

- [ ] **Step 2: 核对多 GPU 与调试边界**

  记录 device count、peer access、P2P buffer 生命周期、错误复现输入和 fixed 版本的差异；把调试示例放在可控的 `debug` 目标中，避免 `make all` 默认运行故意出错的程序。

- [ ] **Step 3: 完成 Week 6 教材同步**

  在现有 MCCL、MUSA GDB、error dump、torch_musa 主线旁加入 CUDA 对照链接；明确哪些示例只用于阅读，哪些可作为跨后端实验。

- [ ] **Step 4: 做 Week 6 静态检查**

  执行 `make -n`、检查库参数、`git diff --check`，并确保 destructive/debug 示例没有被列入默认执行脚本。

## Task 6: 更新全局映射、Quiz 与运行记录模板

**Files:**

- Modify: `docs/cuda-example-map.md`
- Modify: `docs/index.html`
- Modify: `notes/week1.md`、`notes/week2.md`、`notes/week3.md`、`notes/week4.md`、`notes/week5.md`、`notes/week6.md`

- [ ] **Step 1: 完成 CUDA → Week 映射表**

  添加列：上游 chapter、文件、Week、现有对应示例、CUDA 状态、MUSA 状态、验证命令；明确重复案例只保留一条主线入口。

- [ ] **Step 2: 增加 Quiz 题目**

  每个 Week 至少增加两道题，覆盖 `.cu` 后端切换、MUSA `.cu` 识别选项、pinned/managed memory、warp/shuffle、访存布局、shared/atomic/浮点、库映射、多 GPU/P2P 或调试危险点；每题补齐 `answer`、`pitfall`、`source`，source 指向对应新增文件。

- [ ] **Step 3: 补充运行记录模板**

  在每周 notes 增加统一模板：日期、主机/设备、CUDA Toolkit 或 MUSA SDK、编译器命令、架构、目标、输入规模、正确性结果、耗时/带宽、是否真实运行。没有实测数据的字段保留“未运行”，不填估计值。

## Task 7: 全量验证与交付

**Files:**

- Test: `code/week1/cuda-reference/Makefile` 至 `code/week6/cuda-reference/Makefile`
- Test: 所有新增 README、学习笔记、习题、映射表和 Quiz

- [ ] **Step 1: 运行文档校验**

  执行 `node scripts/check-docs.js`；若失败，按输出修复 Quiz source、题目结构或文档入口，直到命令返回 0。

- [ ] **Step 2: 运行 Markdown 和覆盖检查**

  执行 `git diff --check`；用 `rg --files code/week{1..6}/cuda-reference | rg '\.(cu|c)$'` 收集新增源文件，再逐周检查每个 basename 同时出现在 reference README 和 `learning-notes.md` 中。

- [ ] **Step 3: 运行无 SDK 的构建演练**

  在当前 Mac 环境执行六周 Makefile 的 `make -n BACKEND=cuda`、`make -n BACKEND=musa` 和 `make BACKEND=invalid`；记录“只完成命令静态检查，未完成 GPU 编译”的事实。

- [ ] **Step 4: 在可用 Linux GPU 环境执行真实验证**

  CUDA 环境至少编译运行每周一个基础目标；MUSA 环境至少编译运行每周一个无专有库目标。将完整命令和结果写入对应 `notes/weekN.md`，库、P2P、故意错误示例单独记录，不把未运行目标标为通过。

- [ ] **Step 5: 检查范围和工作树**

  执行 `git status --short`、`git diff --stat`，确认没有复制上游未选文件、没有覆盖已有实测记录、没有修改无关目录；由于当前环境不能写入 `.git/index.lock`，不执行 commit，交付时说明这一限制。
