# 每周示例课文补充实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Week 1–6 的每个可运行示例补充与源码对应的课文，同时保留现有内容、实测记录和用户已有修改。

**Architecture:** 只修改六个已有的 `code/weekN/learning-notes.md`，不修改源码、构建系统或 README。每周保留现有章节，在核心知识点后增加按示例顺序排列的“逐示例课文”，最后做覆盖和 Markdown 静态检查。

**Tech Stack:** Markdown、C/C++/MUSA/Python 源码阅读、ripgrep、Git。

---

## 通用写作约束

每个示例新增一个小节，至少包含：示例目标、代码结构、核心 API/语法、输入输出和 buffer 流动、执行流程、一个常见错误、一个可观察实验、与前后示例的关系。性能示例注明计时范围、同步点、数据规模和硬件依赖；不复制大段源码，不修改实验结果。

### Task 1: Week 1

**Files:** Modify `code/week1/learning-notes.md`; read `01_hello_world.mu`、`02_thread_index.mu`、`03_device_info.mu`、`04_memory_basics.mu`、`05_error_check.mu`、`06_async_kernel.mu`。

- [ ] **Step 1:** 用 `rg -n '#include|__global__|<<<|musa|printf|main\(' code/week1/*.mu` 核对源码事实。
- [ ] **Step 2:** 在现有核心知识点后增加六节课文，依次解释 kernel/printf、线程索引、设备属性、显存生命周期、launch/执行错误和异步 launch。
- [ ] **Step 3:** 运行 `git diff --check`，并确认每个 `code/week1/*.mu` 的文件名都出现在 `learning-notes.md` 中。

### Task 2: Week 2

**Files:** Modify `code/week2/learning-notes.md`; read `01_vector_add_runtime.mu` 至 `08_stream_callback.mu`。

- [ ] **Step 1:** 用 `rg -n 'GpuTimer|CpuTimer|musaMallocHost|musaMallocManaged|musaMemcpy|musaStream|musaEvent|musaGraph|Callback|<<<|musaDeviceSynchronize' code/week2/*.mu` 核对异步边界。
- [ ] **Step 2:** 增加八节课文，串起 Runtime 7 步、pinned memory、计时、统一内存、多 stream、event DAG、Graph replay 和 callback；明确 `05_multi_stream.mu` 的 chunk 不重叠、pinned memory、CPU wall-clock，以及 `07_musa_graph.mu` 在同一 stream 上记录 event。
- [ ] **Step 3:** 运行 `git diff --check`，确认八个源码文件名、已有实测表、练习链接和高频混淆点仍存在。

### Task 3: Week 3

**Files:** Modify `code/week3/learning-notes.md`; read `01_warp_divergence.mu`、`02_reduce_naive.mu`、`03_reduce_unrolling.mu`、`04_reduce_shfl.mu`、`05_nested_hello.mu`、`06_sum_matrix_2d.mu`。

- [ ] **Step 1:** 用 `rg -n '__global__|blockIdx|threadIdx|warp|__syncthreads|shuffle|<<<|MUSA_CHECK' code/week3/*.mu` 提取索引、同步和归约边界。
- [ ] **Step 2:** 增加六节课文，依次解释 divergence、naive reduce、unrolling、shuffle、device-side launch 和 2D row-major 索引；注明 MUSA warp size/mask 依赖。
- [ ] **Step 3:** 运行 `git diff --check`，确认每个源码文件都有唯一课文入口。

### Task 4: Week 4

**Files:** Modify `code/week4/learning-notes.md`; read `01_saxpy_bandwidth.mu`、`02_offset_access.mu`、`03_offset_unrolling.mu`、`04_aos_vs_soa.mu`、`05_transpose_naive.mu`、`06_transpose_padded.mu`。

- [ ] **Step 1:** 用 `rg -n '__global__|in\[|out\[|offset|stride|struct|shared|TILE|GpuTimer|GB/s|<<<' code/week4/*.mu` 提取访存模式和计时公式。
- [ ] **Step 2:** 增加六节课文，按带宽、offset、unrolling、AoS/SoA、朴素转置、padding 转置展开；分别说明转置的读、写和 bank mapping。
- [ ] **Step 3:** 运行 `git diff --check`，确认每个源码文件都有唯一课文入口。

### Task 5: Week 5

**Files:** Modify `code/week5/learning-notes.md`; read `01_shared_basics.mu`、`02_reduce_shared.mu`、`03_transpose_shared.mu`、`04_stencil_constant.mu`、`05_naive_gemm.mu`、`06_tiled_gemm.mu`、`07_mublas_sgemm.mu`。

- [ ] **Step 1:** 用 `rg -n '__shared__|extern __shared__|__syncthreads|constant|stencil|GEMM|gemm|tile|TILE|muBLAS|<<<|GpuTimer' code/week5/*.mu` 提取 shared、constant 和 GEMM 边界。
- [ ] **Step 2:** 增加七节课文，解释 shared 生命周期、同步、constant broadcast、naive/tiled GEMM、边界处理和 muBLAS 基线；不承诺手写版本接近库性能。
- [ ] **Step 3:** 运行 `git diff --check`，确认源码覆盖和 `07_mublas_sgemm.mu` 的可选库说明完整。

### Task 6: Week 6

**Files:** Modify `code/week6/learning-notes.md`; read `01_mccl_allreduce.cpp`、`02_musa_gdb_demo.mu`、`03_error_dump.mu`、`04_torch_musa_minimal.py`、`05_torch_musa_custom_op.cpp`。

- [ ] **Step 1:** 用 `rg -n 'main\(|rank|device|communicator|AllReduce|gdb|illegal|dump|torch_musa|TORCH_LIBRARY|MUSA|error' code/week6/01_mccl_allreduce.cpp code/week6/02_musa_gdb_demo.mu code/week6/03_error_dump.mu code/week6/04_torch_musa_minimal.py code/week6/05_torch_musa_custom_op.cpp` 核对跨语言入口和环境要求。
- [ ] **Step 2:** 增加五节课文，依次解释 MCCL、GDB、error dump、torch_musa 和 custom op；说明 rank/device/communicator、复现记录和 Python/C++/MUSA 边界。
- [ ] **Step 3:** 运行 `git diff --check`，确认五个文件名都有课文入口，且原有环境记录要求未删除。

### Task 7: 全量审校与提交

**Files:** `code/week1/learning-notes.md`、`code/week2/learning-notes.md`、`code/week3/learning-notes.md`、`code/week4/learning-notes.md`、`code/week5/learning-notes.md`、`code/week6/learning-notes.md`。

- [ ] **Step 1:** 用 `rg -n 'TBD|TODO|待定|占位|待补充' code/week{1..6}/learning-notes.md` 扫描未完成标记，并运行 `git diff --check`。
- [ ] **Step 2:** 对照各周 README，确认课文顺序、核心知识点、高频混淆点、CUDA 对照、实测数据和链接均保留。
- [ ] **Step 3:** 用 `git diff --stat -- code/week{1..6}/learning-notes.md` 和 `git status --short` 检查范围；只添加六个教材文件，然后提交 `docs: expand weekly learning notes into lessons`。

## Plan self-review

- Week 1–6 均有独立任务，所有示例文件都列在任务范围中。
- Week 2 multi-stream 计时要求与当前源码修改一致。
- 计划不修改源码、README、构建系统或用户已有未提交文件。
- 所有步骤都有实际文件、命令和验证目标，没有空任务或占位内容。
