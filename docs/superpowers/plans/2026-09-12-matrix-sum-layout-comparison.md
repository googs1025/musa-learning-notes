# 矩阵求和布局对比实验实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `06_sum_matrix_2d.mu` 中同时展示二维/二维、 一维/一维两种矩阵行求和方式，并输出 CPU/GPU 计时和精度对比。

**Architecture:** 保留现有二维 kernel，新增一维 kernel；两个 GPU kernel 都输出 `float rows[H]`，由 host 使用 `double` 汇总总和。CPU 参考也按行使用 `double` 累加。GPU 分别报告 event 测得的 kernel-only 时间和 CPU wall-clock 测得的 H2D+kernel+D2H 端到端时间。

**Tech Stack:** MUSA C++、`musa_runtime`、`GpuTimer`、`CpuTimer`、C++ 标准库、Markdown。

---

## 文件边界

- Modify: `code/week3/06_sum_matrix_2d.mu` — 两种 GPU 映射、CPU 参考、计时、误差报告和示意注释。
- Modify: `code/week3/learning-notes.md` — 说明两种映射、计时口径和精度差异。

不修改其他 Week 3 源码、README、Makefile 或外部依赖。

## Task 1: 扩展矩阵 kernel 和 CPU 参考

**Files:** Modify `code/week3/06_sum_matrix_2d.mu`。

- [ ] **Step 1: 写静态 RED 检查**

确认当前文件只有 `matrix_to_row_sums`，没有一维 kernel、误差函数和 CPU 计时输出。检查命令：`rg -n 'matrix_to_row_sums_1d|max_abs|CpuTimer|GPU 2D|GPU 1D' code/week3/06_sum_matrix_2d.mu`；预期新符号尚不存在。

- [ ] **Step 2: 增加一维行求和 kernel**

新增形如下面的 kernel，使用线性下标并恢复二维坐标：

```cpp
__global__ void matrix_to_row_sums_1d(const float* m, float* rows, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx < total) {
        int x = idx % width;
        int y = idx / width;
        atomicAdd(&rows[y], m[idx]);
    }
}
```

保留 `x/y` 恢复代码和边界判断，使课文能够直接对应 `idx = iy * width + ix`。

- [ ] **Step 3: 增加 CPU double 参考**

分配 `double rows_cpu[H]`，用 `CpuTimer` 只包住逐行累加，不包含输入初始化：

```cpp
for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
        rows_cpu[y] += static_cast<double>(h[y * W + x]);
    }
}
```

同时用 `double total_cpu` 汇总 `rows_cpu`。

- [ ] **Step 4: 为两种 GPU 版本增加独立 buffer 和计时**

为 2D/1D 分别分配 `d_rows_2d`、`d_rows_1d`，每次运行前用 `musaMemset` 清零。使用 `GpuTimer` 包住 kernel-only，使用 `CpuTimer` 包住完整 H2D、kernel、`musaDeviceSynchronize`、D2H 流程。两个版本使用相同输入和相同 `block` 总线程数；二维版本使用 `block(16,16)`、`grid((W+15)/16,(H+15)/16)`，一维版本使用 `threads=256`、`grid=(W*H+threads-1)/threads`。

- [ ] **Step 5: 增加误差和输出函数**

实现最大绝对误差、最大相对误差和总和差异计算。相对误差分母使用 `max(abs(reference), 1e-12)`；输出 CPU 时间、GPU 2D/1D kernel-only、GPU 2D/1D end-to-end、首行/末行/总和和 PASS/FAIL。保留 `rows` 的 host 验证和资源释放。

- [ ] **Step 6: 增加源码示意注释**

在两个 kernel 前说明：二维版本直接得到 `(x,y)`，一维版本先得到 `idx` 再用 `% width` 和 `/ width` 恢复 `(x,y)`；补充同一行多个线程共同更新 `rows[y]` 的 `atomicAdd` 图示，以及 kernel-only/端到端的计时范围。

- [ ] **Step 7: 静态验证**

运行：`git diff --check`；确认 `parent`/其他 Week 3 kernel 未被修改；确认文件包含 `matrix_to_row_sums`、`matrix_to_row_sums_1d`、`CpuTimer`、两个 `GpuTimer` 和误差输出字段。

### Task 2: 更新 Week 3 课文

**Files:** Modify `code/week3/learning-notes.md`。

- [ ] **Step 1: 在 06 小节补充两种映射**

解释二维版本：`blockIdx/threadIdx → (ix,iy) → iy*width+ix`；解释一维版本：`idx → ix=idx%width, iy=idx/width`。使用当前 `W=H=1024`、二维 `16×16` block、二维 `64×64` grid 和一维 `256` threads 的实际数字。

- [ ] **Step 2: 补充数据流和总和定义**

明确两个 GPU kernel 都先得到 `rows[y]`，host 再把 H 行相加成矩阵总和；说明 `rows[y]` 不是单个线程结果，而是同一行所有元素的累加结果。

- [ ] **Step 3: 补充计时和精度解释**

说明 CPU 使用 double，GPU 行和使用 float + atomicAdd；原子累加顺序可能不同，所以 GPU 结果允许微小误差。区分 GPU kernel-only 与端到端时间，说明 H2D/D2H 是否包含在指标中。

- [ ] **Step 4: 补充实验建议**

建议修改为非方阵、非 16 整除尺寸，检查两种映射的边界；修改输入模式观察浮点误差；比较一维/二维 block 的线程覆盖和耗时，不能把任意一次测量推广为普遍性能结论。

### Task 3: 全量验证与交付

**Files:** `code/week3/06_sum_matrix_2d.mu`、`code/week3/learning-notes.md`。

- [ ] **Step 1: 静态检查**

运行：`git diff --check`；运行覆盖检查确保 `06_sum_matrix_2d.mu` 的课文入口仍存在；扫描未完成标记。

- [ ] **Step 2: MUSA 环境编译运行**

在 AutoDL 执行：`cd code/week3 && make clean && make 06_sum_matrix_2d && ./06_sum_matrix_2d`。预期输出包含 CPU、GPU 2D、GPU 1D 的时间，首/末行和总和，误差指标，以及所有比较通过。若本地无 `mcc`，明确记录未执行，不伪称编译通过。

- [ ] **Step 3: 检查变更范围**

运行：`git status --short` 和 `git diff --stat -- code/week3/06_sum_matrix_2d.mu code/week3/learning-notes.md`；确认没有修改其他文件或覆盖用户未提交的 `code/week1/06_async_kernel.mu`。

## Plan self-review

- 设计稿的二维/一维映射、CPU/GPU 计时、行和/总和和精度比较均有对应步骤。
- 计划明确 GPU 总和由行和在 host 使用 double 汇总，避免引入未设计的全局 scalar atomic kernel。
- 计划保持修改范围为一个源码文件和一个课文文件。
- 所有验证命令、预期结果和 MUSA 环境限制均已写明，没有空任务。
