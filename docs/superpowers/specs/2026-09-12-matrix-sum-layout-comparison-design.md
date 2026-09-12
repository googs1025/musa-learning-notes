# 矩阵求和布局对比实验设计

## 背景

`code/week3/06_sum_matrix_2d.mu` 当前使用二维 grid 和二维 block，为矩阵的每个元素计算二维坐标并按行累加。学习者还需要看到：同一份矩阵数据也可以使用一维 grid/一维 block，通过线性下标恢复 `(x, y)`，得到相同的结果。

## 目标

- 保留现有二维 grid + 二维 block 的行求和示例。
- 增加一维 grid + 一维 block 的等价行求和 kernel。
- 增加 CPU `double` 参考计算。
- 输出每行求和和整个矩阵总和。
- 同时输出 CPU 计算时间、GPU kernel-only 时间和 GPU 端到端时间。
- 比较 CPU、GPU 二维版本和 GPU 一维版本的绝对误差、相对误差和总和差异。
- 在源码注释和 Week 3 课文中详细解释二维坐标、一维线性下标及 row-major 映射。

## 方案

两个 GPU kernel 都计算 `rows[y]`，而不是直接竞争一个全局 scalar：

```text
CPU reference:
  h[H][W] → double rows_cpu[H] → double total_cpu

GPU 2D:
  (blockIdx, threadIdx) → (x, y) → m[y * width + x] → rows_2d[y]

GPU 1D:
  idx → x = idx % width, y = idx / width
      → m[idx] → rows_1d[y]
```

两个 GPU kernel 均使用 `atomicAdd` 更新行和，保留当前示例的教学重点；GPU 行和使用 `float`，CPU 参考使用 `double`，这样可以观察浮点累加顺序和精度差异。GPU 总和由拷回 host 的行和使用 `double` 汇总，避免额外引入一个竞争全局 scalar 的 kernel。

## 计时口径

- CPU：只测 CPU 参考计算，不包含输入初始化。
- GPU kernel-only：用 `GpuTimer` 只包住 kernel launch，并在事件/同步后读取时间。
- GPU end-to-end：用 `CpuTimer` 包住 H2D、kernel、完成同步和 D2H，反映完整应用路径。
- 二维和一维 GPU 版本使用相同矩阵尺寸、相同输入、相同 block 总线程规模和相同同步口径。

## 输出与精度

输出至少包含：

- 矩阵尺寸、二维 block/grid 配置、一维 block/grid 配置；
- CPU、GPU 2D kernel-only、GPU 2D end-to-end、GPU 1D kernel-only、GPU 1D end-to-end 时间；
- 首行、末行和总和的 CPU/GPU 对照；
- GPU 2D 与 CPU、GPU 1D 与 CPU、GPU 2D 与 GPU 1D 的最大绝对误差和最大相对误差；
- 总和误差及 PASS/FAIL。

输入使用固定、可复现的非全 1 浮点模式，避免所有求和都精确相同而看不到精度差异。误差阈值必须在代码中明确，并允许原子累加顺序导致的微小浮点差异。

## 文件范围

- Modify: `code/week3/06_sum_matrix_2d.mu` — 增加 CPU 参考、一维 kernel、计时、精度报告和详细示意注释。
- Modify: `code/week3/learning-notes.md` — 补充两种映射、计时口径和精度比较说明。

不修改其他 Week 3 kernel、不引入额外依赖、不改变 CMake/Makefile。

## 验收标准

- 二维和一维 GPU kernel 均能编译并覆盖所有矩阵元素。
- 非整除尺寸仍通过边界判断，不发生越界。
- CPU、GPU 2D、GPU 1D 的行和与总和都能输出并比较。
- GPU kernel-only 与端到端时间的测量范围在注释和输出中清楚区分。
- 误差报告包含绝对误差、相对误差和总和差异。
- `git diff --check` 通过；MUSA 环境可用时运行目标程序并验证 PASS。
