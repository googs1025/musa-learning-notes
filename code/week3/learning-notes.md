# Week 3 学习材料

Week 3 的主线是执行模型。Reduce 是最适合入门的实验对象：它既能暴露线程索引、访存、同步和分支问题，又能逐步演进到 unrolling 和 shuffle。

## 阅读顺序

1. `01_warp_divergence.mu`: 先观察分支分化的代价。
2. `02_reduce_naive.mu`: 建立 naive reduce 基线。
3. `03_reduce_unrolling.mu`: 看每个线程多处理元素如何减少开销。
4. `04_reduce_shfl.mu`: 进入 warp-level reduce。
5. `05_nested_hello.mu`: 理解 host 编排的两阶段 kernel 调度。
6. `06_sum_matrix_2d.mu`: 把一维索引扩展到二维数据。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_warp_divergence.mu` | 同一 warp 内分支不一致会串行化路径 | 把 branch 数量等同于性能损失 |
| `02_reduce_naive.mu` | 分块归约、host final reduce | 忽略 block 间不能直接同步 |
| `03_reduce_unrolling.mu` | 每线程多元素、减少 block 数和循环开销 | 只看算术操作不看访存模式 |
| `04_reduce_shfl.mu` | warp shuffle、warp size 差异 | 直接照搬 CUDA 的 32-wide 假设 |
| `05_nested_hello.mu` | host 分两阶段启动 parent/child kernel | 把两个阶段误认为一个普通函数调用 |
| `06_sum_matrix_2d.mu` | 2D grid、row-major 展开 | 混淆 `(x,y)` 和 `row/col` |

## 代码阅读抓手

Reduce 代码重点看三个边界：

- block 内如何同步。
- block 间结果如何汇总。
- 最后一段不足一个 block 或一个 warp 时如何处理。

MUSA 的 warp size 和 CUDA 常见值不同，所有 shuffle 或 warp reduce 都必须先确认设备能力和 SDK intrinsic 行为。

## 高频混淆点

- **block 同步不等于 grid 同步**: `__syncthreads()` 只管同一个 block。不同 block 的 partial sum 要靠另一个 kernel、host final reduce 或专门的全局同步机制处理。
- **分支数量不等于分支代价**: warp divergence 的关键是同一个 warp 内线程是否走不同路径。所有线程都走同一分支时, 分支本身不一定是主要问题。
- **reduce 的最后一段最容易错**: N 不是 block size 或 unroll 粒度整数倍时, 尾部元素必须有边界保护。
- **shuffle 不是 shared memory 的简单替代**: shuffle 只在 warp/group 内传值, 不能跨 block, 也不能替代需要全 block 协作的同步。
- **CUDA 的 32-wide 直觉要重审**: 看到 `32`、`0xffffffff`、`warpSize`、`lane`、`mask` 时, 都要问这段逻辑在 MUSA warp size 下是否仍成立。

## 逐示例课文

### 1. `01_warp_divergence.mu`：同一 warp 的分支路径

#### 示例目标

这个程序把“分支是否发散”变成可测量的对照实验。`mode=0` 时，同一个 block 内的线程都执行相同的算术路径（虽然用 `blockIdx.x` 选择路径）；`mode=1` 时，`threadIdx.x` 的奇偶性让同一执行组中的线程可能走乘法或除法两条路径，观察两者平均耗时和 slowdown 的差异。这里的重点不是分支语句的数量，而是同一 warp/group 是否需要串行执行不同路径。


#### 代码结构

`branch_kernel` 计算全局索引 `i`，越界线程直接返回，再按 `mode` 选择 coherent 或 divergent 分支并写回 `out[i]`。`run` 固定使用 256-thread block，先预热一次，再重复 launch 50 次计时；`main` 分配 `N=1<<24` 的 device buffer，分别运行两种模式并打印结果。


#### 核心知识点

warp divergence 是执行模型问题：若一个 warp 内存在不同分支，硬件通常要分别执行路径并屏蔽不满足条件的 lane，代价取决于路径、占比、访存和编译结果。`blockIdx.x` 在 block 内一致，所以 mode 0 的选择不会制造 block 内分化；`threadIdx.x & 1` 则会在通常的多 lane warp/group 中制造交错分化。不要把这里的“warp”默认解释成 CUDA 的 32 个线程；MUSA 的 warp size/group 规则和 mask 语义应以当前设备与 SDK 为准。


#### 执行流程

host 分配输出 → 启动一次 kernel 预热 → 启动 50 次并用 `GpuTimer` 求平均 → 先测 mode 0，再测 mode 1 → 打印 `coherent`、`divergent` 和比值 → 释放显存。第一次 launch 未计入平均时间，但它仍用于提前暴露 kernel 错误。


#### 常见错误与实验

把所有分支都叫作 divergence，或认为 slowdown 必然固定；实际上 `if (blockIdx.x & 1)` 在 block 内是统一路径。可改变 block size、`N`、重复次数，并把奇偶判断改成按连续区间判断，比较分化比例和测量噪声。还应检查计时同步和 `MUSA_CHECK_KERNEL()` 的位置，避免把异步 launch 当成已完成。


#### 与本周其他示例的关系

它先建立“同一执行组内的控制流”这个观察角度；`02` 的 naive reduce 在折半归约后几轮也会出现活跃线程减少和分支分化，`03` 通过减少循环/归约规模改善开销，`04` 则把组内数据交换改为 shuffle。


### 2. `02_reduce_naive.mu`：shared memory 的分块归约基线

#### 示例目标

把 `N` 个输入值求和，同时展示为什么不能只在一个 kernel 里依靠 `__syncthreads()` 完成整个 grid 的归约。每个 block 先产生一个 partial sum，跨 block 的最后一步明确放到 host 上完成。


#### 代码结构

`reduce_naive` 中每个线程加载一个元素到动态 shared memory `s`，越界填零；随后以 `stride = blockDim.x / 2` 开始的折半循环让前半线程累加后半线程，并在每轮后同步；线程 0 写出 `partial[blockIdx.x]`。`main` 使用 `N=1<<22`、256 threads，分配输入和 partial buffer，执行 kernel，拷回 partial 后由 host 串行累加。


#### 核心知识点

shared memory 是 block 内协作的暂存区，`__syncthreads()` 只保证同一 block 的线程到达同步点，不能让不同 block 互相等待。因此 `partial` 必须通过第二个 kernel、host final reduce 或其他全局同步机制继续归约。折半归约易读，但每轮活跃线程数减半，条件 `tid < stride` 也可能造成分化；尾部通过越界填零保证 `N` 不整除 block size 时仍安全。


#### 执行流程

host 初始化全 1 → 拷入 device → 以 `blocks=(N+threads-1)/threads` 启动一次动态 shared-memory kernel → block 内得到 16384 个 partial → 拷回 host → 串行求和并与 `N` 对照 → 释放资源。kernel 计时只覆盖第一阶段，不包含拷回和 host final reduce。


#### 常见错误与实验

误以为 `__syncthreads()` 是 grid barrier；忘记越界填零；在 shared 写入后少一次同步；把 kernel 时间和完整算法时间混为一谈。可把输入改成非 1、故意使用非整除的 `N`，再增加一个 device-side final kernel 与 host 版本比较；同时观察不同 block size 对 partial 数量和耗时的影响。


#### 与本周其他示例的关系

这是 `03`、`04` 的共同基线：三者都采用“block partial + host final reduce”结构。`03` 保留 shared 折半阶段，只改变加载粒度；`04` 把 block 内的归约前半段下沉到 warp/group shuffle。


### 3. `03_reduce_unrolling.mu`：每线程处理两个元素

#### 示例目标

在不改变最终归约结构的前提下，减少 block 数和部分循环开销，演示最小的 unrolling 优化。它用于和 `02_reduce_naive.mu` 做单变量对比，而不是声称只要展开就一定更快。


#### 代码结构

`reduce_unroll2` 将起点改为 `blockIdx.x * blockDim.x * 2 + tid`；每个线程最多读取 `i` 和 `i + blockDim.x` 两个位置，在寄存器 `v` 中先相加，再写入 shared `s[tid]`。之后仍执行与 naive 相同的 block 内折半归约并由线程 0 写 partial。`main` 用 `blocks=(N+threads*2-1)/(threads*2)`，因此本例是 8192 个 partial。


#### 核心知识点

unrolling2 同时降低了覆盖同样 `N` 所需的 block 数，并让每个线程在进入 shared 归约前做更多工作；两个连续的 block-sized 区段也通常保持规整的合并访问。优化收益还受寄存器、占用率、内存带宽和尾部边界影响，不能只数加法次数。`i` 与 `i+blockDim.x` 都必须独立检查，最后不足两个元素时不能越界。


#### 执行流程

host 初始化并拷入输入 → 每个 block 的 256 个线程各加载最多两个值 → 同步后在 shared 中折半归约 → 写出 8192 个 partial → 拷回 host 串行汇总 → 打印总和、期望值、kernel 时间和 partial 数。


#### 常见错误与实验

只检查第一个加载位置；把 unroll 因子改成 4 却忘了同步修改起点、加载边界和 block 计算；只看 kernel 时间却忽略数据传输。可把因子 2 改为 4，比较 `N` 非整除时的正确性、访存合并、寄存器使用和耗时，并与 `02` 在同一输入和计时口径下比较。


#### 与本周其他示例的关系

它是从 `02` 到更高效 reduce 的中间台阶：先保留 shared 和 host final，隔离出“加载/分块粒度”的影响；`04` 再隔离出“warp/group 内通信方式”的影响，而 `01` 提供了理解折半循环分支代价的执行模型背景。


### 4. `04_reduce_shfl.mu`：warp/group 内寄存器交换

#### 示例目标

展示如何用 `__shfl_down_sync` 在一个 warp/group 内直接交换寄存器值，减少 shared memory 和 block 内同步的使用，同时仍处理一个 block 中多个 warp/group 的结果。


#### 代码结构

`warp_reduce_sum` 从 `warpSize/2` 开始按半递减 offset 做 shuffle 累加；kernel 先让每个线程加载一个元素并完成一次 warp/group reduce，`lane==0` 把各组结果写入 `warp_sums`，同步后由第一个 warp/group 再归约这些结果，线程 0 写出 block partial。host 仍拷回 partial 做最终串行求和。


#### 核心知识点

shuffle 的通信范围是一个 warp/group，不能跨 block，也不能替代多个 warp/group 写 shared 后所需的 `__syncthreads()`。当前源码应视为 SDK 语义探测/教学骨架，而不是可移植的 128-lane 完整归约实现：`0xffffffff` 是 32-bit 的 CUDA 风格 mask，不能表达 128 个 lane；同时 `main` 中 `((threads+127)/128)` 的 shared 大小也带有 MUSA 常见 128-wide 环境的假设。虽然本例用 `warpSize` 计算 offset、lane、warp 编号，但在宣称完整归约正确之前，必须按实际 MUSA SDK 文档验证 mask、width 和 `warpSize` 的语义，并据此修正实现。不要机械套用 CUDA 32-wide 和 32-bit active mask。


#### 执行流程

每线程加载一个值（越界为 0）→ 组内 shuffle 归约 → 每组 lane 0 写一个 shared partial → block 同步 → 第一个组加载各组 partial 并再次 shuffle 归约 → 写出 block partial → host 拷回并求和。256 threads 下，组数由运行时 `warpSize` 决定，不应仅凭 CUDA 经验写死。


#### 常见错误与实验

把 shuffle 当作跨 block 通信；把 offset 固定成 16 或把 mask 固定成 CUDA 的 32-lane 全掩码；没有给 `warp_sums` 分配足够空间；最后一个 block 或最后一个 group 的无效 lane 没有置零。可先打印运行时 `warpSize`，再用不同线程数（尤其不是 warpSize 整数倍的配置）检查 shared 大小、有效 lane 和结果；同时对比 `02` 的正确性与计时。


#### 与本周其他示例的关系

它是在 `02` 的分块归约骨架上替换 block 内通信原语，也是对 `03` 优化边界的进一步推进。`01` 解释了为什么减少活跃路径可能重要；`05` 把关注点从组内同步转向 kernel 的 device-side 调度。


### 5. `05_nested_hello.mu`：host 编排的两阶段 kernel 调度

#### 示例目标

用 host 端连续启动两个 kernel，模拟 parent → child 的两阶段任务关系。这样可以在当前不接受 device-side launch 的 MUSA 编译器上正常编译，同时让读者先理解“阶段之间需要等待”和“启动新的 kernel 不等于调用普通 device 函数”。


#### 代码结构

`child_kernel` 只让每个 child block 的线程 0 打印 block 编号；`parent_kernel` 也只让每个 parent block 的线程 0 打印。`main` 先启动 2 个 parent blocks，调用 `MUSA_CHECK_KERNEL()` 等待完成，再由 host 启动一个 2-block child grid，因此输出是 2 行 parent 加 2 行 child，两个阶段不会交错。


#### 核心知识点

这里不再依赖 device-side launch，也不声称演示动态并行；核心是 host 在两个阶段之间显式同步。第一个 `MUSA_CHECK_KERNEL()` 确认 parent 完成，第二个确认 child 完成。与真正的动态并行相比，这种写法牺牲了由 GPU 自己决定下一阶段启动时机的能力，但更容易观察、调试和编译。


#### 执行流程

host launch 两个 parent blocks → `MUSA_CHECK_KERNEL()` 等 parent 完成 → host launch 一个 `<<<2,4>>>` child grid → child 的两个 block 各由 thread 0 打印 → 再次检查 child 完成。parent 输出在前，child 输出在后；同一阶段内的 block 输出顺序仍不应作为调度保证。


#### 常见错误与实验

把第二阶段 kernel 当成普通函数；忘记在两个阶段之间同步；以为一个 parent block 的每个线程都会产生一行输出。可把 parent grid 或 child grid 的 block 数改掉，观察每个 block 只有 thread 0 打印；再删除第一阶段的 `MUSA_CHECK_KERNEL()`，思考为什么 host 可能在 parent 完成前就提交下一阶段。


#### 与本周其他示例的关系

前四个示例主要研究一个 kernel 内的执行、同步和归约；本例把关注点扩展到 kernel 阶段之间的 host 调度。它与 `02` 的“跨 block 需要新的阶段”形成对照：新阶段先由 host 显式发起，后续再学习设备支持时再讨论 device-side launch；`06` 则回到显式二维数据映射。


### 6. `06_sum_matrix_2d.mu`：二维索引与 row-major 行求和

#### 示例目标

把一维线程索引推广到矩阵，计算每一行的和，演示 2D grid/block 的坐标映射、边界判断和 C/C++ row-major 地址展开。这里 `W` 表示列数、`H` 表示行数，因此输入是 `H` 行 `W` 列，输出 `rows` 有 `H` 个元素。


#### 代码结构

kernel 用 `y = blockIdx.y * blockDim.y + threadIdx.y`、`x = blockIdx.x * blockDim.x + threadIdx.x` 得到矩阵坐标；只有 `x < width && y < height` 的线程执行 `atomicAdd(&rows[y], m[y * width + x])`。`main` 创建 `W=H=1024` 的全 1 矩阵，使用 `block(16,16)` 和 `grid(64,64)` 覆盖矩阵，把行结果清零后拷回并检查首尾行。一个线程负责一个 `m[y][x]`，但同一行的很多线程共同写入同一个 `rows[y]`。


#### 核心知识点

row-major 中第 `y` 行第 `x` 列的线性位置是 `y * width + x`，所以同一行相邻线程通常访问相邻元素；这里的 `(x,y)` 是列/行坐标，而 `rows[y]` 明确按行聚合。以 `W=H=1024` 为例，`block(16,16)` 产生 `grid(64,64)`，总共覆盖 1024×1024 个元素。每个 row 有多个线程同时更新，因此需要 `atomicAdd` 保证累加正确；代价是同一行的原子竞争，代码注释也指出高性能版本应先做 block 内归约再写出。二维 grid 不改变 block 内同步和跨 block 的限制。


#### 执行流程

host 分配并初始化连续 row-major 矩阵 → 拷入 device、将 `rows` 清零 → 以 16×16 block 启动覆盖 1024×1024 元素的 kernel → 合法线程按行原子累加 → 检查并拷回 1024 个行和 → 打印 `row0`、期望值和最后一行 → 释放资源。


#### 常见错误与实验

把 `x` 当行、`y` 当列；地址误写成 `x * height + y`；只检查一维边界；忘记清零 `rows`；把 atomicAdd 当成无代价操作。可改成非方阵并使用不能整除 16 的宽高验证边界，填入递增值手算一行；再实现 block 内行归约，比较原子竞争、访存布局和结果一致性。


#### 与本周其他示例的关系

它复用 `02`/`03` 的“多个线程产生局部贡献、最后汇总”思想，但把一维 partial 换成按 `y` 索引的行结果，并用 atomic 处理并发写入；它也延续 `01` 的线程坐标影响执行/访存的视角，与 `05` 的 kernel 层级调度无直接依赖。


## CUDA_Freshman 对照

- `8_divergence`: 分支分化。
- `10_reduceInteger`、`12_reduce_unrolling`、`29_reduce_shfl`: reduce 三阶演进。
- `28_shfl_test`: shuffle API 探测。
- `13_nested_hello_world`: nested kernel / 分阶段调度概念对照。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
