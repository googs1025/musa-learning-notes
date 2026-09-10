# Week 4 学习材料

Week 4 专注全局内存访问。GPU kernel 的算术经常不是瓶颈，访存地址是否连续、是否合并、数据布局是否适合线程访问，才决定吞吐上限。

## 阅读顺序

1. `01_saxpy_bandwidth.mu`: 建立带宽利用率基线。
2. `02_offset_access.mu`: 观察 offset 如何改变起始对齐与事务数量。
3. `03_offset_unrolling.mu`: 看展开能否弥补访存损失。
4. `04_aos_vs_soa.mu`: 对比 AoS 和 SoA 对连续访问的影响。
5. `05_transpose_naive.mu`: 观察转置中的读写方向冲突。
6. `06_transpose_padded.mu`: 用 shared tile 和 padding 改善转置。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_saxpy_bandwidth.mu` | effective bandwidth、理论带宽对照 | 只看 kernel 时间不算吞吐 |
| `02_offset_access.mu` | 合并访存和 cache line 对齐 | 以为 offset 只是多加一个整数 |
| `03_offset_unrolling.mu` | unroll 与访存模式的交互 | 以为 unroll 一定提速 |
| `04_aos_vs_soa.mu` | 数据布局决定线程读到的地址序列 | 用 CPU 结构体直觉写 GPU 数据 |
| `05_transpose_naive.mu` | 转置的读连续/写跨步问题 | 只检查结果正确不检查带宽 |
| `06_transpose_padded.mu` | shared tile、padding、bank conflict | 以为 shared memory 自动更快 |

## 逐示例课文

### 1. `01_saxpy_bandwidth.mu`：先建立带宽基线

#### 示例目标

用最简单的 SAXPY（`y[i] = a * x[i] + y[i]`）观察一个算术量很小、主要受全局内存吞吐限制的 kernel，并把时间换算成 effective bandwidth。


#### 代码结构

`saxpy` 采用一维线性索引：线程把 `blockIdx.x * blockDim.x + threadIdx.x` 映射为元素下标 `i`，用 `if (i < n)` 保护尾部线程。主函数分配并清零两个长度为 `N` 的数组，先 warmup 一次，再连续 launch 50 次，最后用平均毫秒数计算吞吐。


#### 核心知识点

这里的 effective bandwidth 是“逻辑上约定需要访问的字节数 / kernel 时间”的指标，不等同于设备实际 DRAM 流量。它会受缓存行为、事务实现和测量口径影响，因此不保证严格低于 DRAM 理论峰值，也不能直接当作硬件峰值的宣称。每个元素逻辑上读 `x[i]`、读旧的 `y[i]`、写新的 `y[i]`，代码按 `3 * bytes` 估算，所以 `GB/s = 3 * N * sizeof(float) / (平均毫秒数 / 1000) / 1e9`。这个模型适合跨实验比较；测得的数值还会受并发度、时钟和计时开销影响。相邻线程通常分别访问 `x[i]` 和 `y[i]` 的相邻 float，因而是一个合并访存的基线。


#### 执行流程

设备分配和初始化 → 一次 launch 检查 kernel 错误 → timer 包住 50 次异步 launch → `musaDeviceSynchronize()` 等待完成 → 用总时间除以 50，并按三次数组访问折算 effective bandwidth → 释放内存。计时只围绕 kernel，不应把分配、初始化或释放算进去。


#### 常见错误与实验

把 `N` 个元素的时间直接当作带宽，或把 effective bandwidth 当成实际 DRAM 带宽；漏掉 `y` 的读写之一会使逻辑字节数估算失真。可改变 `N`、block 大小和重复次数，比较 warmup 前后，并用不同初始化方式检查结果；不要因为一次测得更高就断言某个 block 大小永远更快。


#### 与本周其他示例的关系

它提供连续、对齐、合并访问的参照值。`02` 只改变输入起始 offset，`03` 在此基础上改变每线程工作量，`04` 改变数据布局，`05/06` 则把问题推进到二维转置的读写方向。


### 2. `02_offset_access.mu`：offset 改变起始对齐

#### 示例目标

保持每个线程只处理一个输出元素，比较 `in[i + offset]` 在不同 offset 下的访存时间，分离“地址仍然连续”和“起始地址是否对齐”这两个概念。


#### 代码结构

`offset_copy` 的线程下标是 `i = blockIdx.x * blockDim.x + threadIdx.x`，输出始终写 `out[i]`，输入读 `in[i + off]`。`run` 先 warmup，再重复 50 次计时；输入额外分配 64 个 float，使最大 `offset=31` 时仍不会越界。


#### 核心知识点

同一条 load 指令中，相邻线程访问的是 `in[base+off]、in[base+off+1]...`，元素之间的 stride 仍是 1，所以 offset 不会把连续访问变成跨步访问。它改变的是访问区间相对于 cache line/内存事务边界的起点：`offset=0` 通常更容易满足 alignment（对齐），非对齐 offset 可能让一个线程组覆盖更多事务，降低有效带宽。这里的 stride 是相邻线程地址的步长；offset 是统一加到所有线程地址上的位移，二者不能混为一谈。对齐应按字节和具体硬件事务边界理解，不能简单套用某个固定数字。


#### 执行流程

为输入和输出分配空间 → 依次取 `{0,1,2,4,8,16,31}` → 每个 offset 启动一次 warmup → 启动 50 次并同步 → 输出平均毫秒数。分配指针本身的对齐与 `off * sizeof(float)` 共同决定实际起始地址。


#### 常见错误与实验

看到 `i+offset` 就说“线程间 stride 变成了 offset”；忘记给输入留尾部空间；只跑一次就根据噪声排序。可把 offset 换成更宽范围，记录 effective bandwidth，并改变 block 大小、数组起始位置或数据类型观察事务边界的影响。结果可能因缓存、分配器和设备实现而不单调，实验结论应描述趋势和测量条件。


#### 与本周其他示例的关系

这是 `01` 连续访问基线的对齐扰动版，也为 `03` 提供相同 offset 集合。它帮助读者在阅读 `05/06` 时区分“连续但未对齐”和真正的“stride 很大”。


### 3. `03_offset_unrolling.mu`：展开改变工作分配，不修复地址问题

#### 示例目标

观察每线程连续处理四个元素时，循环展开对 offset 访问的影响，理解减少循环控制开销与改善全局内存事务是两个不同问题。


#### 代码结构

线程先计算 `i = global_thread_id * 4`，再用 `#pragma unroll` 展开 `k=0..3`，分别执行 `out[i+k] = in[i+k+off]`，每次仍检查边界。因为一个线程覆盖四个元素，grid 按 `N/1024`（256 线程 × 4）向上取整；主函数对四种 offset 各 warmup 并重复 50 次计时。


#### 核心知识点

展开通常能减少循环分支、计数和部分调度开销，但“每线程连续访问 4 个元素”不等于“同一条指令下线程访问连续”。令线程编号为 `t`，固定 `k` 时相邻线程访问 `4*t+k+off` 与 `4*(t+1)+k+off`，线程间 stride 是 4 个 float；因此 offset 引起的起始对齐问题也没有消失。四次访问还可能增加寄存器占用、改变活跃线程数或让尾部控制更复杂。编译器是否已经自动展开、kernel 是否真的受循环控制限制，都需要测量，不能把 `unroll` 当作一定加速的开关。


#### 执行流程

每个线程取得四元素片段 → 编译器按 pragma 尝试展开四次 load/store → 对 `i+k<n` 的有效元素执行拷贝 → 不同 offset 分组计时并打印。最后一个逻辑片段可能只有部分元素有效，不能因为线程负责四个位置就取消边界检查。


#### 常见错误与实验

把展开误解为让一个线程访问四个相邻地址就必然合并；把 `grid` 仍按 `N/256` 配置而导致大量重复工作；删掉边界判断造成越界。可与 `02` 对相同 offset 直接比较，改变展开因子为 2/8，检查寄存器和 occupancy，并同时看时间与有效带宽；展开可能无收益、收益很小，甚至因资源压力变慢。


#### 与本周其他示例的关系

它是 `02` 的控制流/每线程工作量变体，而不是新的合并访存方案。后面的 `04` 说明布局如何直接改变线程地址序列，`05/06` 说明仅靠展开无法解决二维转置中的跨步写。


### 4. `04_aos_vs_soa.mu`：字段布局决定地址序列

#### 示例目标

比较同一计算 `x+y` 在 AoS 和 SoA 两种布局下的全局内存访问，练习从“相邻线程访问了什么地址”而不是从 CPU 结构体语义判断性能。


#### 代码结构

AoS 用 `Particle{float x,y,z,w}`，线程 `i` 读取 `p[i].x` 和 `p[i].y`；SoA 分别传入连续的 `x[i]`、`y[i]` 数组。两种 kernel 都使用一维 256-thread block、相同规模和各自 30 次计时，打印两者毫秒数及 `ta/ts`。


#### 核心知识点

AoS 中相邻对象间距是 16 字节，同一字段的相邻线程地址相隔 16 字节；本 kernel 只计算 `x+y`，`z/w` 不参与计算，但读取结构体时其他字段可能与所需字段一起带入 cache 或内存事务。SoA 中相邻线程读 `x[i]` 或 `y[i]` 的地址间隔是 4 字节，更容易形成合并访问。SoA 并非一定更快：如果算法经常消费一个对象的全部字段，AoS 的局部性可能更合适；转换成本、缓存命中、对齐、寄存器和具体硬件也会改变结果。输出的 speedup 只是本次布局、规模和设备的测量值。


#### 执行流程

分别分配 Particle、x、y 和 out → 启动 AoS warmup 并重复计时 → 启动 SoA warmup 并重复计时 → 计算时间比。示例没有初始化或校验数据，重点是访问模式与测量框架；实际实验应补充初始化和结果校验。


#### 常见错误与实验

只比较结构体大小而不画出字段地址；看到 SoA 的单次结果更快就推广为普遍规律；忽视两种版本的工作量、重复次数或 cache 状态。可改变结构体字段数、只读取一个字段或全部字段，改变 N 与字段对齐，并用 profiler/有效带宽验证“少搬了哪些字节”。


#### 与本周其他示例的关系

它把 `02` 的地址连续性问题从统一 offset 扩展到固定字段 stride；它也为转置示例提供布局视角：优化目标始终是让同一批线程的 load/store 地址尽量连续。


### 5. `05_transpose_naive.mu`：转置同时拥有两个方向

#### 示例目标

建立二维转置 baseline，明确一个结果正确的转置仍可能因为写方向跨步而浪费全局内存事务。


#### 代码结构

二维线程坐标为 `x = blockIdx.x*blockDim.x+threadIdx.x`、`y = blockIdx.y*blockDim.y+threadIdx.y`；输入坐标是 `(column=x, row=y)`，按 row-major 的 `in[y*w+x]` 读取。转置后的输出坐标是 `(column=y, row=x)`，按 row-major 写作 `out[x*h+y]`，因此输入为 H×W 时，输出行列形状为 W×H。16×16 block 覆盖 2048×2048 矩阵，warmup 后重复 20 次计时。


#### 核心知识点

在同一输入行 `row=y` 的线程中，输入坐标是 `(column=x, row=y)`，所以 `in[y*w+x]` 随 `threadIdx.x` 增加 1 个 float，读是连续的；转置后输出坐标变为 `(column=y, row=x)`，`out[x*h+y]` 中第一维 row-major 下标是输出行 `x`、第二维是输出列 `y`。对固定输入行 `y` 的相邻线程，输出行 `x` 变化而输出列 `y` 相同，所以写地址随 `x` 增加 `h` 个 float，是 stride=`h` 的跨步访问；沿另一方向的输出列地址才相邻。输入为 H×W 时，输出行列形状为 W×H。硬件一次内存指令关注的是同一批线程的地址集合，因此 naive 版本通常读合并、写不合并。转置的优化不能只检查数值结果，必须分别检查 load 和 store 的方向及有效带宽。


#### 执行流程

每个线程读取输入坐标 `(column=x, row=y)` → 按源码 `out[x*h+y]` 写入输出坐标 `(column=y, row=x)` → block 边界线程退出 → 用多次 launch 的平均时间作为 baseline。由于 `W=H=2048`，当前示例的输出分配和索引都匹配；推广到非方阵时仍要牢记输出行列形状为 W×H。


#### 常见错误与实验

把 `out[x*h+y]` 写成 `out[y*w+x]` 而变成复制；把“读连续”误说成“整个 kernel 合并”；混淆 `w`（输入行宽）和 `h`（输出行宽）。可改变 W/H、block 形状并与 `06` 对比，检查转置结果和两方向的地址公式；仅看到结果正确不能证明带宽合理。


#### 与本周其他示例的关系

这是 `01` 连续访问基线在二维坐标和转置写入下的反例，也把 `04` 的 stride 影响具体化。`06` 保留同样的数学转置，只重排数据经过 shared memory 的路径。


### 6. `06_transpose_padded.mu`：shared tile 与 bank mapping

#### 示例目标

用 shared tile 把全局内存的跨步写改造成两次更规整的全局访问，并通过 padding 改善 tile 转置时的 shared-memory bank 冲突。


#### 代码结构

每个 32×32 block 先以 `(threadIdx.x, threadIdx.y)` 从全局内存连续读入 `tile[ty][tx]`，同步后交换 block 坐标和 tile 下标，以 `tile[tx][ty]` 写出转置结果。声明为 `tile[32][33]`，第二维多出的 1 个 float 是 padding；主函数用 32×32 block、20 次计时，与 `05` 的 naive baseline 对照。


#### 核心知识点

第一阶段同一批线程读 `in[y*w+x]`，全局读连续；同步保证所有线程完成 tile 填充后，第二阶段才能安全地从 shared tile 读取并写全局。若 shared tile 是 `[32][32]`，row-major 下同一列的地址间隔为 32 个 float；在典型 32-bank、4-byte bank 的模型中，`bank=(row*32+col) mod 32`，一列访问会映射到同一个 bank。改成 33 列后，`bank=(row*33+col) mod 32=(row+col) mod 32`，不同 row 会错开，因而缓解转置读取的 bank conflict。padding 只改变 shared 布局，不改变矩阵逻辑，也不等于 shared memory 自动更快：同步、额外搬运、占用 shared memory、边界处理和设备实现都可能让收益很小或为负。


#### 执行流程

计算输入坐标并有界写入 tile → `__syncthreads()` 建立 block 内可见性 → 交换 x/y 后有界读取 tile 的转置位置 → 有界写出 `out[y*h+x]` → 重复 launch 计时。当前 W/H 都是 TILE 的整数倍，因此每个 tile 都被完整填充；若改成非整倍数尺寸，边界线程不能简单跳过填充后仍参与第二阶段，否则可能读取未初始化的 shared 元素，需要额外的零填充或更严密的条件设计。


#### 常见错误与实验

漏掉同步导致读到未完成的数据；把 padding 写成第一维加 1 或错误修改索引；只凭“shared + padding”断言一定加速；忽略非整倍数尺寸的 tile 边界。可比较 `[32][32]` 与 `[32][33]`、不同 TILE/block 形状和非方阵输入，结合结果校验、时间和 profiler 观察 bank mapping、全局事务与 occupancy 的共同影响。


#### 与本周其他示例的关系

它是 `05` 的同一转置公式的 tiled 版本，综合了 `02` 的对齐/事务意识、`03` 的“优化不保证加速”原则和 `04` 的布局重排思想，形成本周从地址序列到 shared-memory 映射的收束。


## 代码阅读抓手

每个 kernel 都问同一个问题：相邻线程访问的地址是否相邻。

如果相邻线程访问的是 `a[i]`、`a[i+1]`、`a[i+2]`，通常更容易合并；如果访问的是 `a[i * stride]` 或结构体中的分散字段，就要警惕吞吐下降。

## 高频混淆点

- **结果正确不代表性能正确**: offset、stride、AoS 代码可能结果完全正确, 但相邻线程访问地址不连续, 带宽会明显下降。
- **合并访存看的是线程组访问序列**: 不要只看单个线程访问了什么, 要看相邻线程在同一条 load/store 指令上访问的地址是否相邻。
- **AoS 是 CPU 友好, 不一定 GPU 友好**: GPU 上一组线程常常只读同一个字段, SoA 更容易让地址连续。
- **转置有读写两个方向**: naive transpose 往往读连续但写跨步, 或反过来。优化时要分别看 load 和 store。
- **unroll 不是万能加速**: unroll 能减少循环/调度开销, 但如果访存模式差或寄存器压力变大, 可能收益很小甚至变慢。

## CUDA_Freshman 对照

- `18_sum_array_offset`: offset 访问。
- `19_AoS`、`20_SoA`: 数据布局。
- `21_sum_array_offset_unrolling`: offset + unrolling。
- `22_transform_matrix2D`: matrix transpose。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
