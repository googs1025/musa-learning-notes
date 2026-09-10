# Week 5 学习材料

Week 5 进入片上内存和 GEMM。主线是从 global memory 直接访问，逐步引入 shared memory、constant memory 和 tiled GEMM，再用 muBLAS 建立库函数性能基准。

## 阅读顺序

1. `01_shared_basics.mu`: 先学 static/dynamic shared memory 写法。
2. `02_reduce_shared.mu`: 用 shared memory 改写 reduce。
3. `03_transpose_shared.mu`: 用 shared tile 处理 transpose。
4. `04_stencil_constant.mu`: 用 constant memory 放小型只读参数。
5. `05_naive_gemm.mu`: 建立 GEMM 正确性和性能基线。
6. `06_tiled_gemm.mu`: 用 shared tile 降低 global memory 访问。
7. `07_mublas_sgemm.mu`: 和库实现对照。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_shared_basics.mu` | `__shared__`、动态 shared、block 内可见性 | 把 shared memory 当全局缓存 |
| `02_reduce_shared.mu` | shared reduce、`__syncthreads()` | 忘记同步导致读到旧值 |
| `03_transpose_shared.mu` | tile 读写、padding | shared 访问也可能 bank conflict |
| `04_stencil_constant.mu` | constant memory 适合小型广播只读数据 | 把任意大数组塞进 constant |
| `05_naive_gemm.mu` | 每线程算一个 C 元素 | 只追求正确不算 GFLOPS |
| `06_tiled_gemm.mu` | A/B tile 复用、同步边界 | tile 边界没有处理非整除尺寸 |
| `07_mublas_sgemm.mu` | 自写版和库版对比 | 期望入门 tiled GEMM 接近库性能 |

## 代码阅读抓手

GEMM 阅读重点：

- 每个 thread 负责哪个 `C[row, col]`。
- 每轮 tile 从 global memory 读了哪些 A/B 元素。
- `__syncthreads()` 放在 tile 加载后和下一轮覆盖 shared 前。
- 边界判断是否覆盖非 16/32 整除的矩阵尺寸。

## 高频混淆点

- **shared memory 是手动管理的缓存**: 数据不会自动进 shared, 必须显式从 global load, 用完前还要保证线程同步。
- **`__syncthreads()` 放错位置比不写还危险**: tile 加载后要同步, 下一轮覆盖 shared 前也要确认上一轮计算用完。分支里调用 `__syncthreads()` 要保证 block 内所有线程都能走到。
- **shared 也可能慢**: bank conflict 会让 shared memory 访问串行化。transpose 里 padding 的目的就是改变 bank 映射。
- **constant memory 适合广播, 不适合大数组乱读**: 所有线程读同一个常量很快; 每个线程读不同地址时, 常量缓存优势会下降。
- **GEMM 的 row/col 和库布局容易混**: 手写 kernel 常按 row-major 想, BLAS 接口常按 column-major 语义解释 `lda/ldb/ldc`。对拍小矩阵是最直接的排错方式。
- **入门 tiled GEMM 不该和库硬比**: muBLAS 是高度优化实现。手写版本先用来理解 tile、复用和同步边界。

## 逐示例课文

### 1. `01_shared_basics.mu`：看清 shared memory 的作用域与生命周期

#### 示例目标

这个 smoke test 不追求算法产出，而是把 shared memory 的两个声明形式跑通：静态 shared 在编译期确定大小，动态 shared 在 kernel launch 时由第三个执行配置参数给出大小。重点是建立“它是 block 私有的片上临时存储，不是整个程序的全局缓存”的直觉。

#### 代码结构

`static_shared` 声明 `__shared__ float s[256]`，每个线程写入自己的线程号，再反向读出到 `out`。`dynamic_shared` 使用 `extern __shared__ float s[]`，launch 时传入 `256 * sizeof(float)`，线程写入 `2 * threadIdx.x`，同步后累加到 `out`。主机端先分配一个 256 元素的 device buffer，依次启动两个 kernel。

#### 核心知识点

静态数组的布局和大小由编译器知道，适合固定 tile 或固定临时区；动态数组的字节数由每次 launch 决定，适合运行时才知道的 tile/缓冲区。两者都不是跨 block 共享：每个 block 获得自己的一份存储，block 结束后这份存储的内容和生命周期都结束，下一次 kernel launch 不能依赖它仍然存在。`__syncthreads()` 是 block 内的同步和内存可见性边界；这里必须等全 block 写完，反向读取或累加才不会读到未完成的数据。它不能让不同 block 互相通信。

#### 执行流程

主机分配 `d` 后，静态 kernel 在一个 block 中完成“写 shared → 同步 → 读 shared”；返回后再启动动态 kernel，第三个配置参数为该 block 预留动态 shared 字节数。两个 kernel 都用 `MUSA_CHECK_KERNEL()` 检查启动/执行错误，最后打印完成标记并释放 device 内存。

#### 常见错误与实验

常见错误包括忘记为动态 shared 传字节数、把元素个数误当成字节数、让线程数超过数组长度，或以为 shared 的值能被另一个 block 读取。可实验：把动态 shared 的 launch 大小改小观察越界风险；把同步去掉观察结果是否不稳定；将线程数和数组长度改成不一致并补上显式边界判断。性能实验要记录设备、block 配置和实际测量，不应从这个 smoke test 推断 shared 必然更快。

#### 与本周其他示例的关系

这是后续 reduce、transpose 和 tiled GEMM 的语义前置：`02` 复用动态 shared 做归约，`03` 使用静态二维 shared tile，`06` 用静态 tile 保存 A/B 的分块。`04` 则展示另一种只读存储——constant memory；`05` 的朴素 GEMM提供不使用 shared 的对照。

### 2. `02_reduce_shared.mu`：用 shared 完成 block 内归约

#### 示例目标

把长度为 `N` 的输入先归约成每个 block 一个 partial sum，再由主机把 partial 结果相加。示例目标是理解 shared reduce 的树形数据流、每轮同步的位置，以及非整除输入如何补零，而不是展示一个完整的多级 device reduction 库实现。

#### 代码结构

`reduce_shared` 使用动态 shared `s[]`。线程根据 `blockIdx.x * blockDim.x + threadIdx.x` 计算全局索引，读入一个输入值；越过 `n` 的线程写入 `0.0f`。随后以 `stride = blockDim.x / 2` 逐轮折半，较小线程号把后半段加到前半段，线程 0 将 `s[0]` 写到 `partial[blockIdx.x]`。主机分配输入和 partial 数组，填充全 1，拷回 partial 后用 CPU 得到最终和。

#### 核心知识点

shared 让同一 block 的线程反复交换中间结果，减少对 global 的往返；但每轮读写都必须在 `__syncthreads()` 形成同步边界，否则某线程可能读到上一轮尚未写完的值。该写法要求 block 线程数适合这种二分树（通常为 2 的幂）；如果改用任意线程数，需要重新设计配对和边界。跨 block 的最终归约没有在 kernel 中完成，所以 `partial` 的合并由主机串行执行；这也是结果正确性和总耗时含义的一部分。

#### 执行流程

`N = 1 << 22`、每 block 256 线程，因此 grid 为向上取整后的 block 数，并按 256 个 float 传入动态 shared 大小。kernel 启动后测量 kernel 时间，拷回 `partial`，CPU 累加并打印 `sum`、期望值、kernel 时间和 partial 数。输入全为 1 时，越界补零保证最后一个 block 仍可参加同一棵归约树。

#### 常见错误与实验

常见错误是漏掉加载后的同步、漏掉每轮累加后的同步、未给越界线程补零，或试图在没有额外协调的情况下让一个 block 读取另一个 block 的 shared。可把 `N` 改成不能被 256 整除的值，验证 partial 仍正确；比较不同 block size；再实现第二个 kernel 在 device 上继续归约，区分“kernel 时间”和“端到端时间”。不要只凭 shared 版本就断言一定加速，主机合并、同步和 occupancy 都会影响结果。

#### 与本周其他示例的关系

它把 `01` 中“动态 shared + block 内同步”变成实际算法。`03` 同样先协同加载、同步、再使用 shared，但目标是改变布局；`06` 也有“每轮使用完再覆盖”的双同步边界。`05` 没有 block 内共享归约，可作为访存组织的反例。

### 3. `03_transpose_shared.mu`：用 shared tile 改变读写方向

#### 示例目标

理解矩阵转置为什么适合先连续读入 tile、再交换索引写出，并观察 padding 对 shared bank 映射的影响。示例固定 `TILE = 32`，重点是访问模式和边界，而不是保证某一设备上的绝对速度。

#### 代码结构

kernel 为每个 block 声明 `__shared__ float tile[32][33]`。第一阶段以 `(x, y)` 定位输入，满足边界时把 row-major 的 `in[y * w + x]` 写入 `tile[threadIdx.y][threadIdx.x]`；同步后，block 的 x/y 角色交换，读取 `tile[threadIdx.x][threadIdx.y]`，以 `out[y * h + x]` 写入转置后的 `h × w` 结果。主机使用二维 grid 覆盖输入，并先 warmup，再重复 20 次测平均 kernel 时间。

#### 核心知识点

输入 tile 的协同加载让相邻线程按输入行连续读取，转置后的写出也通过 shared 把原本不友好的方向隔开。第二维多出的 1 个元素是 padding：它改变相邻行的 stride，通常可避免转置读取时一整个 warp 落到相同 bank；这不是“shared 天生无冲突”，也不保证所有硬件/访问模式都得到相同收益。加载完成后的同步是硬边界：任何线程都不能提前读取别人的 tile 数据。非整除尺寸时，读写两侧都分别判断边界；本文件的 `W/H` 恰好整除只是默认实验值，不应成为 kernel 正确性的前提。

#### 执行流程

grid 的 x 方向覆盖宽度、y 方向覆盖高度。每个 block 先把有效输入元素放入 tile，所有线程同步，再用交换后的 block 坐标计算输出位置并写回。第一次启动用于预热，之后的 20 次由事件计时，最后显式设备同步并打印平均时间；代码没有初始化或校验输出，因此正确性实验应自行准备可辨识输入并检查转置关系。

#### 常见错误与实验

常见错误包括把输出索引仍按输入的 `w` 计算、只在加载侧做边界判断、去掉加载后的同步，或误以为 `+1` padding 对所有访问都有效。可将尺寸改为 `W=2050、H=2047`，用 `in[y*w+x] = y*w+x` 检查 `out[x*h+y]`；复制一份不 padding 的版本比较 profiler 的 bank conflict；同时比较朴素转置的连续/跨步访问和端到端时间。结果快慢需以实测为准。

#### 与本周其他示例的关系

它是 `01` 的静态二维 shared 用法和 `02` 的同步模式的结合。`04` 不搬运 tile，而是把小权重放进 constant；`05` 直接按元素计算 GEMM；`06` 将 tile 复用从一次转置扩展到 K 方向多轮乘加。

### 4. `04_stencil_constant.mu`：用 constant memory 广播小型只读参数

#### 示例目标

通过一维五点 stencil 认识 constant memory 的适用边界：权重数组很小、kernel 中许多线程读取相同的只读值时，硬件可能利用广播和 constant cache。这里验证的是符号拷贝和启动链路，不是完整的 stencil 性能结论。

#### 代码结构

文件定义 `__constant__ float c_w[5]`。`stencil5` 为每个线程计算一个位置 `i`，对 `i-2` 到 `i+2` 的输入做加权和，权重从 `c_w[k+2]` 读取。主机在 device 分配输入/输出，用 `musaMemcpyToSymbol(c_w, w, sizeof(w))` 把五个 host 权重写入 constant symbol，再以 256 线程 block 启动一维 grid。

#### 核心知识点

constant memory 的数据由 host 显式拷贝到 device 端符号，kernel 只能读它；它适合小型、只读、线程间有相同地址访问的参数，硬件可以广播同一值。若线程访问不同地址，广播优势会下降；大数组或随机访问不应机械地放进 constant。该 stencil 只对 `2 <= i < n-2` 写输出，因此两端四个元素保持未定义/未写入状态，调用者必须决定边界策略，不能把它当作完整初始化的输出。

#### 执行流程

主机先构造五点权重 `{0.0625, 0.25, 0.375, 0.25, 0.0625}`，申请输入输出，再执行 host-to-constant symbol 拷贝。kernel 内每个有效线程读取五个输入点和同一组权重，完成后检查 kernel 错误并打印完成标记，最后释放两个 global buffer。示例没有初始化输入，也没有校验数值，所以输出标记只说明链路完成。

#### 常见错误与实验

常见错误是把 `musaMemcpyToSymbol` 当普通 device 指针拷贝、把 constant 当作任意大小的全局数组、忽略 `i` 两端边界，或用不同权重访问模式却仍预期广播收益。可初始化输入为常数并验证内部输出为权重和；改写边界为复制、零填充或单独 kernel；比较权重放在 global、constant 和每线程局部变量的版本，并分别记录吞吐和输入规模。不要从单次运行推出 constant 一定更快。

#### 与本周其他示例的关系

它与 shared 的共同点是都需要明确的数据放置策略，但 constant 是 kernel 间可由符号更新、kernel 内只读的参数空间，不是 block 私有 scratchpad。`02/03/06` 主要解决 block 内数据复用与同步，`05` 则提供不依赖这两类片上存储的 GEMM 基线。

### 5. `05_naive_gemm.mu`：建立 row-major GEMM 的正确性和性能基线

#### 示例目标

先用最直接的映射实现 `C = A × B`，明确一个线程对应一个 `C[row, col]`，再用事件计时和 GFLOPS 量化重复 global 读取的代价。它是后续 tiled GEMM 和 muBLAS 对比时的基准，不是最终优化实现。

#### 代码结构

`gemm_naive` 使用二维 grid/block：`blockIdx.y/threadIdx.y` 得到 row，`blockIdx.x/threadIdx.x` 得到 col；有效线程在 `k=0..K-1` 上累加 `A[row*K+k] * B[k*N+col]`，最后写 `C[row*N+col]`。主机以 row-major 方式填充 A 为 1、B 为 2，warmup 后重复 10 次，用事件求平均时间、按 `2*M*N*K` 计算 GFLOPS，并检查 `C[0] = 2*K`。

#### 核心知识点

该文件的 row-major 约定是 A 的 leading dimension 为 K，B 为 N，C 为 N；数学式为 `C[M×N] = A[M×K] × B[K×N]`。每个线程独立从 global 读取 K 个 A 和 K 个 B，同一 block 内不同线程会重复取相同数据，尤其 B 的列方向访问不如连续行访问友好。GFLOPS 是按约定的操作数和 kernel 时间算出的指标，必须和 warmup、重复次数、是否包含拷贝等测量范围一起解释。BLAS 接口常见的 column-major 语义不能直接套用这些索引。

#### 执行流程

主机申请并填充三块 host 矩阵、三块 device 矩阵，拷贝 A/B 后以 `16×16` block 覆盖 M×N。先启动一次并同步预热，再记录事件包围的 10 次 kernel，拷回 C，打印时间、GFLOPS 和首元素，最后释放资源。默认矩阵元素是常数，适合先做形状和结果 sanity check；要验证一般矩阵，应替换输入并加入 CPU 参考结果。

#### 常见错误与实验

常见错误是交换 M/N 的 grid 计算、漏掉 `row < M || col < N` 边界、混淆 `B[k*N+col]` 与 column-major 索引，或把 event 时间和 host-to-device/device-to-host 拷贝时间混为一谈。可把 M/N/K 改成互不相同且不被 16 整除的尺寸，和 CPU 参考逐元素对拍；扫描矩阵大小并记录 GFLOPS；与 `06` 比较 global 读取和同步开销。朴素版的数值正确不代表性能高。

#### 与本周其他示例的关系

它是 `06` 的同一数学问题、同一 row-major 数据布局和同一首元素校验的无 tile 基线；`06` 将重复的 A/B 读取搬入 shared 并在 K 方向复用。`07` 计划把相同问题交给 muBLAS，但库调用和布局参数必须另行确认；`01/02/03/04` 提供构成这些优化的存储与同步概念。

### 6. `06_tiled_gemm.mu`：沿 K 方向装载并复用 A/B tile

#### 示例目标

在不改变 GEMM 数学定义和 row-major 布局的前提下，让一个 block 协同加载 `TS×TS` 的 A/B 子块，并让 block 内线程重复使用它们。学习重点是复用、同步和非整除边界；tile 只是可能改善访存行为的组织方式，不是对所有设备和尺寸的加速承诺。

#### 代码结构

每个 block 声明 `tileA[TS][TS]` 与 `tileB[TS][TS]`，一个线程负责加载一个 A 元素和一个 B 元素。`row = blockIdx.y*TS + threadIdx.y`、`col = blockIdx.x*TS + threadIdx.x` 确定 C 元素；外层循环将 K 切成 `numTiles = (K+TS-1)/TS` 段。每段把 `A[row*K+aCol]` 和 `B[bRow*N+col]` 放入 shared，计算 `TS` 次乘加，再写回有效 C 元素。

#### 核心知识点

一个 tile 被 block 内多个线程消费：`tileA[threadIdx.y][k]` 和 `tileB[k][threadIdx.x]` 让同一 global 元素在计算阶段被复用，从而降低 global 读取需求；shared 读取本身仍可能受 bank conflict、访问模式、寄存器和 occupancy 影响。这里有两个不可互换的同步边界：加载后同步，确保所有 tile 元素就绪；计算后同步，确保没有线程仍在读旧 tile，下一轮才能覆盖 shared。两个边界都必须由整个 block 一致到达，不能把 barrier 放进只有部分线程进入的分支。`TS=16` 还要同时满足 `TS²` 线程数和 shared 容量约束。

#### 执行流程

grid 覆盖 M×N，block 为 `16×16`。每轮先计算本轮 K 片段的 A/B 地址；越过 M、N 或 K 的加载写入 0，这使最后一个不完整 tile 仍可参与固定长度的内层循环。同步后每线程累加 `TS` 项，再同步，进入下一轮；循环完成后只有有效 `(row,col)` 才写 C。主机 warmup，再重复 10 次测 kernel 平均时间，拷回并打印 `C[0]` 与 GFLOPS。

#### 常见错误与实验

常见错误包括只给有效线程写 shared 却不为无效位置清零、遗漏任一同步、把 `tileB` 的行索引写成 col、把 K 不是 TS 整数倍时的越界访问漏掉，以及任意增大 TS 导致线程数超过设备上限或 occupancy 下降。可用 M=37、N=29、K=23 与 CPU 参考对拍；扫描 TS=8/16/32，记录线程数、shared 用量、GFLOPS 和错误情况；比较加 padding 与不 padding 的 profiler 指标。不要把理论复用率直接等同于实际加速，也不要承诺手写 GEMM 接近库性能。

#### 与本周其他示例的关系

它把 `01` 的静态 shared、`03` 的 tile 协同加载和 `02` 的同步边界组合到矩阵乘法；`05` 是直接 global 访问的基线，最适合做同形状对拍。`07` 是 muBLAS 调用骨架/待实测基线入口，后续可用真实库结果观察生产实现中更复杂的分块、寄存器分块和硬件特定优化。

### 7. `07_mublas_sgemm.mu`：muBLAS 调用骨架与待实测基线入口

#### 示例目标

为“朴素 kernel、入门 tiled kernel、官方库实现”提供未来可比的入口，同时认识 SDK/API 可用性本身是实验前提。这个文件只是 muBLAS 调用骨架/待实测基线入口：源码当前只检查 `mublas.h` 是否可包含并打印提示，没有 SGEMM 调用、计时或结果校验，因此尚未建立或完成库性能基线。

#### 代码结构

文件先包含 `musa_runtime.h`，用 `__has_include(<mublas.h>)` 检测头文件；存在时才包含并设置 `HAVE_MUBLAS_HEADER`。主函数固定打印 `M=N=K=1024`。没有头文件时报告缺失并列出确认 include path、补齐 handle/create/SGEMM/destroy 和对比记录的步骤；有头文件时仍只打印“需要依据 SDK API 补齐调用”的提示。也就是说，示例没有分配矩阵、执行 SGEMM 或输出真实性能。

#### 核心知识点

muBLAS 后续可能提供经过库作者优化的 SGEMM 实现，但当前源码只做头文件探测和提示，不能把它称为已经建立的库性能基线。补齐调用前必须确认本机 SDK 的头文件、库名、函数签名、句柄生命周期、数据类型和布局约定。SGEMM 通常涉及 `alpha`、`beta`、leading dimension 以及转置标志；本周 `05/06` 的 A/B/C 是 C 风格 row-major，而 BLAS 接口常按 column-major 解释，因此直接复制索引会得到转置或错误结果。待实测时，库、naive 和 tiled 的比较必须统一矩阵形状、数据类型、编译器与编译选项、warmup、重复次数和计时范围，并加入结果校验；任何性能或相对快慢结论都依赖这些条件和具体设备、尺寸，必须通过实测验证，手写入门 GEMM 不应承诺接近库性能。

#### 执行流程

运行时先打印固定 shape，再走头文件存在性分支。无头文件路径立即返回，告诉学习者如何在有 MUSA SDK 的机器上继续；有头文件路径目前也立即返回，因为真实函数签名需要以本地官方 API 参考为准。可按 README 的 `make optional`（或单独的 `make 07_mublas_sgemm`）构建入口；这仍不会自动产生 SGEMM、计时或校验结果。补齐后才应按“创建 handle → 准备/拷贝矩阵 → 调用 SGEMM → 同步和计时 → 校验结果 → 销毁 handle/释放内存”的顺序实现，并在统一设备、尺寸、编译器和计时条件下分别记录 naive、tiled、muBLAS 的时间和 GFLOPS。

#### 常见错误与实验

常见错误是把“找到 `mublas.h`”误当成“已经链接并能调用库”、忽略库链接参数、handle 未销毁、leading dimension 传错、row-major/column-major 和 transpose 标志不匹配，或只比较一次冷启动时间。可在 SDK 机器上依据官方 API 补齐最小 SGEMM，先用小型非方阵和 CPU 参考验证布局，再对 `1024³` 做 warmup 与多次事件计时；记录三种实现的绝对 GFLOPS、相对比值、编译/链接环境和校验误差。若 SDK 不可用，记录“未测得”比编造库结果更可靠。

#### 与本周其他示例的关系

它承接 `05` 的朴素基线和 `06` 的 tiled 版本，比较时应保持相同的 GEMM 形状与布局语义；同时把 `01` 的资源生命周期意识扩展到库 handle，把 `04` 的“先确认存储/API 适用范围”原则扩展到 SDK。最终目的不是替代前两个 kernel，而是用库实现帮助判断手写优化距离成熟实现还有哪些层次。

## CUDA_Freshman 对照

- `24_shared_memory_read_data`: shared memory 基础。
- `25_reduce_integer_shared_memory`: shared reduce。
- `26_transform_shared_memory`: shared transpose。
- `27_stencil_1d_constant_read_only`: constant/read-only cache。

GEMM 进阶可参考 SGEMM_CUDA: <https://github.com/siboehm/SGEMM_CUDA>。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
