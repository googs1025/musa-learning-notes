# Week 6 学习材料

Week 6 是 MUSA 特色收尾：多卡通信、调试、错误 dump、torch_musa 和 custom op。它和 CUDA_Freshman 的重合度不高，更适合以 MUSA 官方文档和本地环境实测为准。

## 阅读顺序

1. `01_mccl_allreduce.cpp`: 理解多卡通信的最小 AllReduce 骨架。
2. `02_musa_gdb_demo.mu`: 用故意 illegal address 学调试入口。
3. `03_error_dump.mu`: 学会复现、定位和记录 error dump。
4. `04_torch_musa_minimal.py`: 验证 PyTorch 到 MUSA 的最小链路。
5. `05_torch_musa_custom_op.cpp`: 看 custom op 注册边界。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_mccl_allreduce.cpp` | communicator、rank、device、collective | 单卡思维直接套多卡 |
| `02_musa_gdb_demo.mu` | 编译调试符号、断点、非法地址定位 | 只看最终错误码不看触发位置 |
| `03_error_dump.mu` | error dump 复现和归档 | 不记录环境导致问题不可复现 |
| `04_torch_musa_minimal.py` | `torch_musa` 设备检查、模型迁移 | 只测 import 不测实际算子 |
| `05_torch_musa_custom_op.cpp` | C++/PyTorch/MUSA 边界 | 把 custom op 当普通 Python 函数 |

## 逐示例课文

### 1. `01_mccl_allreduce.cpp`：从 rank 到 AllReduce

#### 示例目标

把多卡通信拆成可检查的资源关系：一个进程（或启动器分配的一个工作单元）先确定自己的 `rank`、参与总数 `nranks` 和本地 `device`，再在对应设备上创建 stream 和 MCCL communicator，最后让所有 rank 以一致顺序进入同一个 collective。这个文件是结构骨架，不假设当前机器有多卡、MCCL 头文件或可运行的多进程环境。

#### 代码结构

- `CHECK_MUSA` 统一检查运行时调用；命令行参数分别给出 `rank`、`nranks` 和 `device`，默认值只方便阅读骨架输出。
- `__has_include(<mccl.h>)` 将代码分成两条路径：找不到头文件时打印真实运行所需的准备步骤；找到头文件时设置 MUSA device、创建 stream，并提示还需要补齐 SDK 相关的 unique id 交换、`mcclCommInitRank` 和 `mcclAllReduce`。
- 因而当前分支中“能编译”最多说明运行时/头文件路径可见；并没有由这段程序完成一次实际 AllReduce。

#### 核心知识点

`rank` 是通信域中的逻辑身份，`0..nranks-1` 各自代表一个参与者；它不必等于物理卡号。`device` 是该进程当前绑定的 MUSA 设备，单机简单启动时可以令 `device == rank`，但多机或 launcher 场景必须由 local rank 到本机 device 做映射。`communicator` 是由所有 rank 共同加入的通信上下文，保存拓扑、成员和通信状态；它不是 stream，也不是 buffer。`collective`（如 AllReduce）要求所有参与 rank 对同一 communicator 按一致顺序参与，结果才有定义。真正实验还要让每个 rank 拥有相配套的 device、stream 和 input/output buffer，并通过 out-of-band 方式分发同一个 MCCL unique id。

#### 执行流程

先确认 MUSA SDK、`mccl.h`、MCCL 库和 GPU 数量，再准备一份 unique id，并通过 out-of-band 方式分发给所有 rank，确保所有 rank 使用同一份 id。进程启动后读取 rank 配置，调用 `musaSetDevice(device)`，创建该设备上的 stream；随后各 rank 初始化同一通信域，在各自 buffer 上发起 `mcclAllReduce`，最后同步并检查结果，才谈得上记录 algbw、busbw 和拓扑。`MCCL_DEBUG=INFO` 可作为拓扑诊断线索，但日志本身不等于 collective 成功。

#### 常见错误与实验

常见错误包括把 global rank 直接当 device id、不同 rank 使用了不同 unique id、漏掉某个 rank、collective 顺序不一致，以及在错误 device 上创建 stream 或 buffer。可先用不同参数运行骨架，观察 `rank/nranks/device` 映射；然后在有条件的机器上故意交换两个 rank 的 device，比较初始化/通信错误。真正的 AllReduce 实验应同时核对每个 rank 的输入、输出、同步结果和通信日志，不能只凭程序启动或 MCCL 头文件存在下结论。

#### 与本周其他示例的关系

它建立的是“设备和异步执行资源如何组织”的多卡视角。02、03 继续处理同一类异步设备错误，但把问题缩小到单个 kernel 和错误证据；04 将设备选择和同步放进 PyTorch；05 则进一步讨论框架如何把 Python 调用接到 C++ 扩展。

### 2. `02_musa_gdb_demo.mu`：用调试符号定位非法地址

#### 示例目标

练习从 kernel 触发点追到异步错误暴露点：故意制造越界写，再用带调试符号的构建和 MUSA GDB 观察源代码、线程索引和同步位置。它的目的不是提供一个正确 kernel，而是建立可重复的定位入口。

#### 代码结构

`write_oob` 根据 `blockIdx.x * blockDim.x + threadIdx.x` 得到 `i`，却没有 `if (i < n)` 边界检查。主函数只分配 `1024` 个 float，却启动 `8 * 256 = 2048` 个线程；`musaGetLastError()` 检查发射配置，`musaDeviceSynchronize()` 暴露预期的异步 illegal address。`MUSA_CHECK` 在错误处打印错误码、文件和行号后退出。

#### 核心知识点

非法地址是设备端访问了无效或越界地址；这里的根因是线程索引超过分配范围，而不是 `musaDeviceSynchronize()` 本身访问了错误内存。设备 kernel 往往异步执行，所以 launch 返回成功不代表 kernel 已完成或正确；错误可能在后续 API（本例是 synchronize）才被报告。GDB 调试符号（通常来自 `-g`，并配合不妨碍定位的优化设置）让断点、源代码行、变量和调用栈具备映射，不能把“带 `-g` 编译”误认为已经定位了设备端错误。

#### 执行流程

默认 Makefile 使用 `-O2`，且不保证带有 `-g`，因此不能直接假定其产物适合源码级调试。应先使用带 `-g`、较低优化级别（如 `-O0`/`-Og`，具体以本机 SDK 支持为准）的 debug 编译参数或配置生成程序，再直接运行记录错误出现在哪个 API；随后在 MUSA GDB 中对 `main`、kernel launch 后的检查点或同步点设断点，单步核对 `N`、`threads`、`blocks` 和索引计算。结合编译器/调试器对设备 kernel 的支持，观察越界访问附近的线程状态；必要时再用更小的 grid 缩短复现。最终要把源代码行、错误码、launch 配置和同步位置一起保存。

#### 常见错误与实验

不要只看最终错误码：它可能是异步错误在同步点的集中暴露，也不要把 `musaGetLastError()` 成功当作 kernel 正确。可做两组对照：给 kernel 加上 `if (i < n)` 后重新运行，比较错误是否消失；再改变 blocks 或 N，验证“线程数超过分配元素数”这一因果关系。调试符号缺失、优化过强、错误状态未清理或在错误的 host 行设断点，都会让定位结果产生误导。

#### 与本周其他示例的关系

02 关注“在哪里触发/在哪里暴露”，03 关注“如何把同一故障变成可归档证据”。01 中 communicator/stream 的异步错误也可能遵循类似延迟暴露规律；04 的同步调用同样是实际算子链路中的错误检查点。

### 3. `03_error_dump.mu`：把故障变成可复现记录

#### 示例目标

在 02 的越界写基础上，练习 Error Dump 的复现、采集和归档：不仅知道程序失败，还能让别人凭环境、命令、源码和 dump/log 判断是否为同一个问题。

#### 代码结构

该文件与 02 基本共享同一故障模型：分配 `N=1024` 个 float，却启动 2048 个线程，`write_oob` 无边界检查，并在 `musaDeviceSynchronize()` 处预期看到异步 illegal address。差异在于实验重点是开启 Error Dump 后检查 SDK 生成的 dump/log 文件，而不是把 GDB 断点作为主要产物。

#### 核心知识点

Error Dump 是错误现场的诊断证据，可能包含设备、kernel、地址/状态和运行时上下文；具体字段和开关以本机 MUSA SDK 版本为准。复现质量取决于“同一输入/launch 配置 + 同一软件硬件环境 + 完整错误输出 + 原始 dump/log”。环境记录至少应包括 MUSA SDK、驱动/容器镜像、GPU 型号和卡数、编译命令、程序参数、相关环境变量、时间戳、错误码和 dump 路径。生成文件不应被截断、改名到无法对应命令，或只摘录一行日志。

#### 执行流程

先建立独立输出目录并记录 `pwd`、源码版本和环境信息；按 SDK 文档开启 Error Dump，编译并运行程序，保留 stdout/stderr。确认同步点报告错误后，检查 dump/log 是否生成且时间与本次运行对应；再把命令、环境、完整输出和文件清单打包归档。若程序没有失败或没有生成 dump，应记录“未复现/未生成”及原因，而不是把编译成功或程序启动成功当成 dump 验证。

#### 常见错误与实验

常见问题是忘记开启 dump、输出目录不可写、只记录错误码而不记录 SDK/驱动、复用旧 dump，或者把 02 的 GDB 日志和本例的 Error Dump 混为一种证据。可对照运行：一次不开 dump、一次开启 dump；再修复边界检查后运行，确认正确版本应不再触发相同故障。两次实验都要记录独立时间戳和文件清单，避免误认旧文件为新现场。

#### 与本周其他示例的关系

它把 02 的定位过程产品化为故障报告，也为 01 的多卡通信问题提供归档模板。04/05 若实际算子或扩展失败，同样应沿用“命令—环境—完整输出—复现条件”的记录方式，而不是只报告 import 或编译结果。

### 4. `04_torch_musa_minimal.py`：确认 torch_musa 的实际算子链路

#### 示例目标

从 Python 侧检查 PyTorch 是否能使用 MUSA，并完成一次最小的设备 tensor、elementwise 算子、同步和回 CPU 取样链路。它是 smoke test，不是性能测试，也不是完整的 torch_musa 兼容性验证。

#### 代码结构

脚本先导入 `torch` 和 `torch_musa`，打印 PyTorch 版本及 `torch.musa.is_available()`。只有设备可用时才创建 `device="musa"` 的随机 tensor，执行 `x.relu()`，调用 `torch.musa.synchronize()`，再将一个元素拷回 CPU 打印。`import torch_musa` 的作用是加载并注册 MUSA 后端；后续 tensor/算子/同步才是实际链路的一部分。

#### 核心知识点

链路可以理解为：Python API → PyTorch dispatcher/设备类型 → torch_musa 注册的实现 → MUSA runtime/device kernel → synchronize → 拷回 CPU。`is_available()` 只说明当前环境可用性检查通过；import 成功只说明 Python 包能加载。即使 `relu()` 这个最小算子成功，也不能推出所有 dtype、shape、模型层、混合精度、通信或异步错误路径都已验证。样例中的 CPU 拷回标量用于结果可观察性，不适合作为性能数据。

#### 执行流程

先记录 Python、torch、torch_musa、驱动、SDK、GPU 型号和环境变量，再运行脚本。检查后端可用性、MUSA tensor 的 device、算子输出是否合理、同步是否报错以及 CPU 拷回是否完成；若不可用，应保留明确输出并继续调查环境，而不是删掉条件分支或把 import 成功写成完整验证。进一步验证应增加确定性输入、多个 dtype/shape、典型模型算子和错误检查。

#### 常见错误与实验

常见错误包括只执行 `import torch_musa`、只打印 `is_available()`、忘记同步、把单个 `relu` 的成功当成框架全链路通过，以及把样例取样结果拿来做 benchmark。可将 `x.relu()` 换成一个明确可核对的算子组合，比较同步前后的行为；再测试不同 shape，并记录哪一层由哪个后端实现。任何“通过”结论都应说明测试输入、算子和同步边界。

#### 与本周其他示例的关系

04 把 01 的 device/stream 异步思维提升到 PyTorch 抽象层；02/03 提供底层 kernel 错误的定位和证据方法。当 04 的算子链路失败时，05 帮助判断是否涉及扩展注册边界；若错误发生在设备执行阶段，则回到 02/03 的同步、GDB 和 dump 工作流。

### 5. `05_torch_musa_custom_op.cpp`：看清 custom op 的注册边界

#### 示例目标

识别 Python、PyTorch C++ extension、MUSA 实现三者的边界，并理解一个 custom op 从 C++ 函数暴露到 Python 名称的最小路径。当前代码只提供可替换的绑定骨架，不宣称已经实现或验证了 MUSA kernel。

#### 代码结构

`identity_musa` 接收 `torch::Tensor`，用 `TORCH_CHECK` 要求 contiguous，然后调用 `x.clone()` 返回新 Tensor；这仍是 PyTorch 张量操作，不是自定义 `__global__` MUSA kernel。`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 将 C++ 函数注册为 Python 扩展中的 `identity_musa` 名称。注释所说的“先验证 extension 编译链路”与“替换成真正 MUSA kernel”是两个阶段，不能混为一谈。

#### 核心知识点

Python 侧负责加载扩展并调用导出的模块名；pybind11 负责 Python/C++ 函数边界；`torch::Tensor` 和 `TORCH_CHECK` 属于 PyTorch C++ API；真正的 MUSA custom kernel、设备 dispatch、stream 语义、dtype/布局检查和编译链接则属于实现层。仅有 `m.def` 注册意味着 Python 能看到一个 C++ 入口，不意味着 MUSA kernel 被注册、被 dispatch、在 MUSA 上执行，甚至不意味着返回 tensor 位于 MUSA。要形成完整算子，还需明确设备检查、实现选择、同步/错误传播、构建依赖和 Python 调用验证。

#### 执行流程

先按本机 torch_musa/extension 文档配置头文件、库、编译器和 ABI，构建扩展；然后从 Python 加载生成模块，传入 contiguous 和 non-contiguous tensor，分别观察 `identity_musa` 的返回及 `TORCH_CHECK` 错误。若扩展确实改为 MUSA kernel，还要核对 kernel launch、当前 device/stream、输入输出 device、同步后的数值结果和错误路径。编译通过只证明编译/链接阶段完成，Python 能 import 只证明模块加载边界完成，二者都不是完整算子验证。

#### 常见错误与实验

常见错误是把 `clone()` 当成自定义 MUSA 实现、忽略 contiguous 约束、只测模块 import、把 `PYBIND11_MODULE` 注册名与 Python 文件名混淆，或在 CPU tensor 上调用却没有清晰的设备契约。可做三个实验：传入 contiguous MUSA tensor 检查返回 tensor 的 device 与数值；传入 non-contiguous tensor 验证 `TORCH_CHECK`；再对照一个真正包含 MUSA kernel/dispatch 的版本，比较构建依赖、launch 和同步检查项。

#### 与本周其他示例的关系

05 是 04 的下一层：04 从 Python 验证已有 torch_musa 后端，05 解释新增 C++ 入口如何被 Python 看见。它仍依赖 01 中对 device、stream 和异步错误的理解；若真正 kernel 触发 illegal address，则使用 02 的 GDB 定位和 03 的 Error Dump 归档。

## 代码阅读抓手

本周每个例子都要记录环境：

- MUSA SDK 版本。
- GPU 型号和卡数。
- 驱动/容器镜像。
- torch 和 torch_musa 版本。
- 触发命令和完整错误输出。

Week 6 的价值不只是跑通代码，而是形成可复现的调试记录。

## 高频混淆点

- **rank 不是 device id**: 分布式/多卡代码里 global rank、local rank、device id 是三件事。单机多卡常用 local rank 选择本机 device。
- **当前 device 是线程局部状态**: 多线程或一个线程管理多卡时, 每次分配、拷贝、建 stream 前都要确认 `musaSetDevice`。
- **communicator、stream、buffer 要按卡配套**: A 卡的 buffer 不能拿到 B 卡 stream 上随便用。AllReduce 这类 collective 还要求每个 rank 参与顺序一致。
- **只看 import 不算框架链路跑通**: `import torch_musa` 成功只能说明 Python 包能加载; 还要实际创建 MUSA tensor、跑算子、同步检查结果。
- **调试记录比猜错误码更重要**: 多卡/框架问题必须记录 SDK、驱动、容器、GPU 数、rank 映射、命令和完整错误输出, 否则很难复现。

## CUDA 对照

CUDA_Freshman 基本不覆盖本周主题。可以只把 NVIDIA cuda-samples 的多卡、调试、profiling 示例作为概念对照，MUSA 实现以官方文档和本仓库代码为准。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
