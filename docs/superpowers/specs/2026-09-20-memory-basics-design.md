# 全局、共享与常量内存基础内容设计

日期：2026-09-20

## 目标

为刚进入 Week 4/5 的学习者，在已有内存层级速查之后补上一段可直接用于读 kernel 的基础教材。读者完成后应能区分 global、shared、constant memory 的作用域与访问方式，并能根据数据复用和访问模式作出初步放置选择。

## 范围与边界

本次只修改概念教材与 Quiz，不新增 `.mu` 可执行文件，不改变 Week 4/5 的实验主线，也不产生性能结论。

- 主教材：在 `docs/concepts.md` 的“内存层级”后补充一个小节。
- 自测：在 `docs/quiz.html` 的 `QUESTIONS` 中新增四题，题目来源都指向 `docs/concepts.md`。
- 不写死 shared/constant 容量、bank 数、warp 宽度或周期；这些取决于设备和 SDK。
- 示例采用 MUSA/CUDA 相近的 kernel 语法，并标为“阅读示意”，不声称可直接在所有 SDK 上编译。

## 教材结构

新小节标题为“全局、共享与常量内存：先判断数据该放哪里”。内容按以下顺序组织：

1. **三类内存对照表**：比较作用域、读写方式、典型用途和典型误区。
2. **短示例**：
   - `global_add`：每个线程从 global 读取输入、写回输出，强调相邻线程地址。
   - `block_sum`：先装载到 `__shared__`，再以 `__syncthreads()` 完成一次 block 内协作，强调它不能跨 block。
   - `scale`：从 `__constant__` 读取一个小型只读参数，强调相同地址读取的广播条件。
3. **选型流程**：先判断是否跨 block/跨 kernel 保存，再判断是否小型只读且访问相同地址，最后判断是否由同 block 多次复用；其余默认 global。
4. **读 kernel 检查清单**：明确指针来源、谁读写、是否需要同步、相邻线程地址和边界保护。
5. **通向课程**：global 指向 Week 4，shared/constant 指向 Week 5，并说明 Week 3 是 shared 同步的前置。

## 示例约束

每段示例少于 15 行，只展示一种核心行为：

- `global_add` 必须保留 `if (i < n)`，避免把边界保护误教为可选项。
- `block_sum` 必须让所有 block 内线程一致经过 barrier，避免在条件分支中使用 barrier；示例只产出 block partial，明确最终合并需要额外步骤。
- `scale` 的 constant 只存单个标量，以避免读者把“常量数组索引”与“广播访问”混为一谈。

## Quiz

新增四道选择题，覆盖最常见的错误判断：

1. `__syncthreads()` 只同步同一个 block，不能协调不同 block。
2. global memory 中相邻线程访问相邻元素更有利于合并访问；正确性不等于高带宽。
3. shared memory 的价值来自同一个 block 对同一份数据的复用，而不是自动缓存全部 global 数据。
4. constant memory 适合小型只读且同一批线程经常读取相同地址的参数；每线程不同地址的访问不应假定有广播收益。

所有题目补齐 `id`、`deck`、`type`、`question`、`choices`、`answerIndex`、`answer`、`pitfall` 和 `source`。题目归入 Week 4/5 关联的 deck，避免把课程阶段混淆。

## 验收

1. 新内容不与 Week 4/5 讲义重复大段实验说明，且明确链接到两周的学习材料。
2. 代码块、表格和流程在 Markdown 中可读，示例不含设备无关的性能数字。
3. `node scripts/check-docs.js` 通过，确认题目结构、来源路径和本地链接有效。
4. `git diff --check` 通过。
