# GPU Structure Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a beginner-friendly MUSA GPU hierarchy guide that explains `MPC → MPX → MP`, relates it cautiously to CUDA, and connects the hardware hierarchy to kernel scheduling and performance.

**Architecture:** Keep the detailed explanation in one focused document, `docs/gpu-hierarchy.md`, and add small navigation links from the existing overview documents. Separate official facts, teaching analogies, and architecture-dependent details so readers do not mistake MUSA and CUDA components for strict equivalents.

**Tech Stack:** Markdown, ASCII diagrams, repository-local links, Node.js documentation checker, shell-based link/content assertions.

---

## File Structure

- Create `docs/gpu-hierarchy.md`: owns the complete hierarchy explanation, analogy, MUSA/CUDA comparison, execution path, and performance implications.
- Modify `README.md`: adds the new guide to the beginner reading list.
- Modify `docs/concepts.md`: points the existing short hardware overview to the deeper MUSA-specific guide and adds it to further reading.
- Modify `docs/glossary.md`: defines MP, MPC, and MPX and links to the full guide.
- Do not modify `docs/index.html`, weekly lessons, or kernel examples; they are outside the approved scope.

### Task 1: Establish documentation baseline

**Files:**
- Test: `scripts/check-docs.js`
- Inspect: `README.md`
- Inspect: `docs/concepts.md`
- Inspect: `docs/glossary.md`

- [ ] **Step 1: Run the existing documentation checker**

Run:

```bash
node scripts/check-docs.js
```

Expected output contains both:

```text
quiz questions:
learning materials: ok
```

- [ ] **Step 2: Verify the new guide and navigation do not already exist**

Run:

```bash
test ! -e docs/gpu-hierarchy.md
! rg -n "gpu-hierarchy\.md|MUSA Processor Cluster|MUSA Processor eXecution engine" README.md docs/concepts.md docs/glossary.md
```

Expected: exit status 0 and no output. If either assertion fails, inspect the existing content before continuing and preserve any user-authored material.

### Task 2: Create the GPU hierarchy guide

**Files:**
- Create: `docs/gpu-hierarchy.md`

- [ ] **Step 1: Create the title, reading contract, and two-layer overview**

Add the following opening content:

````markdown
# GPU 结构层次：从 MUSA 的 MPC、MPX、MP 到 CUDA

> 这篇文档先用“物流园”建立直觉，再回到严格的硬件与编程模型。
> MUSA 和 CUDA 的公开术语并非严格一一对应；文中的 CUDA 对照只用于帮助理解。

## 先看结论

- 软件层级回答“程序怎样组织工作”：`kernel → grid → block → warp → thread`。
- 硬件层级回答“工作在哪里执行”：MUSA 公开架构可概括为 `GPU → MPC → MPX → MP → 计算单元`。
- 一个 block 会完整地驻留在一个 MP 上；MP 再把 block 中的线程按 warp 组织和执行。
- CUDA 中最接近 MP 编程角色的是 SM，但 MPC、MPX、MP 与 GPC、SM sub-partition、SM 不能逐级画等号。

## 一张总图

```text
Host CPU
  └── launch kernel
        ↓
MUSA GPU 前端：接收、排队并分发工作
  └── MPC × N
      └── MPX × N
          └── MP × N
              ├── warp 的管理与执行
              ├── FP / INT / SFU / TCE 等计算单元
              └── 局部存储与缓存
```

图中的 `× N` 表示数量由具体架构决定。前端只采用通用描述，不绑定未经公开指南确认的内部模块名称。
````

- [ ] **Step 2: Add the logistics-park analogy and immediately state its limits**

Add a section with this mapping:

````markdown
## 用物流园来理解

| GPU 概念 | 物流园比喻 | 真正含义 |
|---|---|---|
| kernel | 一整批订单的处理规则 | 在 GPU 上执行的函数 |
| grid | 本次订单总清单 | 一次 launch 产生的全部 block |
| block | 不可拆散的一箱任务 | 整体驻留在同一个 MP 上的线程组 |
| MPC | 物流园分区 | 包含多个 MPX 的较高层硬件分组 |
| MPX | 一组相邻车间 | 包含多个 MP 的执行引擎分组 |
| MP | 真正接活的车间 | block 驻留并执行的主要处理器 |
| warp | 同一节拍工作的班组 | SIMT 调度和执行分组 |
| thread | 班组中的工人 | 处理一个或一小部分数据的逻辑线程 |

这个比喻只解释“包含关系”和“工作逐层落下去”。它不代表硬件真的按物流规则排队，也不能用来推断缓存一致性、互连拓扑或精确调度算法。
````

- [ ] **Step 3: Explain MPC, MPX, and MP with sourced architecture examples**

Add these facts and boundaries:

```markdown
## MUSA 的三层硬件组织

### MPC：MUSA Processor Cluster

MPC 是多个 MPX 的上层分组。以公开编程指南中的 MP_10 为例，一颗 GPU 包含 4 个 MPC，每个 MPC 共享一组 L2 缓存资源；MP_21 示例把 MPC 数量扩展到 8。数字只说明这两个架构，不能推广到所有 MUSA GPU。

### MPX：MUSA Processor eXecution engine

MPX 位于 MPC 与 MP 之间。MP_10 示例中，每个 MPC 包含 2 个 MPX，每个 MPX 包含 2 个 MP；同一 MPX 内的 MP 共享部分 L1 数据缓存和指令缓存。

### MP：MUSA Processor

MP 是执行 SIMT 工作的主要处理器。线程块会被分配给可用 MP，并受该 MP 的寄存器、局部存储及最大驻留 block/线程数量约束。MP 内含浮点、整数、特殊函数等执行资源；MP_21 还在公开示例中加入了 TCE。具体单元数量和容量必须查目标架构文档。
```

Then add an MP_10 example diagram showing `4 MPC × 2 MPX × 2 MP = 16 MP`, and an MP_21 note showing `8 MPC × 2 MPX × 2 MP = 32 MP`. Label both as architecture examples, not universal formulas.

- [ ] **Step 4: Add the cautious CUDA comparison**

Add this table:

```markdown
## 和 CUDA 怎样对照

| MUSA 概念 | CUDA 中帮助理解的概念 | 可以怎样理解 | 为什么不能画等号 |
|---|---|---|---|
| MPC | GPC | 都是多个下级处理器的较高层组织 | 缓存、互连和工作分发边界由各自架构定义 |
| MPX | SM sub-partition 的组织思想 | 都体现执行资源还会继续分组 | MPX 位于多个 MP 之上，SM sub-partition 位于一个 SM 内部，层级并不相同 |
| MP | SM | block 在这里驻留，warp 在这里被管理并执行 | 执行单元数量、warp 宽度、缓存和调度实现不同 |
| warp | warp | 都是 SIMT 线程分组 | 宽度和同步原语以目标设备与 SDK 为准 |

最稳妥的记法是：**编程角色可以类比，物理结构不能逐层翻译。**
```

- [ ] **Step 5: Add the software-to-hardware execution path**

Add this sequence and explanation:

````markdown
## 一个 kernel 的旅行

```text
CPU 提交 kernel
  → GPU 前端接收工作
  → grid 中的 block 等待可用处理器
  → 一个完整 block 被分配到一个 MP
  → block 中的线程被组成多个 warp
  → 调度逻辑从就绪 warp 中选择工作
  → FP、INT、访存等管线执行指令
  → warp 等待数据时，MP 尝试运行其他就绪 warp
```

这里有两个不同边界：

- **block 是资源分配和协作边界**：同一 block 的线程共享片上存储并可进行 block 内同步。
- **warp 是 SIMT 执行分组**：同一 warp 的线程执行同一条指令流，分支分化会降低有效利用率。

block 的先后顺序和落到哪个 MP 通常不是程序可依赖的语义。正确的 kernel 不应假设不同 block 会同时运行或按编号顺序运行。
````

- [ ] **Step 6: Add practical performance implications and misconceptions**

Cover exactly these points:

```markdown
## 这些层级怎样影响性能

### 1. block 大小与尾部 warp

block 线程数不是 warp 大小的整数倍时，最后一个 warp 会有一部分 lane 长期空闲。优先通过设备属性或编译目标确认 warp 大小，不要把某个平台的常数机械搬到另一个平台。

### 2. 驻留资源与 occupancy

一个 block 使用的寄存器和片上存储越多，同一 MP 能同时驻留的 block/warp 往往越少。更多就绪 warp 通常更有利于在某个 warp 等待内存或依赖时隐藏延迟。

### 3. occupancy 不是最终目标

高 occupancy 只说明驻留程度，不保证计算管线、内存带宽或数据复用效率都高。应把 occupancy 与 profiler 中的等待原因、带宽和指令吞吐一起判断。

## 常见误区

- `grid/block/thread` 是软件组织，不是芯片上固定焊接的物理层级。
- MPC 不等于 GPC、MPX 不等于 SM sub-partition、MP 也不等于某代 CUDA SM 的全部细节。
- “前端调度器”是方便描述工作接收与分发的通用说法，不能据此杜撰内部模块名称。
- 某一代架构的 MPC 数、缓存容量和执行单元数量不能推广到所有 MUSA GPU。
- occupancy 越高不代表 kernel 一定越快。
```

- [ ] **Step 7: Add official references and related repository links**

End with direct official sources and local links:

```markdown
## 参考资料

- [摩尔线程 MUSA 编程指南：硬件架构](https://docs.mthreads.com/en/musa-sdk/musa-sdk-doc-online/history_version/rc4.3/programming_guide/Chapter02/)
- [摩尔线程 MUSA 编程指南：GPU 并行计算](https://docs.mthreads.com/musa-sdk/version-4.3.x/programming_guide/Chapter01/)
- [NVIDIA CUDA Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [NVIDIA Nsight Compute Profiling Guide：Streaming Multiprocessor](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## 接下来读什么

- [`concepts.md`](concepts.md)：SIMT、线程索引、内存层次和同步。
- [`cuda-vs-musa.md`](cuda-vs-musa.md)：MUSA / CUDA API 和迁移差异。
- [`musa-cuda-pitfalls.md`](musa-cuda-pitfalls.md)：索引、同步和 warp 相关陷阱。
```

- [ ] **Step 8: Verify the guide's required and forbidden content**

Run:

```bash
rg -n "MPC|MPX|MP：|物流园|一个 kernel 的旅行|occupancy|不能画等号" docs/gpu-hierarchy.md
test "$(rg -n "MPE" docs/gpu-hierarchy.md | wc -l | tr -d ' ')" = "0"
git diff --check -- docs/gpu-hierarchy.md
```

Expected: the first command finds every teaching section, the MPE assertion exits 0, and `git diff --check` produces no output.

- [ ] **Step 9: Commit the standalone guide**

```bash
git add docs/gpu-hierarchy.md
git -c commit.gpgsign=false commit -m "docs: explain MUSA GPU hierarchy"
```

### Task 3: Integrate navigation and glossary entries

**Files:**
- Modify: `README.md:55-61`
- Modify: `docs/concepts.md:30-57`
- Modify: `docs/concepts.md:236-241`
- Modify: `docs/glossary.md:3-4`
- Modify: `docs/glossary.md:80-96`
- Modify: `docs/glossary.md:162-166`

- [ ] **Step 1: Add the guide to the README learning list**

Insert after `docs/concepts.md`:

```markdown
- [`docs/gpu-hierarchy.md`](docs/gpu-hierarchy.md)：MUSA 的 MPC / MPX / MP 层级、CUDA 近似对照与 kernel 执行路径。
```

- [ ] **Step 2: Link the short concepts overview to the detailed guide**

After the sentence `block 是调度的边界,warp 是执行的边界。`, add:

```markdown

> 这里用 CUDA 常见的 SM 心智模型做快速入门。MUSA 的 `MPC → MPX → MP` 物理层级、CUDA 近似对照和完整执行路径见 [`gpu-hierarchy.md`](gpu-hierarchy.md)。
```

Add this as the first item under `## 延伸阅读`:

```markdown
- [`gpu-hierarchy.md`](gpu-hierarchy.md) — MUSA 的 MPC / MPX / MP 硬件层级与 CUDA 近似对照
```

- [ ] **Step 3: Add glossary definitions**

Update the opening note to include the hierarchy guide:

```markdown
> 想看完整心智模型,看 [`concepts.md`](concepts.md);想理解 MPC / MPX / MP,看 [`gpu-hierarchy.md`](gpu-hierarchy.md);想查 API,看 [`cuda-vs-musa.md`](cuda-vs-musa.md)。
```

Under `## M`, add the entries in alphabetical order around `Managed Memory`, `MCCL`, and `MTLink`:

```markdown
**MP (MUSA Processor)** — MUSA 中执行 SIMT 工作的主要处理器，线程块在 MP 上驻留并被组织为 warp 执行。可用 CUDA SM 帮助理解，但两者的具体资源和实现不等同。

**MPC (MUSA Processor Cluster)** — 包含多个 MPX 的较高层硬件分组。数量和共享资源取决于具体 MUSA 架构。

**MPX (MUSA Processor eXecution engine)** — 位于 MPC 与 MP 之间的执行引擎分组，一个 MPX 包含多个 MP，并可共享部分缓存资源。
```

Add this under `## 还想看?` after the concepts link:

```markdown
- MUSA 硬件层级与 CUDA 近似对照 → [`gpu-hierarchy.md`](gpu-hierarchy.md)
```

- [ ] **Step 4: Verify every local navigation link**

Run:

```bash
rg -n "gpu-hierarchy\.md" README.md docs/concepts.md docs/glossary.md
test "$(rg -l "gpu-hierarchy\.md" README.md docs/concepts.md docs/glossary.md | wc -l | tr -d ' ')" = "3"
test -f docs/gpu-hierarchy.md
```

Expected: links appear in all three integration files, the file count assertion equals 3, and the target file exists.

- [ ] **Step 5: Commit the navigation and glossary changes**

```bash
git add README.md docs/concepts.md docs/glossary.md
git -c commit.gpgsign=false commit -m "docs: link GPU hierarchy guide"
```

### Task 4: Run final documentation verification

**Files:**
- Verify: `docs/gpu-hierarchy.md`
- Verify: `README.md`
- Verify: `docs/concepts.md`
- Verify: `docs/glossary.md`
- Test: `scripts/check-docs.js`

- [ ] **Step 1: Run the repository documentation checker**

Run:

```bash
node scripts/check-docs.js
```

Expected output contains `quiz questions:` and ends with `learning materials: ok`.

- [ ] **Step 2: Check Markdown whitespace and accidental placeholders**

Run:

```bash
git diff --check HEAD~2..HEAD
! rg -n "TBD|TODO|待补充|严格对应|完全等价" docs/gpu-hierarchy.md README.md docs/concepts.md docs/glossary.md
```

Expected: both commands exit 0 with no output.

- [ ] **Step 3: Confirm the committed file scope**

Run:

```bash
git diff --name-only HEAD~2..HEAD
```

Expected output contains exactly:

```text
README.md
docs/concepts.md
docs/glossary.md
docs/gpu-hierarchy.md
```

- [ ] **Step 4: Inspect final status**

Run:

```bash
git status --short
```

Expected: no tracked implementation changes remain. The existing untracked `.superpowers/` visual-companion directory may still appear; do not add it to the implementation commits.
