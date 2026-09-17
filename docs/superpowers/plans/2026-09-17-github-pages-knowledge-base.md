# GitHub Pages MUSA Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing GitHub Pages quiz into a dual-entry MUSA knowledge base with a six-week learning path, topic navigation, curated code points and questions, a GPU hierarchy article, and the preserved 150-question quiz.

**Architecture:** Keep the site as dependency-free static HTML/CSS/JavaScript under `docs/`. Move the current quiz to `quiz.html`, make `index.html` the knowledge hub, use one shared stylesheet and one small progressive-enhancement script, and give each week and the GPU hierarchy its own readable page.

**Tech Stack:** Semantic HTML5, CSS Grid/Flexbox, vanilla JavaScript, Node.js validation, GitHub Actions Pages deployment.

---

## File Structure

- Modify `docs/index.html`: replace the quiz UI with the knowledge-base landing page.
- Create `docs/quiz.html`: preserve the current quiz and its 150 questions.
- Create `docs/week1.html` through `docs/week6.html`: focused weekly knowledge pages.
- Create `docs/gpu-hierarchy.html`: web article based on `docs/gpu-hierarchy.md`.
- Create `docs/assets/knowledge.css`: shared visual system for the knowledge hub, weekly pages, and topic page.
- Create `docs/assets/knowledge.js`: card search/filter and active navigation enhancement.
- Modify `scripts/check-docs.js`: validate quiz data, required knowledge structure, and local links.
- Modify `.github/workflows/pages.yml`: run the repository checker instead of parsing quiz data from `index.html`.

The weekly HTML files own their educational content. Shared CSS and JavaScript own presentation and lightweight behavior only; they must not contain factual course material.

### Task 1: Migrate the quiz and update continuous validation

**Files:**
- Create: `docs/quiz.html`
- Modify: `scripts/check-docs.js`
- Modify: `.github/workflows/pages.yml`
- Test: `scripts/check-docs.js`

- [ ] **Step 1: Copy the current quiz page before replacing the root page**

Create `docs/quiz.html` as an exact copy of the current `docs/index.html`. Then add a normal anchor near the brand block:

```html
<a class="knowledge-link" href="index.html">← 返回知识库</a>
```

Do not change:

- `const STORAGE_KEY = "musa-learning-quiz-v1";`
- any question ID or question object;
- filtering, answer, review, wrong-answer, shuffle, reset, or progress behavior.

- [ ] **Step 2: Change the quiz checker to read `docs/quiz.html`**

In `scripts/check-docs.js`, change `checkQuizData()` so its first line reads:

```js
const html = readText("docs/quiz.html");
```

Update quiz-related error messages from `docs/index.html` to `docs/quiz.html`.

- [ ] **Step 3: Make the Pages workflow use the repository checker**

Replace the inline `node -e` quiz parser in `.github/workflows/pages.yml` with:

```yaml
      - name: Validate documentation site
        run: node scripts/check-docs.js
```

Keep the existing triggers, artifact upload, and deploy jobs unchanged.

- [ ] **Step 4: Verify the migrated Quiz before replacing the home page**

Run:

```bash
node scripts/check-docs.js
test "$(rg -n 'musa-learning-quiz-v1' docs/quiz.html | wc -l | tr -d ' ')" = "1"
```

Expected: `quiz questions: 150`, `learning materials: ok`, and both commands exit 0.

- [ ] **Step 5: Commit the migration**

```bash
git add docs/quiz.html scripts/check-docs.js .github/workflows/pages.yml
git -c commit.gpgsign=false commit -m "feat: migrate quiz for knowledge base"
```

### Task 2: Build the shared design system and knowledge-base home

**Files:**
- Create: `docs/assets/knowledge.css`
- Create: `docs/assets/knowledge.js`
- Modify: `docs/index.html`

- [ ] **Step 1: Create the shared CSS design system**

Create `docs/assets/knowledge.css` with these tokens and component responsibilities:

```css
:root {
  --bg: #f4f7f8;
  --surface: #ffffff;
  --surface-soft: #eaf4f2;
  --ink: #17202a;
  --muted: #5f6f7f;
  --line: #d9e2e7;
  --nav: #101820;
  --accent: #0f766e;
  --accent-strong: #115e59;
  --code: #111827;
  --shadow: 0 12px 30px rgba(23, 32, 42, 0.08);
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; color: var(--ink); background: var(--bg); line-height: 1.65; }
a { color: var(--accent-strong); }
pre { overflow-x: auto; padding: 16px; border-radius: 12px; color: #e5e7eb; background: var(--code); }
code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
```

Also implement reusable classes used by every knowledge page:

- `.site-header`, `.site-nav`, `.brand`, `.nav-links`
- `.page-shell`, `.hero`, `.hero-actions`
- `.week-grid`, `.topic-grid`, `.card`, `.tag`
- `.article-layout`, `.article-nav`, `.article-content`
- `.knowledge-grid`, `.knowledge-card`, `.callout`
- `.code-card`, `.source-link`
- `.pitfall-list`, `.quiz-preview`, `details`, `summary`
- `.pager`, `.footer`
- `.is-hidden`, `.empty-state`

At `max-width: 760px`, collapse multi-column grids and the article sidebar into one column, wrap navigation, reduce padding, and keep code blocks scrollable.

- [ ] **Step 2: Implement progressive card filtering**

Create `docs/assets/knowledge.js`:

```js
document.addEventListener("DOMContentLoaded", () => {
  const search = document.querySelector("[data-knowledge-search]");
  const cards = [...document.querySelectorAll("[data-search-text]")];
  const empty = document.querySelector("[data-empty-state]");

  if (!search || cards.length === 0) return;

  const applyFilter = () => {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    for (const card of cards) {
      const matched = card.dataset.searchText.toLowerCase().includes(query);
      card.classList.toggle("is-hidden", !matched);
      if (matched) visible += 1;
    }
    if (empty) empty.hidden = visible !== 0;
  };

  search.addEventListener("input", applyFilter);
});
```

Without JavaScript, cards remain visible because `.is-hidden` is only added by the script.

- [ ] **Step 3: Replace `docs/index.html` with the semantic knowledge hub**

Use this document shell:

```html
<!doctype html>
<html lang="zh-CN" data-page-kind="home">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="MUSA GPU 编程六周知识库：核心概念、关键代码、易错点与自测。">
  <title>MUSA GPU 编程知识库</title>
  <link rel="stylesheet" href="assets/knowledge.css">
</head>
<body>
  <header class="site-header">
    <a class="brand" href="index.html">MUSA GPU 编程知识库</a>
    <nav class="site-nav" aria-label="主导航">
      <a href="#weeks">六周学习</a>
      <a href="#topics">专题索引</a>
      <a href="quiz.html">完整自测</a>
    </nav>
  </header>
  <main class="page-shell">
    <section class="hero">
      <p class="tag">MUSA Learning Notes</p>
      <h1>从第一个 kernel 到多卡与框架扩展</h1>
      <p>沿六周路线学习核心概念、关键代码与常见错误，也可以按专题快速查阅。</p>
      <div class="hero-actions">
        <a href="week1.html">从 Week 1 开始</a>
        <a href="quiz.html">进入完整自测</a>
      </div>
    </section>
    <section id="weeks" aria-labelledby="weeks-title">
      <h2 id="weeks-title">六周学习地图</h2>
      <div class="week-grid"></div>
    </section>
    <section id="topics" aria-labelledby="topics-title">
      <h2 id="topics-title">专题索引</h2>
      <div class="topic-grid"></div>
    </section>
    <section id="how-to-use">
      <h2>怎样使用这套知识库</h2>
      <p>初学者按周阅读；复习时按专题定位；学完一节后进入完整自测检查理解。</p>
    </section>
  </main>
  <footer class="footer">MUSA Learning Notes · 内容以官方文档和仓库实测为准</footer>
  <script src="assets/knowledge.js"></script>
</body>
</html>
```

Populate `.week-grid` and `.topic-grid` with the real cards specified in Steps 4 and 5. Do not leave the containers empty in the implementation. The header must link to `index.html`, `#weeks`, `#topics`, and `quiz.html` through the brand and navigation. The hero must retain the `从 Week 1 开始` and `进入完整自测` calls to action.

- [ ] **Step 4: Add the six-week map with exact topics**

Create six `.card` links with `data-search-text`:

| Link | Title | Question | Tags |
|---|---|---|---|
| `week1.html` | Week 1 · 执行模型 | 一个 kernel 怎样启动并落到 GPU 硬件？ | GPU 层级、索引、内存、错误 |
| `week2.html` | Week 2 · 并发与编排 | 多个 GPU 操作怎样排队、计时和重放？ | Stream、Event、Graph、Pinned |
| `week3.html` | Week 3 · 线程协作 | 线程怎样协作完成归约与二维计算？ | Warp、Reduce、Shuffle、2D |
| `week4.html` | Week 4 · 访存性能 | 为什么结果正确但带宽利用率很低？ | Coalescing、AoS/SoA、Transpose |
| `week5.html` | Week 5 · 片上复用 | 怎样用片上存储提高数据复用？ | Shared、Constant、GEMM、muBLAS |
| `week6.html` | Week 6 · 工程系统 | 怎样调试并扩展到多卡和框架？ | GDB、Error Dump、MCCL、torch_musa |

- [ ] **Step 5: Add topic entries and no-result state**

Add topic cards linking to the most relevant page anchors:

- GPU 层级 → `gpu-hierarchy.html`
- Grid / Block / Thread → `week1.html#execution-model`
- Host / Device 内存 → `week1.html#memory-boundary`
- Stream / Event / Graph → `week2.html#streams`
- Warp / Reduce / Shuffle → `week3.html#reduction`
- 合并访存 / Bank Conflict → `week4.html#memory-access`
- Shared Memory / GEMM → `week5.html#gemm`
- 调试 / 多卡 / 框架 → `week6.html#engineering`

Add:

```html
<input type="search" data-knowledge-search aria-label="搜索周次和专题" placeholder="搜索：warp、stream、GEMM、MCCL">
<p class="empty-state" data-empty-state hidden>没有匹配内容，请换一个关键词。</p>
```

- [ ] **Step 6: Verify home behavior and commit**

Run:

```bash
rg -n 'data-page-kind="home"|week[1-6]\.html|gpu-hierarchy\.html|quiz\.html|data-knowledge-search' docs/index.html
rg -n 'DOMContentLoaded|data-search-text|is-hidden' docs/assets/knowledge.js
git diff --check -- docs/index.html docs/assets/knowledge.css docs/assets/knowledge.js
```

Expected: all page links, search hooks and CSS/JS files are present; whitespace check is clean.

Commit:

```bash
git add docs/index.html docs/assets/knowledge.css docs/assets/knowledge.js
git -c commit.gpgsign=false commit -m "feat: add MUSA knowledge hub"
```

### Task 3: Build Week 1 and the GPU hierarchy topic

**Files:**
- Create: `docs/week1.html`
- Create: `docs/gpu-hierarchy.html`
- Source: `docs/gpu-hierarchy.md`
- Source: `code/week1/learning-notes.md`
- Source: `code/week1/02_thread_index.mu`
- Source: `code/week1/04_memory_basics.mu`
- Source: `code/week1/05_error_check.mu`
- Source: `docs/quiz.html`

- [ ] **Step 1: Create Week 1 with the shared article shell**

Use `<html lang="zh-CN" data-page-kind="week">`, link `assets/knowledge.css`, and include the common header. The article navigation must link to these real IDs:

```html
<a href="#goals">本周目标</a>
<a href="#execution-model">核心知识</a>
<a href="#code-points">关键代码</a>
<a href="#pitfalls">注意事项</a>
<a href="#questions">精选题目</a>
```

The top navigation and footer must link to `index.html`, `quiz.html`, `week2.html`, `../code/week1/README.md` only when browsing locally. For GitHub Pages source links, use stable repository URLs such as:

```text
https://github.com/googs1025/musa-learning-notes/blob/main/code/week1/02_thread_index.mu
```

- [ ] **Step 2: Add Week 1 knowledge cards**

Write concise cards covering:

1. `kernel → grid → block → warp → thread` software hierarchy.
2. `GPU → MPC → MPX → MP` hardware hierarchy with a link to `gpu-hierarchy.html`.
3. One block stays on one MP; a warp is the SIMT execution grouping.
4. `threadIdx`, `blockIdx`, `blockDim`, `gridDim` and global indexing.
5. Host/device pointer boundary and explicit copies.
6. launch error versus asynchronous execution error.
7. launch is asynchronous; synchronization establishes completion.

Use cautious device-dependent language for warp width and resource counts.

- [ ] **Step 3: Add three exact Week 1 code patterns**

Include short excerpts derived from the real files:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
    output[idx] = idx;
}
```

```cpp
musaMalloc(&d_data, bytes);
musaMemcpy(d_data, h_data, bytes, musaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_data, n);
musaMemcpy(h_data, d_data, bytes, musaMemcpyDeviceToHost);
```

```cpp
kernel<<<grid, block>>>(d_data, n);
MUSA_CHECK(musaGetLastError());
MUSA_CHECK(musaDeviceSynchronize());
```

Each code card must explain what the pattern proves and link to its actual source file on GitHub.

- [ ] **Step 4: Add Week 1 pitfalls and curated questions**

Pitfalls must include missing boundary checks, host dereference of device pointers, byte-oriented `musaMemset`, trusting GPU `printf` order, and checking only launch errors.

Create `<details>` questions based on IDs:

- `w1-q1`
- `w1-q3`
- `w1-q7`
- `w1-q10`

Copy each question, answer and pitfall from `docs/quiz.html`; do not add scoring behavior.

- [ ] **Step 5: Create `docs/gpu-hierarchy.html` from the approved Markdown**

Use `<html lang="zh-CN" data-page-kind="topic">` and preserve these sections:

- 先看结论
- 一张总图
- 用物流园来理解
- MUSA 的三层硬件组织
- 和 CUDA 怎样对照
- 一个 kernel 的旅行
- 这些层级怎样影响性能
- 常见误区
- 参考资料

The page must contain the exact boundary sentence `编程角色可以类比，物理结构不能逐层翻译。`, have zero `MPE` occurrences, and link back to `week1.html` and `index.html`.

- [ ] **Step 6: Verify and commit Week 1/topic pages**

Run:

```bash
test "$(rg -n 'MPE' docs/gpu-hierarchy.html | wc -l | tr -d ' ')" = "0"
rg -n 'data-page-kind="week"|本周要回答的问题|核心知识|关键代码|注意事项|精选题目|完整自测' docs/week1.html
rg -n 'MPC|MPX|一个 kernel 的旅行|不能逐层翻译' docs/gpu-hierarchy.html
git diff --check -- docs/week1.html docs/gpu-hierarchy.html
```

Commit:

```bash
git add docs/week1.html docs/gpu-hierarchy.html
git -c commit.gpgsign=false commit -m "feat: add Week 1 knowledge pages"
```

### Task 4: Build Week 2 concurrency knowledge page

**Files:**
- Create: `docs/week2.html`
- Source: `code/week2/learning-notes.md`
- Source: `code/week2/02_vector_add_pinned.mu`
- Source: `code/week2/03_vector_add_timer.mu`
- Source: `code/week2/06_stream_event_dep.mu`
- Source: `code/week2/07_musa_graph.mu`
- Source: `docs/quiz.html`

- [ ] **Step 1: Add Week 2 structure and knowledge cards**

Use the shared weekly shell and IDs `goals`, `memory`, `streams`, `code-points`, `pitfalls`, `questions`. Cover:

- pageable versus pinned host memory;
- CPU wall-clock versus GPU event timing;
- stream FIFO ordering and cross-stream independence;
- H2D → kernel → D2H chunk pipelines;
- event-based cross-stream dependencies;
- graph capture/instantiate/replay and the requirement to measure benefit;
- callback completion order and host notification role;
- unified-memory capability and behavior are platform dependent.

- [ ] **Step 2: Add four Week 2 code patterns**

Include compact excerpts for:

```cpp
musaHostAlloc(&h_data, bytes, musaHostAllocDefault);
musaMemcpyAsync(d_data, h_data, bytes, musaMemcpyHostToDevice, stream);
```

```cpp
musaEventRecord(start, stream);
vector_add<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
musaEventRecord(stop, stream);
```

```cpp
musaEventRecord(ready, producer);
musaStreamWaitEvent(consumer, ready, 0);
```

```cpp
musaStreamBeginCapture(stream, musaStreamCaptureModeGlobal);
vector_add<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
musaStreamEndCapture(stream, &graph);
```

Link each pattern to its real GitHub source.

- [ ] **Step 3: Add pitfalls and curated questions**

Pitfalls: pageable memory with async APIs, timing without synchronization/events, accidental device-wide synchronization, assuming callback order across streams, assuming Graph is automatically faster.

Use question IDs `w2-q1`, `w2-q4`, `w2-q8`, `w2-q9`, copying question/answer/pitfall from `docs/quiz.html`.

- [ ] **Step 4: Verify and commit Week 2**

Run the weekly marker search, verify all source URLs include `/code/week2/`, run `git diff --check`, then commit:

```bash
git add docs/week2.html
git -c commit.gpgsign=false commit -m "feat: add Week 2 knowledge page"
```

### Task 5: Build Week 3 execution and reduction knowledge page

**Files:**
- Create: `docs/week3.html`
- Source: `code/week3/learning-notes.md`
- Source: `code/week3/01_warp_divergence.mu`
- Source: `code/week3/02_reduce_naive.mu`
- Source: `code/week3/04_reduce_shfl.mu`
- Source: `code/week3/06_sum_matrix_2d.mu`
- Source: `docs/quiz.html`

- [ ] **Step 1: Add Week 3 knowledge cards**

Use IDs `goals`, `warps`, `reduction`, `mapping`, `code-points`, `pitfalls`, `questions`. Cover divergence, block reduction, unrolling, shuffle, tail bounds, host/two-kernel final reduction, 2D grid/block mapping, and the repository's host-orchestrated nested example boundary.

- [ ] **Step 2: Add four code patterns**

Show:

- a branch condition that groups adjacent lanes versus alternating lanes;
- shared-memory tree reduction with `__syncthreads()`;
- a shuffle-down reduction loop expressed with the target platform's width/group rather than a hardcoded CUDA mask;
- `x/y` calculation and `matrix[y * width + x]` row-major indexing.

Every code card links to the exact Week 3 source file on GitHub.

- [ ] **Step 3: Add pitfalls and four questions**

Pitfalls: hardcoding CUDA warp assumptions, missing synchronization in shared reduction, dropping tail elements, assuming shuffle replaces block-wide synchronization, confusing 1D launch shape with 1D data.

Use `w3-q1`, `w3-q2`, `w3-q7`, `w3-q8` from the quiz.

- [ ] **Step 4: Verify and commit Week 3**

Run weekly markers, source-link checks and diff check, then:

```bash
git add docs/week3.html
git -c commit.gpgsign=false commit -m "feat: add Week 3 knowledge page"
```

### Task 6: Build Week 4 memory-access knowledge page

**Files:**
- Create: `docs/week4.html`
- Source: `code/week4/learning-notes.md`
- Source: `code/week4/02_offset_access.mu`
- Source: `code/week4/04_aos_vs_soa.mu`
- Source: `code/week4/05_transpose_naive.mu`
- Source: `code/week4/06_transpose_padded.mu`
- Source: `docs/quiz.html`

- [ ] **Step 1: Add Week 4 knowledge cards**

Use IDs `goals`, `memory-access`, `layouts`, `transpose`, `code-points`, `pitfalls`, `questions`. Cover effective bandwidth, coalescing, offset/alignment, why unrolling does not repair an address pattern, AoS/SoA address sequences, transpose read/write directions, shared tiles, and bank conflict padding.

- [ ] **Step 2: Add four code patterns**

Show:

- offset access `out[i] = in[i + offset]` with bounds;
- contrasting AoS and SoA field loads;
- naive transpose row-major input/output indices;
- padded shared tile declaration and synchronized load/store phases.

Link each card to the real source.

- [ ] **Step 3: Add pitfalls and four questions**

Pitfalls: reporting only milliseconds rather than effective bandwidth, treating all offsets equally, claiming unrolling fixes coalescing, optimizing only reads in transpose, assuming shared memory is automatically conflict-free.

Use `w4-q1`, `w4-q2`, `w4-q3`, `w4-q8`.

- [ ] **Step 4: Verify and commit Week 4**

Run weekly markers, source-link checks and diff check, then:

```bash
git add docs/week4.html
git -c commit.gpgsign=false commit -m "feat: add Week 4 knowledge page"
```

### Task 7: Build Week 5 on-chip reuse and GEMM knowledge page

**Files:**
- Create: `docs/week5.html`
- Source: `code/week5/learning-notes.md`
- Source: `code/week5/01_shared_basics.mu`
- Source: `code/week5/04_stencil_constant.mu`
- Source: `code/week5/05_naive_gemm.mu`
- Source: `code/week5/06_tiled_gemm.mu`
- Source: `code/week5/07_mublas_sgemm.mu`
- Source: `docs/quiz.html`

- [ ] **Step 1: Add Week 5 knowledge cards**

Use IDs `goals`, `shared`, `constant`, `gemm`, `code-points`, `pitfalls`, `questions`. Cover shared-memory scope/lifecycle, synchronization, constant broadcast, naive GEMM mapping, K-dimension tiling, two barriers per tile iteration, edge tiles, and muBLAS as a measured reference rather than assumed equivalence.

- [ ] **Step 2: Add four code patterns**

Show:

- shared allocation/load/barrier;
- constant-memory declaration and symbol copy;
- naive GEMM row/column accumulation;
- tiled GEMM load → barrier → inner-product → barrier skeleton.

Link to exact Week 5 sources, including muBLAS in the surrounding explanation.

- [ ] **Step 3: Add pitfalls and four questions**

Pitfalls: reading shared data before all writers finish, divergent barriers, missing edge bounds, assuming larger tile is always better, comparing custom GEMM and muBLAS with different inputs/timing/error tolerance.

Use `w5-q1`, `w5-q2`, `w5-q3`, `w5-q9`.

- [ ] **Step 4: Verify and commit Week 5**

Run weekly markers, source-link checks and diff check, then:

```bash
git add docs/week5.html
git -c commit.gpgsign=false commit -m "feat: add Week 5 knowledge page"
```

### Task 8: Build Week 6 debugging, multi-GPU and framework knowledge page

**Files:**
- Create: `docs/week6.html`
- Source: `code/week6/learning-notes.md`
- Source: `code/week6/01_mccl_allreduce.cpp`
- Source: `code/week6/02_musa_gdb_demo.mu`
- Source: `code/week6/03_error_dump.mu`
- Source: `code/week6/04_torch_musa_minimal.py`
- Source: `code/week6/05_torch_musa_custom_op.cpp`
- Source: `docs/quiz.html`

- [ ] **Step 1: Add Week 6 knowledge cards**

Use IDs `goals`, `engineering`, `debugging`, `multi-gpu`, `frameworks`, `code-points`, `pitfalls`, `questions`. Cover asynchronous error reporting, debug symbols and failing instruction location, reproducible error dumps, rank/nranks/device/communicator, AllReduce, torch_musa smoke-test chain, and custom-op registration/dispatch/kernel boundaries.

- [ ] **Step 2: Add four code patterns**

Show compact, source-derived excerpts for:

- MCCL rank/device selection and AllReduce call;
- deliberate illegal address plus synchronization for debugger/error-dump observation;
- torch_musa tensor creation → operation → device-to-CPU validation;
- custom-op registration and MUSA dispatch boundary.

Link every pattern to its exact source.

- [ ] **Step 3: Add pitfalls and five questions**

Pitfalls: blaming the next API for an earlier async failure, omitting environment/reproduction data, mixing rank and device IDs, assuming P2P support without topology checks, validating only the custom kernel and not registration/dispatch.

Use `w6-q2`, `w6-q4`, `w6-q5`, `w6-q7`, `w6-q8`.

- [ ] **Step 4: Verify and commit Week 6**

Run weekly markers, source-link checks and diff check, then:

```bash
git add docs/week6.html
git -c commit.gpgsign=false commit -m "feat: add Week 6 knowledge page"
```

### Task 9: Complete integration, validation and responsive review

**Files:**
- Modify: `docs/quiz.html`
- Modify: `docs/assets/knowledge.css`
- Modify: `scripts/check-docs.js`
- Verify: `docs/index.html`
- Verify: `docs/week1.html` through `docs/week6.html`
- Verify: `docs/gpu-hierarchy.html`
- Verify: `.github/workflows/pages.yml`

- [ ] **Step 1: Style the Quiz knowledge-base link without changing quiz behavior**

Add `.knowledge-link` styling inside `docs/quiz.html`'s existing stylesheet so the link is visible against the dark sidebar, has a focus state, and meets the existing control spacing. Do not import `knowledge.css` into the Quiz; keep its current self-contained presentation.

- [ ] **Step 2: Add required-page, semantic-structure and local-link checks**

Add these constants and helpers to `scripts/check-docs.js`:

```js
const knowledgePages = [
  "docs/index.html",
  "docs/week1.html",
  "docs/week2.html",
  "docs/week3.html",
  "docs/week4.html",
  "docs/week5.html",
  "docs/week6.html",
  "docs/gpu-hierarchy.html",
  "docs/quiz.html",
];

function requireText(relativePath, patterns) {
  requireFile(relativePath);
  const text = readText(relativePath);
  for (const pattern of patterns) {
    if (!text.includes(pattern)) {
      throw new Error(`missing ${JSON.stringify(pattern)} in ${relativePath}`);
    }
  }
  return text;
}

function checkKnowledgePages() {
  requireText("docs/index.html", [
    "data-page-kind=\"home\"",
    "week1.html",
    "week2.html",
    "week3.html",
    "week4.html",
    "week5.html",
    "week6.html",
    "gpu-hierarchy.html",
    "quiz.html",
  ]);

  for (let week = 1; week <= 6; week += 1) {
    const relativePath = `docs/week${week}.html`;
    const text = requireText(relativePath, [
      "data-page-kind=\"week\"",
      "index.html",
      "quiz.html",
      "本周要回答的问题",
      "核心知识",
      "关键代码",
      "注意事项",
      "精选题目",
      "完整自测",
    ]);

    const previous = week > 1 ? `week${week - 1}.html` : null;
    const next = week < 6 ? `week${week + 1}.html` : null;
    if (previous && !text.includes(previous)) throw new Error(`missing previous-week link in ${relativePath}`);
    if (next && !text.includes(next)) throw new Error(`missing next-week link in ${relativePath}`);

    const detailsCount = (text.match(/<details/g) || []).length;
    if (detailsCount < 3) throw new Error(`too few curated questions in ${relativePath}`);

    const sourceCount = (text.match(/github\.com\/googs1025\/musa-learning-notes\/blob\/main\/code\//g) || []).length;
    if (sourceCount < 2) throw new Error(`too few source links in ${relativePath}`);
  }

  requireText("docs/gpu-hierarchy.html", [
    "data-page-kind=\"topic\"",
    "index.html",
    "week1.html",
    "MPC",
    "MPX",
    "一个 kernel 的旅行",
    "不能逐层翻译",
  ]);
  requireText("docs/quiz.html", ["index.html", "musa-learning-quiz-v1"]);
  console.log(`knowledge pages: ${knowledgePages.length}`);
}

function checkLocalLinks() {
  let checked = 0;
  for (const relativePath of knowledgePages) {
    const html = readText(relativePath);
    const hrefs = [...html.matchAll(/href="([^"]+)"/g)].map((match) => match[1]);
    for (const href of hrefs) {
      if (/^(https?:|mailto:|#)/.test(href)) continue;
      const cleanHref = href.split("#")[0].split("?")[0];
      if (!cleanHref) continue;
      const target = path.resolve(path.dirname(path.join(root, relativePath)), cleanHref);
      if (!fs.existsSync(target)) throw new Error(`broken link ${href} in ${relativePath}`);
      checked += 1;
    }
  }
  console.log(`local links: ${checked}`);
}
```

Call `checkKnowledgePages()` and `checkLocalLinks()` after `checkLearningMaterials()`.

The assertions must guarantee:

- home contains links to all six weeks, the topic page and Quiz;
- each week contains `index.html` and `quiz.html`;
- Week 1 links to Week 2, Weeks 2–5 link to both adjacent weeks, Week 6 links to Week 5;
- `gpu-hierarchy.html` links to home and Week 1;
- Quiz links to home;
- every weekly page has at least three `<details>` blocks and two GitHub source links.

- [ ] **Step 3: Run the complete site checker**

Run:

```bash
node scripts/check-docs.js
```

Expected output includes:

```text
quiz questions: 150
learning materials: ok
knowledge pages: 9
local links:
```

and exits 0.

- [ ] **Step 4: Run static content and whitespace assertions**

Run:

```bash
test "$(rg -l 'data-page-kind="week"' docs/week{1,2,3,4,5,6}.html | wc -l | tr -d ' ')" = "6"
test "$(rg -n 'MPE' docs/gpu-hierarchy.html | wc -l | tr -d ' ')" = "0"
test "$(rg -n 'musa-learning-quiz-v1' docs/quiz.html | wc -l | tr -d ' ')" = "1"
git diff --check
```

Expected: every assertion exits 0 and `git diff --check` prints nothing.

- [ ] **Step 5: Serve locally and inspect desktop/mobile layouts**

Run:

```bash
python3 -m http.server 8000 --directory docs
```

Inspect at desktop width and 375px width:

- `/index.html`: navigation wraps, week/topic cards are readable, search hides/shows cards, no-result state appears.
- `/week1.html` and `/week6.html`: article navigation does not obscure content, code scrolls horizontally, `<details>` works, pager is reachable.
- `/gpu-hierarchy.html`: ASCII hierarchy and comparison table remain readable or horizontally scroll.
- `/quiz.html`: original quiz controls still work and the knowledge link is visible.

Stop the server after inspection.

- [ ] **Step 6: Commit final integration**

```bash
git add docs/quiz.html docs/assets/knowledge.css scripts/check-docs.js
git -c commit.gpgsign=false commit -m "test: validate knowledge base pages"
```

- [ ] **Step 7: Verify the full implementation range**

Run:

```bash
node scripts/check-docs.js
git diff --check HEAD~9..HEAD
git status --short
```

Expected: checker passes; diff check is clean; only `.superpowers/` may remain untracked and must not be committed.
