# Pages and MUSA Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish all six weeks of learning notes, exercises, completion criteria, and real MTT S4000 validation evidence inside GitHub Pages.

**Architecture:** Keep Markdown and a JSON validation manifest as canonical sources. A zero-dependency Node generator renders trusted Markdown into committed static HTML, while existing document checks enforce navigation, freshness, allowed validation states, and evidence links. Remote runs produce immutable raw logs that are summarized in the manifest and weekly notes.

**Tech Stack:** Node.js built-ins (`node:test`, `fs`, `crypto`, `child_process`), static HTML/CSS, Bash/SSH, MUSA `mcc`, GitHub Actions.

---

### Task 1: Establish an isolated execution baseline

**Files:**
- Preserve: `docs/superpowers/plans/2026-09-20-memory-basics.md`
- Verify: `scripts/check-docs.test.js`
- Verify: `scripts/check-docs.js`

- [ ] **Step 1: Use the worktree workflow**

Run the `using-git-worktrees` skill. Create an isolated branch from commit `387f3cd` unless the current checkout is already isolated. Never stage the user's untracked `docs/superpowers/plans/2026-09-20-memory-basics.md`.

- [ ] **Step 2: Verify the baseline**

Run:

```bash
node --test scripts/check-docs.test.js
node scripts/check-docs.js
git status --short
```

Expected: 38 tests pass, the checker reports 154 quiz questions, 9 knowledge pages and 136 local links, and the only unrelated path is the preserved user plan when working in place.

### Task 2: Run Week 1 and Week 2 on the MTT S4000

**Files:**
- Create: `validation/raw/2026-10-02-s4000/environment.txt`
- Create: `validation/raw/2026-10-02-s4000/week1/*.log`
- Create: `validation/raw/2026-10-02-s4000/week2/*.log`

- [ ] **Step 1: Capture the immutable environment record**

On the remote checkout at `/root/musa-learning-notes`, run and save:

```bash
{
  date -Iseconds
  git rev-parse HEAD
  uname -a
  mcc --version
  mthreads-gmi
} > /root/musa-validation/environment.txt 2>&1
```

Expected: the record names MTT S4000, MUSA SDK 3.1.0 compiler, driver 2.7.0, and the exact source revision.

- [ ] **Step 2: Build Week 1 targets independently**

For each target below, run `make clean`, build only that target, then execute it with a 60-second timeout. Store combined build/run output and the final exit status in a target-specific log.

```text
01_hello_world
02_thread_index
03_device_info
04_memory_basics
05_error_check
06_async_kernel
```

Expected: ordinary examples exit zero. A nonzero result remains evidence and is not overwritten by a retry.

- [ ] **Step 3: Build Week 2 targets independently**

Use the same procedure for:

```text
01_vector_add_runtime
02_vector_add_pinned
03_vector_add_timer
04_vector_add_unified
05_multi_stream
06_stream_event_dep
07_musa_graph
08_stream_callback
```

Expected: every target has one log containing the compiler command, program output, and explicit build/run exit status.

- [ ] **Step 4: Copy evidence into the local repository**

Use `scp` with the temporary key to copy `environment.txt`, `week1/`, and `week2/` into `validation/raw/2026-10-02-s4000/`. Do not copy binaries.

### Task 3: Run Week 3 through Week 6 and classify constraints

**Files:**
- Create: `validation/raw/2026-10-02-s4000/week3/*.log`
- Create: `validation/raw/2026-10-02-s4000/week4/*.log`
- Create: `validation/raw/2026-10-02-s4000/week5/*.log`
- Create: `validation/raw/2026-10-02-s4000/week6/*.log`

- [ ] **Step 1: Run Week 3**

Build and run independently with a 120-second timeout:

```text
01_warp_divergence
02_reduce_naive
03_reduce_unrolling
04_reduce_shfl
05_nested_hello
06_sum_matrix_2d
07_sum_matrix_1d
```

Expected: unsupported dynamic parallelism or shuffle behavior is preserved as `BUILD_FAIL` or `RUN_FAIL`; it is never converted to `PASS` based on expected output.

- [ ] **Step 2: Run Week 4**

Build and run independently with a 120-second timeout:

```text
01_saxpy_bandwidth
02_offset_access
03_offset_unrolling
04_aos_vs_soa
05_transpose_naive
06_transpose_padded
```

Expected: logs preserve reported time/bandwidth and correctness output without comparing it to undocumented peak values.

- [ ] **Step 3: Run Week 5**

Build and run the six default targets independently. Attempt `07_mublas_sgemm` separately so a missing library does not block the default examples.

```text
01_shared_basics
02_reduce_shared
03_transpose_shared
04_stencil_constant
05_naive_gemm
06_tiled_gemm
07_mublas_sgemm
```

Expected: the optional muBLAS result is classified from the actual compiler/linker output.

- [ ] **Step 4: Probe Week 6 safely**

Build `02_musa_gdb_demo` and `03_error_dump`, but do not run either fault-injection binary in the batch. Run `04_torch_musa_minimal.py` with a 120-second timeout. Attempt `make mccl` and `make custom-op`; classify MCCL execution as `ENV_LIMITED` because only one GPU is available even if compilation succeeds.

Expected: no intentional illegal-address program is labeled `PASS` merely because it produced an expected failure.

- [ ] **Step 5: Copy Week 3–6 evidence locally**

Copy only text logs into `validation/raw/2026-10-02-s4000/week3` through `week6` and verify that no credential, private key, executable, core dump, or device dump is present.

### Task 4: Define and validate the evidence manifest

**Files:**
- Create: `validation/musa-3.1.0-s4000.json`
- Modify: `scripts/check-docs.test.js`
- Modify: `scripts/check-docs.js`

- [ ] **Step 1: Write failing manifest tests**

Add Node tests that use the existing in-memory `runChecker(changes)` fixture and assert failure for an unknown status and for `PASS` without an existing evidence file:

```js
test("rejects an unknown validation status", () => {
  const mutate = text => {
    const data = JSON.parse(text);
    data.weeks[0].targets[0].status = "MAYBE";
    return JSON.stringify(data);
  };
  assert.throws(
    () => runChecker({ "validation/musa-3.1.0-s4000.json": mutate }),
    /unknown validation status MAYBE/,
  );
});

test("requires evidence for passing targets", () => {
  const mutate = text => {
    const data = JSON.parse(text);
    data.weeks[0].targets[0].evidence = "validation/raw/missing.log";
    return JSON.stringify(data);
  };
  assert.throws(
    () => runChecker({ "validation/musa-3.1.0-s4000.json": mutate }),
    /missing validation evidence/,
  );
});
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
node --test scripts/check-docs.test.js
```

Expected: the two new tests fail because manifest validation does not exist.

- [ ] **Step 3: Create the manifest from observed results**

Use this exact schema, filling every target from Tasks 2–3 with observed status, exit codes, summary, and evidence path:

```json
{
  "runId": "2026-10-02-s4000",
  "sourceRevision": "da0c6ee515ebec72bcc79fc7f8e10be9180a5e4b",
  "environment": {
    "gpu": "MTT S4000 48 GiB",
    "musaSdk": "3.1.0",
    "driver": "2.7.0",
    "gpuCount": 1
  },
  "weeks": [
    {
      "week": 1,
      "targets": [
        {
          "name": "01_hello_world",
          "status": "PASS",
          "buildExitCode": 0,
          "runExitCode": 0,
          "summary": "CPU and five GPU threads printed successfully.",
          "evidence": "validation/raw/2026-10-02-s4000/week1/01_hello_world.log"
        }
      ]
    }
  ]
}
```

The example entry may remain `PASS` because it was already observed; all other entries must use their actual results.

- [ ] **Step 4: Implement manifest validation**

Add `checkValidationManifest()` to `scripts/check-docs.js`. It must require all six weeks, reject duplicate targets, allow only `PASS`, `BUILD_FAIL`, `RUN_FAIL`, `ENV_LIMITED`, `NOT_RUN`, require non-empty reasons for non-PASS states, and require every evidence path to remain inside `validation/raw/<runId>/` and exist as a regular file.

Update the baseline assertion in `scripts/check-docs.test.js` so its expected output also contains the manifest summary. Derive the expectation from the fixture:

```js
const validation = JSON.parse(fs.readFileSync(path.join(root, "validation/musa-3.1.0-s4000.json"), "utf8"));
const statuses = validation.weeks.flatMap(week => week.targets).map(target => target.status);
const statusCounts = Object.fromEntries([...new Set(statuses)].sort().map(status => [
  status,
  statuses.filter(candidate => candidate === status).length,
]));
const validationSummary = `validation targets: ${statuses.length} ${JSON.stringify(statusCounts)}`;

test("baseline validates the complete knowledge base", () => {
  assert.deepEqual(runChecker(), [
    "quiz questions: 154",
    "learning materials: ok",
    validationSummary,
    "knowledge pages: 9",
    "local links: 136",
  ]);
});
```

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
node --test scripts/check-docs.test.js
node scripts/check-docs.js
```

Expected: all old and new tests pass and the checker prints a validation target/status summary.

- [ ] **Step 6: Commit evidence and manifest validation**

```bash
git add validation scripts/check-docs.js scripts/check-docs.test.js
git -c commit.gpgsign=false commit -m "test: record MUSA 3.1 validation evidence"
```

### Task 5: Build a zero-dependency Markdown page generator

**Files:**
- Create: `scripts/page-renderer.js`
- Create: `scripts/build-pages.js`
- Create: `scripts/build-pages.test.js`
- Create: `docs/generated/week1-learning.html`
- Create: `docs/generated/week1-exercises.html`
- Create: `docs/generated/week1-records.html`
- Create: corresponding generated pages for Week 2–6

- [ ] **Step 1: Write failing renderer tests**

Create tests for escaping raw HTML, headings, fenced code, lists, task boxes, links and GFM tables:

```js
test("renders repository markdown without allowing raw HTML", () => {
  const html = renderMarkdown(`# Title\n\n<script>alert(1)</script>\n\n- [ ] run\n\n| A | B |\n|---|---|\n| x | y |`);
  assert.match(html, /<h1>Title<\/h1>/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<script>alert/);
  assert.match(html, /type="checkbox" disabled/);
  assert.match(html, /<table>/);
});
```

Add a generator test asserting 18 expected files, source hashes, site navigation and `--check` freshness behavior.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
node --test scripts/build-pages.test.js
```

Expected: FAIL because `page-renderer.js` and `build-pages.js` do not exist.

- [ ] **Step 3: Implement the renderer**

Export these stable functions:

```js
module.exports = {
  escapeHtml,
  renderInline,
  renderMarkdown,
  renderDocument,
};
```

`escapeHtml()` runs before Markdown substitutions. `renderMarkdown()` supports only the constructs covered by tests and never passes raw source HTML through. `renderDocument()` uses the existing `assets/knowledge.css` and `assets/knowledge.js`, adds a source link, and emits `<meta name="source-sha256">` computed from the canonical Markdown bytes.

- [ ] **Step 4: Implement deterministic generation**

`scripts/build-pages.js` maps exactly these sources for weeks 1–6:

```js
const KINDS = [
  ["learning", week => `code/week${week}/learning-notes.md`, "完整学习材料"],
  ["exercises", week => `code/week${week}/exercises.md`, "完整练习"],
  ["records", week => `notes/week${week}.md`, "真机记录"],
];
```

Normal mode writes only changed files. `--check` compares generated bytes with committed files and exits nonzero with `stale generated page: <path>` without writing.

- [ ] **Step 5: Generate pages and verify GREEN**

Run:

```bash
node scripts/build-pages.js
node --test scripts/build-pages.test.js
node scripts/build-pages.js --check
```

Expected: 18 HTML files exist, tests pass, and the freshness check is silent/successful.

- [ ] **Step 6: Commit the generator**

```bash
git add scripts/page-renderer.js scripts/build-pages.js scripts/build-pages.test.js docs/generated
git -c commit.gpgsign=false commit -m "feat: publish complete weekly materials"
```

### Task 6: Add completion criteria and validation pages

**Files:**
- Create: `docs/validation.html`
- Modify: `docs/week1.html`
- Modify: `docs/week2.html`
- Modify: `docs/week3.html`
- Modify: `docs/week4.html`
- Modify: `docs/week5.html`
- Modify: `docs/week6.html`
- Modify: `docs/index.html`
- Modify: `scripts/check-docs.test.js`
- Modify: `scripts/check-docs.js`

- [ ] **Step 1: Write failing navigation and completion tests**

Require the homepage to link `validation.html`, every week page to link its three generated pages, and every week page to contain a `本周通关标准` section with five checklist items. Require validation rows to have a known status and evidence link.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
node --test scripts/check-docs.test.js
```

Expected: FAIL with missing `validation.html` and missing weekly generated-page links.

- [ ] **Step 3: Add weekly completion sections**

Add this structure to each `docs/weekN.html`, replacing the week-specific labels and paths:

```html
<section class="card completion-card" aria-labelledby="completion-title">
  <h2 id="completion-title">本周通关标准</h2>
  <ul class="completion-list">
    <li>能独立回答本周核心问题</li>
    <li>必跑示例通过，或有明确环境限制记录</li>
    <li>完成基础练习和至少一个边界实验</li>
    <li>对应 Week 题库正确率达到 80%</li>
    <li>真机记录包含设备、SDK、命令和结果</li>
  </ul>
</section>
```

Add local links to `generated/weekN-learning.html`, `generated/weekN-exercises.html`, and `generated/weekN-records.html` alongside the existing GitHub source links.

- [ ] **Step 4: Generate the validation overview**

Extend `scripts/build-pages.js` so `docs/validation.html` is rendered from `validation/musa-3.1.0-s4000.json`. Show the fixed environment, counts per status, a table per week, evidence links, and an explicit note that one GPU cannot validate multi-GPU behavior.

- [ ] **Step 5: Update homepage navigation**

Add a `真机验证` link to the main navigation and a homepage card that explains the difference between curriculum completeness and hardware verification status.

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```bash
node scripts/build-pages.js
node --test scripts/build-pages.test.js scripts/check-docs.test.js
node scripts/check-docs.js
```

Expected: all tests pass, all generated links stay inside `docs/`, and the checker includes 28 knowledge pages (home, six weeks, topic, quiz, validation, and 18 generated pages).

- [ ] **Step 7: Commit the learning loop**

```bash
git add docs scripts/build-pages.js scripts/check-docs.js scripts/check-docs.test.js
git -c commit.gpgsign=false commit -m "feat: add weekly completion and validation pages"
```

### Task 7: Reconcile weekly records and project status

**Files:**
- Modify: `notes/week1.md`
- Modify: `notes/week2.md`
- Modify: `notes/week3.md`
- Modify: `notes/week4.md`
- Modify: `notes/week5.md`
- Modify: `notes/week6.md`
- Modify: `README.md`
- Modify: `docs/roadmap.md`

- [ ] **Step 1: Add observed rows to weekly notes**

For each target, add a row containing date `2026-10-02`, `MTT S4000`, `MUSA SDK 3.1.0`, `mcc/clang 14`, backend `musa`, the actual target, result, key measurement/output and evidence path. Keep existing templates below the observed table.

- [ ] **Step 2: Reconcile status language**

Change status labels only from observed evidence:

- Curriculum content: `✅` when learning notes, exercises, Pages and quiz coverage exist.
- Hardware validation: report a separate `PASS/部分受限/未验证` value derived from the manifest.

Do not use a single emoji to imply both curriculum completeness and full hardware support.

- [ ] **Step 3: Regenerate Pages and verify records**

Run:

```bash
node scripts/build-pages.js
node scripts/build-pages.js --check
node --test scripts/build-pages.test.js scripts/check-docs.test.js
node scripts/check-docs.js
```

Expected: generated record pages contain the new rows and no source hash is stale.

- [ ] **Step 4: Commit the reconciled status**

```bash
git add README.md docs/roadmap.md docs/generated notes
git -c commit.gpgsign=false commit -m "docs: publish MUSA hardware validation results"
```

### Task 8: Wire CI and complete verification

**Files:**
- Modify: `.github/workflows/docs-check.yml`
- Modify: `.github/workflows/pages.yml`
- Modify: `scripts/check-docs.test.js`

- [ ] **Step 1: Write a failing workflow assertion**

Add a test that requires both workflows to run the generator freshness check before `node scripts/check-docs.js`:

```js
for (const workflow of [".github/workflows/docs-check.yml", ".github/workflows/pages.yml"]) {
  const text = readText(workflow);
  assert.ok(text.indexOf("node scripts/build-pages.js --check") < text.indexOf("node scripts/check-docs.js"));
}
```

- [ ] **Step 2: Run tests and verify RED**

Run `node --test scripts/check-docs.test.js`.

Expected: FAIL because neither workflow checks generated-page freshness.

- [ ] **Step 3: Update workflows**

Insert this command after unit tests and before the document checker in both workflows:

```yaml
- name: Check generated Pages freshness
  run: node scripts/build-pages.js --check
```

Add `validation/**`, `notes/**`, `code/**/*.md`, and `scripts/build-pages*.js` to relevant path filters so canonical-source changes trigger checks and deployment.

- [ ] **Step 4: Run complete local verification**

Run:

```bash
node scripts/build-pages.js --check
node --test scripts/build-pages.test.js scripts/check-docs.test.js
node scripts/check-docs.js
git diff --check
git status --short
```

Expected: all tests pass, generated pages are current, no whitespace errors exist, and only intended files plus the preserved user plan appear.

- [ ] **Step 5: Inspect representative pages**

Serve `docs/` with a local static server and inspect `index.html`, `week1.html`, one generated learning page, one generated exercise page, one record page, `validation.html`, and `quiz.html` at desktop and narrow width. Confirm readable tables, wrapped code, working navigation, visible focus states, and no raw Markdown syntax leaks.

- [ ] **Step 6: Commit CI enforcement**

```bash
git add .github/workflows/docs-check.yml .github/workflows/pages.yml scripts/check-docs.test.js
git -c commit.gpgsign=false commit -m "ci: verify generated learning pages"
```

- [ ] **Step 7: Request publication authorization**

Report the final commits and test evidence. Ask before pushing the branch or merging to `main`; successful local work does not authorize a remote push.

### Task 9: Remove temporary machine access after acceptance

**Files:**
- No repository files.

- [ ] **Step 1: Provide the exact public-key removal command**

After all remote validation is finished and no rerun is needed, ask the user to run:

```bash
sed -i '/codex-musa-validation-2026-10-02$/d' /root/.ssh/authorized_keys
```

- [ ] **Step 2: Remove the local temporary private key**

After the user confirms remote-key removal, delete `/private/tmp/musa-learning-codex-key-20261002` and its `.pub` file using an explicit, validated path, then report that the temporary access material is gone.
