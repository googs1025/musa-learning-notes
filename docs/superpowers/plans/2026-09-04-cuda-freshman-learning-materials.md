# CUDA Freshman Learning Materials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a consistent learning-material layer for every week and introduce a separate CUDA example mapping area for CUDA_Freshman-inspired MUSA rewrites.

**Architecture:** Keep `code/weekX/` as the MUSA mainline and add one focused `learning-notes.md` per week. Keep external CUDA source tracking in `code/cuda-freshman/` and a cross-week mapping in `docs/cuda-example-map.md`, with MUSA rewrites placed in `external-cases/` under the relevant week.

**Tech Stack:** Markdown documentation, existing MUSA `.mu` examples, GitHub source references.

---

### Task 1: Add Cross-Week Learning Notes

**Files:**
- Create: `code/week1/learning-notes.md`
- Create: `code/week2/learning-notes.md`
- Create: `code/week3/learning-notes.md`
- Create: `code/week4/learning-notes.md`
- Create: `code/week5/learning-notes.md`
- Create: `code/week6/learning-notes.md`

- [x] **Step 1: Write each week's learning notes**

Each file uses the same sections: reading order, core concepts, code-reading handles, and external CUDA references.

- [ ] **Step 2: Review for consistency**

Run:

```bash
for f in code/week{1,2,3,4,5,6}/learning-notes.md; do sed -n '1,40p' "$f"; done
```

Expected: every file starts with `# Week N 学习材料` and includes the same high-level structure.

### Task 2: Add CUDA_Freshman Mapping

**Files:**
- Create: `docs/cuda-example-map.md`
- Create: `code/cuda-freshman/README.md`

- [x] **Step 1: Map CUDA_Freshman directories to the MUSA roadmap**

The mapping groups Tony-Tan/CUDA_Freshman directories by week and records whether the MUSA route already covers the idea or needs an external case.

- [ ] **Step 2: Check links and relative paths**

Run:

```bash
rg "cuda-example-map|CUDA_Freshman|SGEMM_CUDA" docs code/cuda-freshman
```

Expected: links point to `../../docs/cuda-example-map.md` from `code/cuda-freshman/README.md` and to GitHub URLs for external sources.

### Task 3: Wire The New Materials Into Week READMEs

**Files:**
- Modify: `code/week1/README.md`
- Modify: `code/week2/README.md`
- Modify: `code/week3/README.md`
- Modify: `code/week4/README.md`
- Modify: `code/week5/README.md`
- Modify: `code/week6/README.md`
- Modify: `README.md`

- [ ] **Step 1: Add learning-material links**

Add a short `学习材料` section to each week README pointing to `learning-notes.md`, `exercises.md`, and external cases.

- [ ] **Step 2: Add project-level navigation**

Add links from the root `README.md` to `docs/cuda-example-map.md` and `code/cuda-freshman/README.md`.

### Task 4: Verification

**Files:**
- Check: all Markdown files changed in this plan

- [ ] **Step 1: Check changed files**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only documentation and directory changes from this task plus pre-existing user edits remain.
