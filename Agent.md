# Agent Notes

本仓库是 MUSA SDK 学习笔记，按 week 组织代码、习题和文档。后续代理或协作者修改时，优先保持“学习日志 + 可运行示例”的风格。

## 项目结构

| 路径 | 用途 |
|---|---|
| `README.md` | 项目入口、当前进度、快速运行说明 |
| `docs/` | 概念文档、路线图、CUDA/MUSA 对照、文章 |
| `docs/index.html` | 项目级学习检测页，GitHub Pages 入口 |
| `notes/` | 每周运行记录和学习笔记 |
| `code/weekN/` | 每周 MUSA 示例、习题和构建文件 |
| `code/case/` | 跨 week 的专题手撸 case，例如 attention / flash kernel |
| `code/leetgpu/easy/` | LeetGPU Easy 题的 MUSA 移植版 |
| `code/include/musa_common.h` | 通用检查宏和计时工具 |

## 修改原则

1. 代码示例优先保持短小、可读、能独立运行。
2. `.mu` 示例沿用三段式注释：`PART I` 知识点或题面，`PART II` 代码，`PART III` Q&A 或坑点。
3. Week 主线练习放 `code/weekN/exercises.md`；刷题索引放 `code/weekN/question-bank/`；快速复习材料可放 `code/weekN/concept-review.md`。
4. 性能数字不要伪造。需要实测的表格可以留 `?`，并说明记录到对应 `notes/weekN.md`。
5. MUSA 与 CUDA 很像，但不要默认性能经验完全相同。尤其注意 MUSA warp size 通常按 128 讨论。
6. 新增知识点、示例、case、概念文档或复习材料时，必须同步评估是否需要给 `docs/index.html` 增加对应测试题；如果不增加题目，需要在提交或 PR 描述里说明原因。

## 构建与运行

常用构建方式：

```bash
cd code
cmake -B build -DMUSA_PATH=/usr/local/musa
cmake --build build -j
```

单周快速运行：

```bash
cd code/week1
make
./01_hello_world
```

Mac 本地没有 `mcc` 时，可以本地编辑、远端运行：

```bash
./scripts/musa.sh run 01_hello_world
```

## 文档风格

- 面向初学者，优先解释“为什么会错”和“怎么验证”。
- 中文标点和英文 API 名混排可以接受；API、路径、命令用反引号。
- 不把 `docs/` 的长解释重复塞进代码注释，代码注释只保留运行当前示例需要的最小知识。
- 新增题目或复习材料时，明确它属于 Week 1/2/3/4/5/6 哪个阶段，避免学习路线混乱。
- `docs/index.html` 的题目要覆盖新增知识点的关键易错点，优先写成可以自测的选择题或短答题，并补齐 `answer`、`pitfall`、`source` 字段。

## README 与学习笔记职责

- `README.md` 面向学习者，只保留项目定位、学习路线、章节重点、入口链接和最小启动方式。
- `Agent.md` 面向代理和协作者，记录本文件中的维护约束、构建方式、文档同步规则和验证要求。
- `docs/roadmap.md` 保存完整的周次路线和文件清单，不要把完整课程表复制回 README。
- `code/weekN/README.md` 说明该周的示例、运行方式和产出；`learning-notes.md` 解释概念和阅读顺序。
- `notes/` 保存真实运行结果、错误记录、截图和官方文档摘录；官方内容必须保留原始链接、版本号和整理日期。

## 官方文档摘录

当前主线参考 MUSA SDK v5.2.0 编程指南：

<https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/>

整理官方资料时：

1. 优先记录对当前学习路线有直接帮助的概念、限制、API 和性能检查方法。
2. 使用自己的中文解释和短例子，不大段复制官方原文。
3. 官方图示可以作为学习材料保存，但要保留来源页面、资源 URL 和 SDK 版本。
4. 如果官方示例包含特定硬件、架构或版本前提，必须在笔记中明确标注，不要推广成所有 MUSA 设备都适用的结论。

## Git 与验证

- 提交前运行 `git diff --check`。
- 文档改动应检查 Markdown 链接、图片路径和代码块是否闭合。
- MUSA 示例在没有 MUSA SDK 的 Mac 上不能声称“编译通过”；应明确说明只做了静态检查或远程验证。
- 不要把用户已有的未提交修改混入无关 PR；提交前按目标路径检查 staged diff。
