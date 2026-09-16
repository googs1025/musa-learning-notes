# GitHub Pages MUSA 知识库设计

日期：2026-09-17

## 目标

把仓库已有的六周学习材料整理成可在 GitHub Pages 浏览的知识库。知识库既支持按 Week 1–6 顺序学习，也支持按 GPU 层级、执行模型、内存、并发、归约、访存、GEMM、调试、多卡与框架等专题查阅。

每个周次页提炼核心知识、关键代码、常见错误和代表性题目。现有 150 道题的页面继续承担完整自测职责，不在知识页复制整套题库。

## 用户与使用方式

主要面向两种使用场景：

1. 初学者从 Week 1 按顺序学习到 Week 6。
2. 已学过的读者按专题快速查询概念、代码模式和注意事项。

知识库首页同时提供“六周学习地图”和“专题索引”，避免只能按一种方式查找内容。

## 页面架构

```text
GitHub Pages 根地址
└── docs/index.html                  知识库首页
    ├── docs/week1.html              执行模型与第一个 kernel
    ├── docs/week2.html              Stream、Event 与 Graph
    ├── docs/week3.html              归约、Shuffle 与线程协作
    ├── docs/week4.html              访存、布局与 Bank Conflict
    ├── docs/week5.html              Shared Memory、GEMM 与 muBLAS
    ├── docs/week6.html              调试、多卡与框架扩展
    ├── docs/gpu-hierarchy.html      GPU 层级专题网页入口
    └── docs/quiz.html               完整 150 题自测
```

`docs/gpu-hierarchy.md` 继续作为仓库内 Markdown 原文。`docs/gpu-hierarchy.html` 将其核心内容组织为适合 GitHub Pages 阅读的专题页，并从首页、Week 1 和相关专题入口访问。

现有 `docs/index.html` 的 Quiz 功能迁移到 `docs/quiz.html`。题库内容和 `localStorage` key `musa-learning-quiz-v1` 保持不变，避免已有答题进度丢失。

## 导航模型

所有知识库页面使用一致的顶部导航：

- 知识库首页
- 六周学习地图
- 专题索引
- 完整自测

周次页额外提供：

- 上一周 / 下一周
- 本周源码目录
- 本周练习
- 与本周相关的专题页
- 去完整 Quiz

Quiz 页面增加“返回知识库”入口，但保留现有题库筛选、模式、搜索、统计和答题交互。

## 首页结构

`docs/index.html` 使用双入口知识地图：

1. 首屏说明仓库目标，并提供“从 Week 1 开始”和“进入完整自测”。
2. 六周学习地图，以卡片展示每周要回答的问题、核心主题和页面入口。
3. 专题索引，至少包括 GPU 层级、执行模型、内存、并发、归约、访存、矩阵乘法、调试、多卡和框架扩展。
4. 使用建议，说明顺序学习、专题查阅和 Quiz 自测怎样配合。

首页提供轻量的文本搜索或专题筛选，只检索知识库卡片和入口，不在浏览器中构建全文搜索引擎。

## 周次页内容模板

每个 `weekN.html` 使用相同结构：

1. 本周要回答的问题。
2. 4–7 张核心知识卡。
3. 2–4 个关键短代码片段，并链接真实源码。
4. 3–5 个高频错误、验证方法或性能注意点。
5. 3–5 道精选题，使用原生 `<details>` 展开答案与易错点。
6. 延伸阅读、源码、练习、完整 Quiz、上一周和下一周入口。

知识页用于理解和快速复习，因此代码片段保持短小，只展示索引、同步、内存访问或 API 组合的关键部分。完整程序始终链接回 `code/weekN/` 中的实际文件。

## 六周内容边界

| 周次 | 核心知识 | 关键代码考点 |
|---|---|---|
| Week 1 | GPU/MUSA 层级、SIMT、grid/block/thread、Host/Device、错误与异步 | 全局索引、边界判断、内存拷贝、错误检查、同步 |
| Week 2 | pinned/统一内存、stream、event、graph、callback | 异步拷贝、跨流依赖、GPU 计时、多流流水线 |
| Week 3 | warp divergence、reduce、unroll、shuffle、2D grid、动态并行 | 归约同步、warp 原语、二维索引、子 kernel |
| Week 4 | 合并访存、offset、AoS/SoA、transpose、bank conflict | 地址模式、stride、shared padding、转置 |
| Week 5 | shared/constant memory、naive/tiled GEMM、muBLAS | tile 加载、同步、数据复用、GEMM 索引 |
| Week 6 | MUSA GDB、Error Dump、MCCL、torch_musa、自定义算子 | rank/device/stream、通信器、扩展边界、排错流程 |

每一条事实优先来自仓库现有的 `code/weekN/README.md`、`learning-notes.md`、`exercises.md` 和示例源码。性能数字只在仓库已有实测记录并带清晰环境说明时引用；否则使用定性描述。

## 精选题策略

每周从现有题库挑选 3–5 道代表题，覆盖：

- 本周最重要的概念边界；
- 一段关键代码的含义；
- 一个常见错误或错误假设；
- 一个验证或性能判断问题。

知识页用 `<details>` 展示题目、答案和易错点，不实现评分和进度保存。完整答题、复习、错题与随机模式继续由 `quiz.html` 负责。精选题文案与现有 Quiz 保持一致或明确标明是提炼版。

## GPU 层级专题

`docs/gpu-hierarchy.html` 根据 `docs/gpu-hierarchy.md` 呈现：

- `GPU → MPC → MPX → MP` 总图；
- 物流园比喻及其边界；
- MUSA 与 CUDA 的近似对照；
- kernel 从 launch 到 MP 执行的路径；
- block、warp、occupancy 对实际编程的影响；
- 官方参考资料。

专题页不引入 MPE，不将 MUSA 与 CUDA 的物理层级写成严格等价。

## 视觉与交互

知识库延续现有 Quiz 的视觉语言：深色导航、青绿色强调色、浅色内容面板。页面需满足：

- 桌面宽屏下有清晰的导航和内容层级；
- 小屏下导航与卡片单列排列；
- 代码块可横向滚动；
- 链接、按钮和 `<details>` 可通过键盘操作；
- 不依赖外部字体、图标库或 JavaScript 框架。

首页筛选使用少量原生 JavaScript。问答展开使用 `<details>`，在禁用 JavaScript 时仍可阅读主要内容。

## 文件职责

- `docs/index.html`：知识库首页与卡片筛选。
- `docs/quiz.html`：现有 Quiz 页面和原题库数据。
- `docs/week1.html` … `docs/week6.html`：六周知识内容。
- `docs/gpu-hierarchy.html`：GPU 层级专题。
- `docs/assets/knowledge.css`：首页、周次页和专题页的共享样式。
- `docs/assets/knowledge.js`：首页卡片搜索/筛选和少量渐进增强交互。
- `scripts/check-docs.js`：Quiz 数据、知识库页面、关键结构和本地链接检查。
- `.github/workflows/pages.yml`：PR 时校验页面，main 推送时构建并部署 `docs/`。

知识库不引入 npm、静态站点生成器或新的构建步骤。

## 数据和迁移

Quiz 页面迁移遵循：

1. 复制现有 `docs/index.html` 到 `docs/quiz.html`。
2. 保留 `QUESTIONS` 数据、题目 ID、筛选逻辑和 `musa-learning-quiz-v1`。
3. 在 Quiz 导航中增加返回知识库的链接。
4. 用新的知识库首页替换 `docs/index.html`。
5. 更新脚本和 workflow，使题库检查读取 `docs/quiz.html`。

由于 URL 从根页变为 `quiz.html`，已有浏览器 localStorage 仍属于同一个 GitHub Pages origin，进度可继续读取。

## 错误处理与降级

- 首页搜索无结果时显示清晰提示和“清除筛选”操作。
- JavaScript 失败时，所有周次和专题卡片仍显示，用户仍可通过普通链接导航。
- 本地链接检查失败时阻止 PR 校验通过。
- Quiz 数据缺字段、ID 重复或选择题答案索引无效时继续由检查脚本报错。

## 验收

实现完成后验证：

1. `node scripts/check-docs.js` 通过，并报告 150 道 Quiz 题目、六个周次页和本地链接检查结果。
2. `docs/index.html` 能进入 Week 1–6、GPU 层级专题和 Quiz。
3. 六个周次页都包含目标、核心概念、关键代码、注意事项、精选题和延伸入口。
4. 所有代码链接、周次导航和专题链接均指向存在的文件。
5. `docs/quiz.html` 保留 150 道题、原有题目 ID 和 `musa-learning-quiz-v1`。
6. `docs/gpu-hierarchy.html` 不出现 `MPE`，MUSA/CUDA 对照包含非严格等价说明。
7. 页面在 375px 手机宽度和常见桌面宽度下不存在关键内容遮挡或不可访问操作。
8. GitHub Pages workflow 在 pull request 上完成校验；合并到 `main` 后部署 `docs/`。
9. `git diff --check` 无空白错误，提交不包含 `.superpowers/` 临时内容。

## 不在本次范围

- 全文搜索引擎或搜索索引服务；
- 用户账户、云端进度同步或评论系统；
- 自动从 Markdown 生成 HTML 的构建工具；
- 将全部 150 道题复制到知识页；
- 修改 MUSA 示例代码或六周课程的技术结论。

