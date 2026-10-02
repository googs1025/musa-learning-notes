# Pages 全量学习资料与 MUSA 真机验证设计

## 目标

把仓库补齐为一套可在 GitHub Pages 内完成学习、练习和复习的 CUDA/MUSA 入门资料，并在现有 MTT S4000、MUSA SDK 3.1.0 环境中运行各周可执行示例，留下可复核的真实证据。

完成后，学习者应能从 Pages 首页进入六周主线，阅读完整学习材料与练习，查看通关标准和真机状态，再通过题库完成复习。项目不得把未运行、缺少依赖或受单卡环境限制的项目标记为通过。

## 范围

### Pages 内容

- 保留 `code/weekN/learning-notes.md`、`code/weekN/exercises.md` 和 `notes/weekN.md` 作为唯一内容来源。
- 使用构建期生成器把这些 Markdown 转换为静态 HTML，避免手工维护第二份正文。
- 为每周生成学习材料、练习和运行记录页面，并从现有 `docs/weekN.html` 和首页进入。
- 每周页面展示统一通关标准：核心概念、必跑示例、必做练习、题库目标和实测记录。
- 首页增加完整资料入口和真机验证总览。
- 生成页面沿用现有导航、配色、响应式布局和无障碍语义。

### 真机验证

- 基准环境固定记录为 MTT S4000、MUSA SDK 3.1.0、驱动 2.7.0。
- 先运行 Week 1–5 的普通单卡示例，再运行 Week 6 中当前环境能够满足的调试和 `torch_musa` 示例。
- 每个目标记录源码版本、完整命令、编译结果、运行退出码、关键输出和限制原因。
- 状态只允许 `PASS`、`BUILD_FAIL`、`RUN_FAIL`、`ENV_LIMITED`、`NOT_RUN`。
- 多卡 MCCL/P2P、缺失库或 SDK 3.1.0 不支持的功能标记为 `ENV_LIMITED`，不能用 dry-run 或推测替代实测。
- 原始远程日志保存在结构化验证目录，周记录只引用和总结证据。

## 架构

```text
Canonical Markdown
  code/weekN/learning-notes.md
  code/weekN/exercises.md
  notes/weekN.md
            │
            ▼
  scripts/build-pages.js
            │
            ▼
  docs/generated/weekN-*.html
            │
            ├── docs/index.html
            ├── docs/weekN.html
            └── docs/validation.html

Remote MUSA machine
  build + run targets
            │
            ▼
  validation/raw/<run-id>/
            │
            ▼
  notes/weekN.md + validation manifest
```

生成器只处理仓库使用到的 Markdown 子集，包括标题、段落、列表、任务框、引用、代码块、表格、行内代码和链接。生成前先转义不受信任的文本；只允许生成器自身产生的 HTML 结构，避免把任意 Markdown HTML 注入 Pages。

## 页面与导航

每个周次增加三个稳定入口：

- `generated/weekN-learning.html`
- `generated/weekN-exercises.html`
- `generated/weekN-records.html`

现有周页面继续承担精炼课程页的角色，不复制完整正文。页面底部的“本周源码目录”和 GitHub 外链保留，同时增加上述站内入口。生成页提供返回周主页、上一份/下一份材料和完整自测链接。

真机验证总览页按周和目标显示状态，不显示臆测性能。`PASS` 必须能链接到仓库内的原始日志或明确的周记录；其他状态必须带原因。

## 通关标准

每周使用相同结构，但具体目标随主题变化：

1. 能用自己的语言回答本周核心问题。
2. 必跑示例在目标环境得到正确结果，或有明确的环境限制记录。
3. 完成规定的基础练习和至少一个边界实验。
4. 对应 Week 题库正确率达到 80%。
5. `notes/weekN.md` 中存在带日期、设备、SDK、命令和结果的真实记录。

通关状态由证据决定，不根据页面是否存在自动宣称完成。

## 测试与验收

采用测试先行：

- 先扩展 `scripts/check-docs.test.js`，要求生成页、站内入口、通关标准和验证总览存在；确认测试因功能缺失而失败。
- 实现生成器和页面导航后，让新增测试通过。
- 校验生成结果与源 Markdown 的标题、更新时间或内容摘要一致，防止生成物过期。
- 保留现有题库、页面结构和本地链接测试。
- CI 先构建生成页，再运行结构检查，最后上传 `docs/`。
- 本地最终验收包括所有 Node 测试、文档检查、链接检查和生成器幂等性检查。

真机验收要求每个已尝试目标都有状态，且 `PASS` 同时满足编译成功、运行退出码为零和示例自身的正确性条件。仅成功编译不能算 `PASS`。

## 安全和环境约束

- SSH 私钥不进入仓库；验证完成后提示用户删除临时公钥。
- 不升级远程驱动、MUSA SDK 或系统软件。
- 不删除远程用户数据；构建限定在 `/root/musa-learning-notes`。
- 故障注入示例逐个运行并设置超时，避免连续错误影响后续验证。
- 不把用户密码、主机凭据或私钥写入日志和 Pages。

## 不在本次范围内

- 升级 MUSA SDK 3.1.0。
- 伪造多卡环境或性能结果。
- 把 Week 6 的所有工程扩展强行定义为入门必修。
- 重写现有 154 道题或改变站点视觉风格。

## 完成定义

- 六周完整教材、练习和记录均可从 Pages 站内访问。
- 六周均有可量化通关标准。
- 当前单卡环境能运行的示例全部被尝试并分类。
- 真机结果和受限原因写入仓库，且能从 Pages 查看。
- 自动生成、链接、题库和站点结构测试全部通过。
- 线上 Pages 工作流成功部署生成内容。
