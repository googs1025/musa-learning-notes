#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { escapeHtml, renderDocument } = require("./page-renderer");

const defaultRoot = path.resolve(__dirname, "..");
const KINDS = [
  ["learning", (week) => `code/week${week}/learning-notes.md`, "完整学习材料"],
  ["exercises", (week) => `code/week${week}/exercises.md`, "完整练习"],
  ["records", (week) => `notes/week${week}.md`, "真机记录"],
];

function generatedTarget(repositoryPath) {
  let match = repositoryPath.match(/^code\/week([1-6])\/learning-notes\.md$/);
  if (match) return `week${match[1]}-learning.html`;
  match = repositoryPath.match(/^code\/week([1-6])\/exercises\.md$/);
  if (match) return `week${match[1]}-exercises.html`;
  match = repositoryPath.match(/^notes\/week([1-6])\.md$/);
  if (match) return `week${match[1]}-records.html`;
  return null;
}

function linkResolver(sourcePath) {
  return (href) => {
    if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(href)) return href;
    const [pathname, suffix = ""] = href.split(/(?=[?#])/u, 2);
    const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), pathname));
    const generated = generatedTarget(resolved);
    if (generated) return `${generated}${suffix}`;
    return `https://github.com/googs1025/musa-learning-notes/blob/main/${resolved}${suffix}`;
  };
}

function validationDocument(manifest) {
  const targets = manifest.weeks.flatMap((week) => week.targets);
  const statuses = ["PASS", "BUILD_FAIL", "RUN_FAIL", "ENV_LIMITED", "NOT_RUN"];
  const counts = Object.fromEntries(statuses.map((status) => [
    status,
    targets.filter((target) => target.status === status).length,
  ]));
  const evidenceUrl = (evidence) => `https://github.com/googs1025/musa-learning-notes/blob/main/${evidence}`;
  const weekTables = manifest.weeks.map((week) => `      <section aria-labelledby="week${week.week}-title">
        <h2 id="week${week.week}-title">Week ${week.week}</h2>
        <div class="table-wrap"><table>
          <thead><tr><th>目标</th><th>状态</th><th>结果</th><th>证据</th></tr></thead>
          <tbody>${week.targets.map((target) => `<tr>
            <td><code>${escapeHtml(target.name)}</code></td>
            <td><strong>${escapeHtml(target.status)}</strong></td>
            <td>${escapeHtml(target.summary)}${target.reason ? `<br><small>${escapeHtml(target.reason)}</small>` : ""}</td>
            <td><a href="${evidenceUrl(target.evidence)}">${escapeHtml(target.evidence)}</a></td>
          </tr>`).join("")}</tbody>
        </table></div>
      </section>`).join("\n");

  return `<!doctype html>
<html lang="zh-CN" data-page-kind="validation">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>真机验证 · MUSA GPU 编程知识库</title>
  <link rel="stylesheet" href="assets/knowledge.css">
  <script src="assets/knowledge.js" defer></script>
</head>
<body>
  <header class="site-header"><nav class="site-nav" aria-label="主导航">
    <a class="brand" href="index.html">MUSA GPU 编程知识库</a>
    <ul><li><a href="index.html#weeks">每周路径</a></li><li><a href="quiz.html">知识自测</a></li></ul>
  </nav></header>
  <main class="page-shell">
    <header class="hero">
      <p class="tag">Hardware Validation</p>
      <h1>MTT S4000 真机验证</h1>
      <p>课程内容完整度与硬件验证状态分开记录。本页只展示 MUSA SDK ${escapeHtml(manifest.environment.musaSdk)}、驱动 ${escapeHtml(manifest.environment.driver)}、${escapeHtml(manifest.environment.gpu)} 上真实采集的结果。</p>
    </header>
    <section class="knowledge-grid" aria-label="验证状态统计">
      ${statuses.map((status) => `<article class="card"><h2>${status}</h2><p>${counts[status]} 个目标</p></article>`).join("")}
    </section>
    <p class="callout">当前环境只有 ${manifest.environment.gpuCount} 张 GPU，不能据此宣称多卡 MCCL 或 P2P 行为已经验证。故障注入示例也不会在无人值守批处理中自动执行。</p>
${weekTables}
    <nav class="pager" aria-label="页面导航"><a href="index.html">← 返回知识库首页</a><a href="quiz.html">进入完整自测 →</a></nav>
  </main>
  <footer class="footer"><p>MUSA Learning Notes · 内容以官方文档和仓库实测为准</p></footer>
</body>
</html>
`;
}

function writeOrCheck({ outputPath, html, docsRoot, check, outputs }) {
  outputs.push(outputPath);
  const relativeOutput = path.relative(docsRoot, outputPath).split(path.sep).join("/");
  if (check) {
    if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== html) {
      throw new Error(`stale generated page: docs/${relativeOutput}`);
    }
  } else if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== html) {
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, html);
  }
}

function buildPages({ root = defaultRoot, docsRoot = path.join(root, "docs"), check = false } = {}) {
  const outputs = [];
  for (let week = 1; week <= 6; week += 1) {
    for (const [kind, sourceForWeek, label] of KINDS) {
      const sourcePath = sourceForWeek(week);
      const source = fs.readFileSync(path.join(root, sourcePath), "utf8");
      const relativeOutput = `generated/week${week}-${kind}.html`;
      const outputPath = path.join(docsRoot, relativeOutput);
      const html = renderDocument({
        title: `Week ${week} ${label}`,
        week,
        kind: label,
        sourcePath,
        source,
        resolveLink: linkResolver(sourcePath),
      });
      writeOrCheck({ outputPath, html, docsRoot, check, outputs });
    }
  }
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "validation/musa-3.1.0-s4000.json"), "utf8"));
  writeOrCheck({
    outputPath: path.join(docsRoot, "validation.html"),
    html: validationDocument(manifest),
    docsRoot,
    check,
    outputs,
  });
  return outputs;
}

if (require.main === module) {
  try {
    buildPages({ check: process.argv.includes("--check") });
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { KINDS, buildPages, generatedTarget, linkResolver, validationDocument };
