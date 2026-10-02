#!/usr/bin/env node

const crypto = require("node:crypto");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInline(value, options = {}) {
  const resolveLink = options.resolveLink || ((href) => href);
  let text = escapeHtml(value);
  const code = [];
  text = text.replace(/`([^`]+)`/g, (_match, body) => {
    const marker = `\u0000CODE${code.length}\u0000`;
    code.push(`<code>${body}</code>`);
    return marker;
  });
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)/g, (_match, label, href) => {
    const resolved = escapeHtml(resolveLink(href.replaceAll("&amp;", "&")));
    return `<a href="${resolved}">${label}</a>`;
  });
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  text = text.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  text = text.replace(/\u0000CODE(\d+)\u0000/g, (_match, index) => code[Number(index)]);
  return text;
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function isTableDivider(line) {
  const cells = tableCells(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function renderMarkdown(markdown, options = {}) {
  const lines = String(markdown).replaceAll("\r\n", "\n").split("\n");
  const output = [];
  const inline = (value) => renderInline(value, options);
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(/^\s*```([^\s`]*)\s*$/);
    if (fence) {
      const body = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const language = fence[1] ? ` class="language-${escapeHtml(fence[1])}"` : "";
      output.push(`<pre><code${language}>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      output.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^\s*(?:---+|\*\*\*+)\s*$/.test(line)) {
      output.push("<hr>");
      index += 1;
      continue;
    }

    if (line.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      const headers = tableCells(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      output.push([
        "<div class=\"table-wrap\"><table>",
        `<thead><tr>${headers.map((cell) => `<th>${inline(cell)}</th>`).join("")}</tr></thead>`,
        `<tbody>${rows.map((row) => `<tr>${headers.map((_header, cellIndex) => `<td>${inline(row[cellIndex] || "")}</td>`).join("")}</tr>`).join("")}</tbody>`,
        "</table></div>",
      ].join(""));
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quote = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) {
        quote.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      output.push(`<blockquote><p>${inline(quote.join(" "))}</p></blockquote>`);
      continue;
    }

    const listMatch = line.match(/^\s*(?:([-+*])|(\d+)\.)\s+(.+)$/);
    if (listMatch) {
      const ordered = Boolean(listMatch[2]);
      const tag = ordered ? "ol" : "ul";
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*(?:([-+*])|(\d+)\.)\s+(.+)$/);
        if (!item || Boolean(item[2]) !== ordered) break;
        let body = item[3];
        const task = body.match(/^\[([ xX])\]\s+(.+)$/);
        if (task) {
          const checked = task[1].toLowerCase() === "x" ? " checked" : "";
          body = `<input type="checkbox" disabled${checked}> ${inline(task[2])}`;
        } else {
          body = inline(body);
        }
        items.push(`<li>${body}</li>`);
        index += 1;
      }
      output.push(`<${tag}>${items.join("")}</${tag}>`);
      continue;
    }

    const paragraph = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim()) {
      const next = lines[index];
      if (/^(?:#{1,6})\s+/.test(next)
        || /^\s*```/.test(next)
        || /^\s*>/.test(next)
        || /^\s*(?:[-+*]|\d+\.)\s+/.test(next)
        || (next.includes("|") && index + 1 < lines.length && isTableDivider(lines[index + 1]))) break;
      paragraph.push(next.trim());
      index += 1;
    }
    output.push(`<p>${inline(paragraph.join(" "))}</p>`);
  }

  return output.join("\n");
}

function renderDocument({ title, week, kind, sourcePath, source, resolveLink }) {
  const hash = crypto.createHash("sha256").update(source).digest("hex");
  const sourceUrl = `https://github.com/googs1025/musa-learning-notes/blob/main/${sourcePath}`;
  return `<!doctype html>
<html lang="zh-CN" data-page-kind="generated">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="source-sha256" content="${hash}">
  <title>${escapeHtml(title)} · MUSA GPU 编程知识库</title>
  <link rel="stylesheet" href="../assets/knowledge.css">
  <script src="../assets/knowledge.js" defer></script>
</head>
<body>
  <header class="site-header">
    <nav class="site-nav" aria-label="主导航">
      <a class="brand" href="../index.html">MUSA GPU 编程知识库</a>
      <ul><li><a href="../week${week}.html">Week ${week}</a></li><li><a href="../quiz.html">知识自测</a></li></ul>
    </nav>
  </header>
  <main class="page-shell">
    <article class="article-content generated-content">
      <p class="eyebrow">Week ${week} · ${escapeHtml(kind)}</p>
      <h1>${escapeHtml(title)}</h1>
      <p>本页由仓库中的规范 Markdown 自动生成。学习记录和性能数字只有在真实设备运行后才会写入。</p>
      <p><a class="source-link" href="${sourceUrl}">查看原始 Markdown</a></p>
      ${renderMarkdown(source, { resolveLink })}
    </article>
    <nav class="pager" aria-label="材料导航">
      <a href="../week${week}.html">← 返回 Week ${week}</a>
      <a href="../quiz.html">进入完整自测 →</a>
    </nav>
  </main>
  <footer class="footer"><p>MUSA Learning Notes · 内容以官方文档和仓库实测为准</p></footer>
</body>
</html>
`;
}

module.exports = { escapeHtml, renderInline, renderMarkdown, renderDocument };
