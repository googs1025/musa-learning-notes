#!/usr/bin/env node

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");

const root = path.resolve(__dirname, "..");

test("renders repository markdown without allowing raw HTML", () => {
  assert.ok(fs.existsSync(path.join(__dirname, "page-renderer.js")), "page renderer must exist");
  const { renderMarkdown } = require("./page-renderer");
  const html = renderMarkdown([
    "# Title",
    "",
    "<script>alert(1)</script>",
    "",
    "- [ ] run `kernel`",
    "",
    "| A | B |",
    "|---|---|",
    "| x | y |",
    "",
    "```cpp",
    "if (x < y) return;",
    "```",
  ].join("\n"));

  assert.match(html, /<h1>Title<\/h1>/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<script>alert/);
  assert.match(html, /type="checkbox" disabled/);
  assert.match(html, /<code>kernel<\/code>/);
  assert.match(html, /<table>/);
  assert.match(html, /<pre><code class="language-cpp">if \(x &lt; y\) return;/);
});

test("generates eighteen deterministic weekly pages, validation and detects stale output", () => {
  assert.ok(fs.existsSync(path.join(__dirname, "build-pages.js")), "page generator must exist");
  const { buildPages } = require("./build-pages");
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "musa-pages-"));
  const docsRoot = path.join(tempRoot, "docs");
  fs.mkdirSync(docsRoot, { recursive: true });

  const written = buildPages({ root, docsRoot });
  assert.equal(written.length, 19);
  assert.equal(written.filter((file) => file.includes(`${path.sep}generated${path.sep}`)).length, 18);
  assert.ok(fs.statSync(path.join(docsRoot, "validation.html")).isFile());
  const validationHtml = fs.readFileSync(path.join(docsRoot, "validation.html"), "utf8");
  assert.doesNotMatch(validationHtml, /[ \t]+$/m, "validation page must not contain trailing whitespace");

  for (let week = 1; week <= 6; week += 1) {
    for (const kind of ["learning", "exercises", "records"]) {
      const output = path.join(docsRoot, "generated", `week${week}-${kind}.html`);
      assert.ok(fs.statSync(output).isFile());
      const html = fs.readFileSync(output, "utf8");
      assert.match(html, /data-page-kind="generated"/);
      assert.equal((html.match(/<h1>/g) || []).length, 1, `${output} must have one h1`);
      assert.match(html, new RegExp(`href="\.\./week${week}\\.html"`));
      assert.match(html, /href="\.\.\/quiz\.html"/);
      const source = kind === "learning"
        ? path.join(root, `code/week${week}/learning-notes.md`)
        : kind === "exercises"
          ? path.join(root, `code/week${week}/exercises.md`)
          : path.join(root, `notes/week${week}.md`);
      const hash = crypto.createHash("sha256").update(fs.readFileSync(source)).digest("hex");
      assert.match(html, new RegExp(`<meta name="source-sha256" content="${hash}">`));
    }
  }

  assert.doesNotThrow(() => buildPages({ root, docsRoot, check: true }));
  fs.appendFileSync(path.join(docsRoot, "generated/week1-learning.html"), "\nstale");
  assert.throws(
    () => buildPages({ root, docsRoot, check: true }),
    /stale generated page: docs\/generated\/week1-learning\.html/,
  );
});
