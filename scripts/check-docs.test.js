#!/usr/bin/env node

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const root = path.resolve(__dirname, "..");
const checker = fs.readFileSync(path.join(__dirname, "check-docs.js"), "utf8");
const validation = JSON.parse(fs.readFileSync(path.join(root, "validation/musa-3.1.0-s4000.json"), "utf8"));
const validationStatuses = validation.weeks.flatMap((week) => week.targets).map((target) => target.status);
const validationCounts = Object.fromEntries([...new Set(validationStatuses)].sort().map((status) => [
  status,
  validationStatuses.filter((candidate) => candidate === status).length,
]));
const validationSummary = `validation targets: ${validationStatuses.length} ${JSON.stringify(validationCounts)}`;
const baselineLocalLinks = 270;

// Override reads in memory so regression cases never modify the documentation.
function runChecker(changes = {}) {
  const output = [];
  const fixtureFs = Object.create(fs);
  fixtureFs.readFileSync = (file, ...args) => {
    const text = fs.readFileSync(file, ...args);
    const mutate = changes[path.relative(root, file)];
    return mutate ? mutate(text) : text;
  };
  vm.runInNewContext(checker, {
    require: (name) => name === "fs" ? fixtureFs : require(name),
    __dirname,
    console: { log: (line) => output.push(line) },
  });
  return output;
}

const append = (markup) => (html) => html.replace("</main>", `${markup}</main>`);
const commentLink = (href) => (html) => html.replace(/<a\b[^>]*>[\s\S]*?<\/a>/g,
  (link) => link.includes(`href="${href}"`) ? `<!--${link}-->` : link);

test("baseline validates all pages, evidence and 270 local links", () => {
  assert.deepEqual(runChecker(), [
    "quiz questions: 154",
    "learning materials: ok",
    validationSummary,
    "knowledge pages: 28",
    `local links: ${baselineLocalLinks}`,
  ]);
});

const invalidCases = [
  ["missing local script", "docs/index.html", (html) => html.replace('src="assets/knowledge.js"', 'src="assets/missing.js"'), /assets\/missing\.js.*docs\/index\.html/],
  ["script path escapes deployed root", "docs/index.html", append('<script src="../README.md"></script>'), /outside.*docs.*\.\.\/README\.md.*docs\/index\.html/],
  ["script directory requires index", "docs/index.html", append('<script src="assets/"></script>'), /index\.html.*assets\/.*docs\/index\.html/],
  ["malformed script encoding has context", "docs/index.html", append('<script src="assets/%ZZ.js"></script>'), /encoding.*assets\/%ZZ\.js.*docs\/index\.html/],
  ["missing visible marker", "docs/week1.html", (html) => html.replaceAll("本周要回答的问题", "REMOVED"), /本周要回答的问题/],
  ["commented home link", "docs/index.html", commentLink("week6.html"), /missing required link week6\.html/],
  ["commented topic link", "docs/gpu-hierarchy.html", commentLink("week1.html"), /missing required link week1\.html/],
  ["commented quiz link", "docs/quiz.html", commentLink("index.html"), /missing required link index\.html/],
  ["nested title cannot supply anchor href", "docs/quiz.html", (html) => html.replace('href="index.html">', '><span title=\'nested href="index.html"\'></span>'), /missing required link index\.html/],
  ["nested tag cannot supply anchor href", "docs/quiz.html", (html) => html.replace('href="index.html">', '><span href="index.html"></span>'), /missing required link index\.html/],
  ["commented home page kind", "docs/index.html", (html) => html.replace('data-page-kind="home"', '').replace('</head>', '<!-- data-page-kind="home" --></head>'), /data-page-kind/],
  ["commented week page kind", "docs/week1.html", (html) => html.replace('data-page-kind="week"', '').replace('</head>', '<!-- data-page-kind="week" --></head>'), /data-page-kind/],
  ["commented topic page kind", "docs/gpu-hierarchy.html", (html) => html.replace('data-page-kind="topic"', '').replace('</head>', '<!-- data-page-kind="topic" --></head>'), /data-page-kind/],
  ["page-kind text is not an HTML attribute", "docs/index.html", (html) => append('<pre>data-page-kind="home"</pre>')(html.replace('data-page-kind="home"', '')), /data-page-kind/],
  ["commented topic marker", "docs/gpu-hierarchy.html", (html) => html.replaceAll("不能逐层翻译", "<!--不能逐层翻译-->"), /不能逐层翻译/],
  ["wrong previous pager", "docs/week6.html", (html) => html.replace('href="week5.html" rel="prev"', 'href="week4.html" rel="prev"'), /week5/],
  ["missing local file", "docs/index.html", append('<a href="missing.html">broken</a>'), /missing\.html.*docs\/index\.html/],
  ["missing page fragment", "docs/index.html", append('<a href="week1.html?check=1#missing">broken</a>'), /fragment.*week1\.html\?check=1#missing.*docs\/index\.html/],
  ["missing same-page fragment", "docs/index.html", append('<a href="#missing">broken</a>'), /fragment.*#missing.*docs\/index\.html/],
  ["text id cannot satisfy fragment", "docs/index.html", append('<pre> id="missing"</pre><a href="#missing">broken</a>'), /fragment.*#missing/],
  ["parent path escapes deployed root", "docs/index.html", append('<a href="../README.md">outside</a>'), /outside.*docs.*\.\.\/README\.md.*docs\/index\.html/],
  ["encoded parent path escapes deployed root", "docs/index.html", append('<a href="%2e%2e/README.md">outside</a>'), /outside.*docs.*%2e%2e\/README\.md/],
  ["directory without index is not a page", "docs/index.html", append('<a href="assets/">directory</a>'), /index\.html.*assets\/.*docs\/index\.html/],
  ["directory fragments target index", "docs/index.html", append('<a href="./#missing">broken</a>'), /fragment.*\.\/#missing/],
  ["malformed path percent encoding has context", "docs/index.html", append('<a href="%ZZ.html">broken</a>'), /encoding.*%ZZ\.html.*docs\/index\.html/],
  ["malformed fragment percent encoding has context", "docs/index.html", append('<a href="week1.html#%ZZ">broken</a>'), /encoding.*week1\.html#%ZZ.*docs\/index\.html/],
];
for (const [name, page, mutate, expected] of invalidCases) {
  test(name, () => assert.throws(() => runChecker({ [page]: mutate }), expected));
}

const validCases = [
  ["src text inside pre is ignored", '<pre> src="assets/missing.js"</pre>', 0],
  ["commented script is ignored", '<!-- <script src="assets/missing.js"></script> -->', 0],
  ["script body is not parsed for src", '<script>const sample = \'<script src="assets/missing.js">\';</script>', 0],
  ["external and protocol script sources are ignored", '<script src="https://example.com/app.js"></script><script src="//example.com/app.js"></script><script src="data:text/javascript,void(0)"></script>', 0],
  ["encoded local script path is valid", '<script src="assets/knowledge%2Ejs?q=1&amp;v=2"></script>', 0],
  ["href and rel text inside pre are ignored", '<pre> href="missing.html" rel="prev"</pre>', 0],
  ["href text in a title is ignored", '<span title=\'nested href="missing.html"\'>text</span>', 0],
  ["encoded path query and fragment resolve", '<a href="week%31.html?q=a&amp;b=2#execution%2Dmodel">encoded</a>', 1],
  ["directory index fragment resolves", '<a href="./?q=1#weeks">directory</a>', 1],
  ["named and numeric entities in href and id resolve", '<div id="a&amp;b&quot;c&apos;d&lt;e&gt;f&#49;&#x32;"></div><a href="#a&#38;b%22c%27d%3Ce%3Ef12">entities</a>', 1],
  ["external links and pure hash are ignored", '<a href="https://example.com/">web</a><a href="mailto:a@example.com">email</a><a href="#">top</a>', 0],
];
for (const [name, markup, addedLinks] of validCases) {
  test(name, () => assert.equal(
    runChecker({ "docs/index.html": append(markup) }).at(-1),
    `local links: ${baselineLocalLinks + addedLinks}`,
  ));
}

test("rejects an unknown validation status", () => {
  const mutate = (text) => {
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
  const mutate = (text) => {
    const data = JSON.parse(text);
    data.weeks[0].targets[0].evidence = "validation/raw/2026-10-02-s4000/missing.log";
    return JSON.stringify(data);
  };
  assert.throws(
    () => runChecker({ "validation/musa-3.1.0-s4000.json": mutate }),
    /missing validation evidence/,
  );
});

test("homepage links the hardware validation overview", () => {
  const home = fs.readFileSync(path.join(root, "docs/index.html"), "utf8");
  assert.match(home, /href="validation\.html"/);
});

test("weekly pages expose completion criteria and all local materials", () => {
  for (let week = 1; week <= 6; week += 1) {
    const html = fs.readFileSync(path.join(root, `docs/week${week}.html`), "utf8");
    assert.match(html, /本周通关标准/, `Week ${week} completion heading`);
    assert.match(html, new RegExp(`href="generated/week${week}-learning\\.html"`));
    assert.match(html, new RegExp(`href="generated/week${week}-exercises\\.html"`));
    assert.match(html, new RegExp(`href="generated/week${week}-records\\.html"`));
    assert.equal((html.match(/class="completion-item"/g) || []).length, 5);
  }
});

test("validation page renders known states and evidence links", () => {
  const validationPath = path.join(root, "docs/validation.html");
  assert.ok(fs.existsSync(validationPath), "validation page must exist");
  const html = fs.readFileSync(validationPath, "utf8");
  assert.match(html, /data-page-kind="validation"/);
  assert.match(html, /PASS/);
  assert.match(html, /ENV_LIMITED/);
  assert.match(html, /validation\/raw\/2026-10-02-s4000/);
});
