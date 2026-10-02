#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { renderDocument } = require("./page-renderer");

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
      outputs.push(outputPath);
      if (check) {
        if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== html) {
          throw new Error(`stale generated page: docs/${relativeOutput}`);
        }
      } else if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== html) {
        fs.mkdirSync(path.dirname(outputPath), { recursive: true });
        fs.writeFileSync(outputPath, html);
      }
    }
  }
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

module.exports = { KINDS, buildPages, generatedTarget, linkResolver };
