#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const docsRoot = path.join(root, "docs");
const knowledgePages = [
  "docs/index.html",
  "docs/week1.html",
  "docs/week2.html",
  "docs/week3.html",
  "docs/week4.html",
  "docs/week5.html",
  "docs/week6.html",
  "docs/gpu-hierarchy.html",
  "docs/quiz.html",
];

function readText(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function requireFile(relativePath) {
  const fullPath = path.join(root, relativePath);
  if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
    throw new Error(`missing required file: ${relativePath}`);
  }
}

function requireText(relativePath, requiredTexts, raw = false) {
  requireFile(relativePath);
  const text = raw ? readText(relativePath) : staticHtml(readText(relativePath));
  for (const required of requiredTexts) {
    if (!text.includes(required)) {
      throw new Error(`missing required text ${required} in ${relativePath}`);
    }
  }
  return text;
}

function decodeEntities(text) {
  const named = { amp: "&", quot: '"', apos: "'", lt: "<", gt: ">" };
  return text.replace(/&(#x[\da-f]+|#\d+|amp|quot|apos|lt|gt);/gi, (entity, code) => {
    if (code[0] !== "#") return named[code.toLowerCase()];
    const value = code[1].toLowerCase() === "x" ? parseInt(code.slice(2), 16) : Number(code.slice(1));
    return value > 0 && value <= 0x10ffff && !(value >= 0xd800 && value <= 0xdfff)
      ? String.fromCodePoint(value) : "\ufffd";
  });
}

// Tokenize opening tags first, then consume complete attribute values so text
// such as title='example href="..."' cannot introduce a fictitious attribute.
function openingTags(html) {
  const tags = /<([a-z][\w:-]*)\b((?:[^"'<>]|"[^"]*"|'[^']*')*)>/gi;
  return Array.from(html.matchAll(tags), (tag) => {
    const attributes = Object.create(null);
    const pattern = /([^\s"'<>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
    for (const match of tag[2].matchAll(pattern)) {
      const name = match[1].toLowerCase();
      if (!(name in attributes)) attributes[name] = decodeEntities(match[2] ?? match[3] ?? match[4] ?? "");
    }
    return { name: tag[1].toLowerCase(), attributes };
  });
}

function attributeValues(html, name) {
  return openingTags(html).filter((tag) => name in tag.attributes).map((tag) => tag.attributes[name]);
}

function staticHtml(html, preserveScriptTags = false) {
  return html.replace(/<!--[\s\S]*?-->|(<script\b(?:[^"'<>]|"[^"]*"|'[^']*')*>)[\s\S]*?<\/script\s*>|<style\b[^>]*>[\s\S]*?<\/style\s*>/gi,
    (_match, scriptTag) => preserveScriptTags && scriptTag ? scriptTag : "");
}

function requireLink(html, href, relativePath, label, rel) {
  const links = html.match(/<a\b(?:[^"'<>]|"[^"]*"|'[^']*')*>[\s\S]*?<\/a\s*>/gi) || [];
  if (!links.some((link) => {
    const { attributes } = openingTags(link)[0];
    return attributes.href === href
      && (!label || decodeEntities(link.replace(/<[^>]*>/g, "")).trim() === label)
      && (!rel || (attributes.rel || "").split(/\s+/).includes(rel));
  })) {
    throw new Error(`missing required link ${href}${label ? ` (${label})` : ""} in ${relativePath}`);
  }
}

function requirePageKind(html, kind, relativePath) {
  const htmlTag = openingTags(html).find((tag) => tag.name === "html");
  if (htmlTag?.attributes["data-page-kind"] !== kind) {
    throw new Error(`missing data-page-kind="${kind}" on html in ${relativePath}`);
  }
}

function checkKnowledgePages() {
  const homePath = "docs/index.html";
  const home = requireText(homePath, []);
  requirePageKind(home, "home", homePath);
  for (const page of knowledgePages.slice(1)) {
    requireLink(home, path.basename(page), homePath);
  }

  const markers = ["本周要回答的问题", "核心知识", "关键代码", "注意事项", "精选题目", "完整自测"];
  for (let week = 1; week <= 6; week += 1) {
    const page = `docs/week${week}.html`;
    const html = requireText(page, []);
    requirePageKind(html, "week", page);
    const visibleText = html.replace(/<[^>]*>/g, "");
    for (const marker of markers) {
      if (!visibleText.includes(marker)) {
        throw new Error(`missing visible marker ${marker} in ${page}`);
      }
    }
    requireLink(html, "index.html", page);
    requireLink(html, "quiz.html", page);
    if (week > 1) requireLink(html, `week${week - 1}.html`, page, null, "prev");
    if (week < 6) requireLink(html, `week${week + 1}.html`, page, null, "next");
    const relations = attributeValues(html, "rel").flatMap((value) => value.split(/\s+/));
    if ((week === 1 && relations.includes("prev")) || (week === 6 && relations.includes("next"))) {
      throw new Error(`unexpected previous/next link in ${page}`);
    }
    if ((html.match(/<details\b/gi) || []).length < 3) {
      throw new Error(`expected at least 3 details in ${page}`);
    }
    const sourceLinks = attributeValues(html, "href").filter((href) => /^https:\/\/github\.com\/[^/]+\/[^/]+\/blob\//.test(href));
    if (sourceLinks.length < 2) {
      throw new Error(`expected at least 2 GitHub source blob links in ${page}`);
    }
    const repository = "https://github.com/googs1025/musa-learning-notes";
    requireLink(html, `${repository}/tree/main/code/week${week}/`, page, "本周源码目录");
    requireLink(html, `${repository}/blob/main/code/week${week}/exercises.md`, page, "本周练习");
  }

  const topicPath = "docs/gpu-hierarchy.html";
  const topic = requireText(topicPath, ["MPC", "MPX", "一个 kernel 的旅行", "不能逐层翻译"]);
  requirePageKind(topic, "topic", topicPath);
  requireLink(topic, "index.html", topicPath);
  requireLink(topic, "week1.html", topicPath);
  if (topic.includes("MPE")) throw new Error(`unexpected MPE in ${topicPath}`);
  const quiz = requireText("docs/quiz.html", ["musa-learning-quiz-v1"], true);
  requireLink(staticHtml(quiz), "index.html", "docs/quiz.html");
  console.log(`knowledge pages: ${knowledgePages.length}`);
}

function requireLocalTarget(reference, page) {
  const link = reference.trim();
  const hashIndex = link.indexOf("#");
  let fragment;
  let pathname;
  try {
    fragment = hashIndex < 0 ? "" : decodeURIComponent(link.slice(hashIndex + 1));
    pathname = decodeURIComponent(link.split(/[?#]/, 1)[0]);
  } catch {
    throw new Error(`invalid URL encoding in ${reference} in ${page}`);
  }
  let target = pathname
    ? path.resolve(pathname.startsWith("/") ? docsRoot : path.dirname(path.join(root, page)), pathname.replace(/^\//, ""))
    : path.join(root, page);
  const relativeTarget = path.relative(docsRoot, target);
  if (relativeTarget.startsWith("..") || path.isAbsolute(relativeTarget)) {
    throw new Error(`local link outside deployed docs root: ${reference} in ${page}`);
  }
  if (!fs.existsSync(target)) {
    throw new Error(`broken local link ${reference} in ${page}`);
  }
  if (fs.statSync(target).isDirectory()) {
    target = path.join(target, "index.html");
    if (!fs.existsSync(target) || !fs.statSync(target).isFile()) {
      throw new Error(`missing directory index.html for ${reference} in ${page}`);
    }
  }
  if (!fs.statSync(target).isFile()) {
    throw new Error(`invalid local link target ${reference} in ${page}`);
  }
  if (fragment && /\.html?$/i.test(target)) {
    const ids = attributeValues(staticHtml(fs.readFileSync(target, "utf8")), "id");
    if (!ids.includes(fragment)) {
      throw new Error(`broken local fragment ${reference} in ${page}`);
    }
  }
}

function checkLocalLinks() {
  let count = 0;
  for (const page of knowledgePages) {
    const raw = readText(page);
    for (const href of attributeValues(staticHtml(raw), "href")) {
      const link = href.trim();
      if (/^(?:https?:|mailto:|\/\/)/i.test(link) || link === "#") continue;
      requireLocalTarget(href, page);
      count += 1;
    }
    for (const tag of openingTags(staticHtml(raw, true))) {
      if (tag.name !== "script" || !("src" in tag.attributes)) continue;
      const src = tag.attributes.src.trim();
      if (/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(src)) continue;
      requireLocalTarget(src, page);
    }
  }
  console.log(`local links: ${count}`);
}

function checkQuizData() {
  const html = readText("docs/quiz.html");
  const match = html.match(/const QUESTIONS = ([\s\S]*?\n    \];)/);
  if (!match) {
    throw new Error("QUESTIONS not found in docs/quiz.html");
  }

  const questions = Function(`return ${match[1].replace(/;$/, "")}`)();
  if (!Array.isArray(questions)) {
    throw new Error("QUESTIONS is not an array");
  }

  const ids = new Set(questions.map((q) => q.id));
  if (ids.size !== questions.length) {
    throw new Error("duplicate question ids in docs/quiz.html");
  }

  const required = ["id", "deck", "type", "question", "answer", "pitfall", "source"];
  for (const q of questions) {
    for (const key of required) {
      if (!q[key]) {
        throw new Error(`missing ${key} in question ${q.id || "<unknown>"}`);
      }
    }

    const sourcePath = path.join(root, q.source);
    if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isFile()) {
      throw new Error(`missing source ${q.source} in question ${q.id}`);
    }

    if (q.type === "choice") {
      if (!Array.isArray(q.choices) || q.choices.length < 2) {
        throw new Error(`bad choices in question ${q.id}`);
      }
      if (typeof q.answerIndex !== "number" || !q.choices[q.answerIndex]) {
        throw new Error(`bad answerIndex in question ${q.id}`);
      }
    }
  }

  console.log(`quiz questions: ${questions.length}`);
}

function checkLearningMaterials() {
  for (let week = 1; week <= 6; week += 1) {
    requireFile(`code/week${week}/learning-notes.md`);
    requireFile(`code/week${week}/external-cases/README.md`);
  }

  requireFile("docs/cuda-example-map.md");
  requireFile("code/cuda-freshman/README.md");
  console.log("learning materials: ok");
}

checkQuizData();
checkLearningMaterials();
checkKnowledgePages();
checkLocalLinks();
