#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
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

function requireText(relativePath, requiredTexts) {
  requireFile(relativePath);
  const text = readText(relativePath);
  for (const required of requiredTexts) {
    if (!text.includes(required)) {
      throw new Error(`missing required text ${required} in ${relativePath}`);
    }
  }
  return text;
}

function attributeValues(html, name) {
  const pattern = new RegExp(`\\s${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'=<>\x60]+))`, "gi");
  return Array.from(html.matchAll(pattern), (match) => match[1] ?? match[2] ?? match[3]);
}

function staticHtml(html) {
  return html.replace(/<!--[\s\S]*?-->|<script\b[^>]*>[\s\S]*?<\/script\s*>|<style\b[^>]*>[\s\S]*?<\/style\s*>/gi, "");
}

function requireLink(html, href, relativePath, label, rel) {
  const links = html.match(/<a\b[^>]*>[\s\S]*?<\/a\s*>/gi) || [];
  if (!links.some((link) => attributeValues(link, "href").includes(href)
    && (!label || link.replace(/<[^>]*>/g, "").trim() === label)
    && (!rel || attributeValues(link, "rel").some((value) => value.split(/\s+/).includes(rel))))) {
    throw new Error(`missing required link ${href}${label ? ` (${label})` : ""} in ${relativePath}`);
  }
}

function checkKnowledgePages() {
  const homePath = "docs/index.html";
  const home = requireText(homePath, ['data-page-kind="home"']);
  for (const page of knowledgePages.slice(1)) {
    requireLink(home, path.basename(page), homePath);
  }

  const markers = ["本周要回答的问题", "核心知识", "关键代码", "注意事项", "精选题目", "完整自测"];
  for (let week = 1; week <= 6; week += 1) {
    const page = `docs/week${week}.html`;
    const html = staticHtml(requireText(page, ['data-page-kind="week"']));
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
  const topic = requireText(topicPath, ['data-page-kind="topic"', "MPC", "MPX", "一个 kernel 的旅行", "不能逐层翻译"]);
  requireLink(topic, "index.html", topicPath);
  requireLink(topic, "week1.html", topicPath);
  if (topic.includes("MPE")) throw new Error(`unexpected MPE in ${topicPath}`);
  const quiz = requireText("docs/quiz.html", ["musa-learning-quiz-v1"]);
  requireLink(quiz, "index.html", "docs/quiz.html");
  console.log(`knowledge pages: ${knowledgePages.length}`);
}

function checkLocalLinks() {
  let count = 0;
  for (const page of knowledgePages) {
    const html = staticHtml(readText(page));
    for (const href of attributeValues(html, "href")) {
      const link = href.trim().replace(/&amp;/gi, "&");
      if (/^(?:https?:|mailto:|\/\/)/i.test(link) || link === "#") continue;
      const hashIndex = link.indexOf("#");
      const fragment = hashIndex < 0 ? "" : decodeURIComponent(link.slice(hashIndex + 1));
      const pathname = decodeURIComponent(link.split(/[?#]/, 1)[0]);
      const target = pathname
        ? path.resolve(pathname.startsWith("/") ? path.join(root, "docs") : path.dirname(path.join(root, page)), pathname.replace(/^\//, ""))
        : path.join(root, page);
      if (!fs.existsSync(target)) {
        throw new Error(`broken local link ${href} in ${page}`);
      }
      const stat = fs.statSync(target);
      if (!stat.isFile() && !stat.isDirectory()) {
        throw new Error(`invalid local link target ${href} in ${page}`);
      }
      if (fragment && stat.isFile() && /\.html?$/i.test(target)) {
        const ids = attributeValues(staticHtml(fs.readFileSync(target, "utf8")), "id");
        if (!ids.includes(fragment)) {
          throw new Error(`broken local fragment ${href} in ${page}`);
        }
      }
      count += 1;
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
