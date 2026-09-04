#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");

function readText(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function requireFile(relativePath) {
  const fullPath = path.join(root, relativePath);
  if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
    throw new Error(`missing required file: ${relativePath}`);
  }
}

function checkQuizData() {
  const html = readText("docs/index.html");
  const match = html.match(/const QUESTIONS = ([\s\S]*?\n    \];)/);
  if (!match) {
    throw new Error("QUESTIONS not found in docs/index.html");
  }

  const questions = Function(`return ${match[1].replace(/;$/, "")}`)();
  if (!Array.isArray(questions)) {
    throw new Error("QUESTIONS is not an array");
  }

  const ids = new Set(questions.map((q) => q.id));
  if (ids.size !== questions.length) {
    throw new Error("duplicate question ids in docs/index.html");
  }

  const required = ["id", "deck", "type", "question", "answer", "pitfall", "source"];
  for (const q of questions) {
    for (const key of required) {
      if (!q[key]) {
        throw new Error(`missing ${key} in question ${q.id || "<unknown>"}`);
      }
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
