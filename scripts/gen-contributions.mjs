/**
 * 从 Git 提交记录生成打卡（贡献热力图）数据
 * 输出: src/.vuepress/components/contributions.json
 * 用法: npm run docs:contrib
 */
import { execSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outFile = path.join(
  root,
  "src/.vuepress/components/contributions.json",
);

// 使用作者日期（author date），保留真实的创作时间
const output = execSync('git log --date=short --format="%ad"', {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 16 * 1024 * 1024,
});

const counts = {};
for (const line of output.split("\n")) {
  const date = line.trim();
  if (!date) continue;
  counts[date] = (counts[date] ?? 0) + 1;
}

const total = Object.values(counts).reduce((a, b) => a + b, 0);
const dates = Object.keys(counts).sort();
const data = {
  generatedAt: new Date().toISOString().slice(0, 10),
  firstCommitDate: dates[0] ?? null,
  lastCommitDate: dates[dates.length - 1] ?? null,
  totalCommits: total,
  counts,
};

writeFileSync(outFile, JSON.stringify(data, null, 2));
console.log(
  `✓ 生成 ${path.relative(root, outFile)}：共 ${total} 次提交，` +
    `${dates.length} 个活跃日期` +
    (dates[0] ? `（${dates[0]} ~ ${dates[dates.length - 1]}）` : ""),
);
