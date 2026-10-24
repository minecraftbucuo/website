<template>
  <div class="maze-game">
    <div class="maze-toolbar">
      <div class="maze-seg" role="group" aria-label="迷宫难度">
        <button
          v-for="(size, name) in DIFFS"
          :key="name"
          :class="{ active: difficulty === name }"
          @click="setDifficulty(name)"
        >
          {{ name }}
        </button>
      </div>
      <button class="maze-btn" @click="newMaze">🔀 换个迷宫</button>
      <button class="maze-btn" @click="retryMaze">↺ 重走</button>
      <button class="maze-btn" @click="startDemo">🛰 BFS 演示</button>
    </div>

    <div class="maze-hud">
      <span>👣 步数 <b>{{ steps }}</b></span>
      <span>⏱ {{ timeText }}</span>
      <span>🎯 最短 <b>{{ shortest }}</b> 步</span>
      <span v-if="bestText">🏆 最佳 {{ bestText }}</span>
    </div>

    <div ref="wrapRef" class="maze-wrap" tabindex="0" aria-label="迷宫游戏区域，点击后可用方向键操作">
      <canvas
        ref="canvasRef"
        @pointerdown="onPointerDown"
        @pointerup="onPointerUp"
      />

      <div v-if="phase === 'demo' && !demoDone" class="maze-chip">🛰 BFS 扩散中…</div>

      <div v-if="phase === 'idle'" class="maze-overlay" @click="startPlay">
        <div class="maze-card">
          <div class="maze-card-title">🏁 走到右下角的旗子</div>
          <p class="maze-card-text">
            方向键 / WASD 移动，手机直接滑动<br />
            用尽量少的步数 —— BFS 会来揭榜
          </p>
          <button class="maze-btn primary">开始探索</button>
          <p class="maze-card-sub">先点击一下游戏区域，键盘才归你管</p>
        </div>
      </div>

      <div v-else-if="phase === 'demo' && demoDone" class="maze-overlay">
        <div class="maze-card">
          <div class="maze-card-title">🛰 BFS 的答卷</div>
          <p class="maze-card-text">
            一共扩散了 <b>{{ demoVisited }}</b> 个格子，<br />
            从起点到终点最短需要 <b>{{ shortest }}</b> 步
          </p>
          <div class="maze-card-actions">
            <button class="maze-btn primary" @click="startPlay">我自己试试</button>
            <button class="maze-btn" @click="newMaze">换个迷宫</button>
          </div>
        </div>
      </div>

      <div v-else-if="phase === 'won' && winCard" class="maze-overlay">
        <div class="maze-card">
          <div class="maze-rating" :data-r="rating">{{ rating }}</div>
          <div class="maze-card-title">{{ ratingTitle }}</div>
          <p class="maze-card-text">
            你走了 <b>{{ steps }}</b> 步 · BFS 最短 <b>{{ shortest }}</b> 步 · 用时 {{ timeText }}
          </p>
          <p v-if="newRecord" class="maze-record">🎉 新纪录！</p>
          <div class="maze-card-actions">
            <button class="maze-btn primary" @click="retryMaze">再走一遍</button>
            <button class="maze-btn" @click="newMaze">换个迷宫</button>
          </div>
        </div>
      </div>
    </div>

    <div class="maze-dpad">
      <span></span>
      <button aria-label="上" @click="dpad(0)">▲</button>
      <span></span>
      <button aria-label="左" @click="dpad(3)">◀</button>
      <button aria-label="下" @click="dpad(2)">▼</button>
      <button aria-label="右" @click="dpad(1)">▶</button>
    </div>

    <p class="maze-tip">回头路也计步 —— 想拿 S 评价，就不能走一步冤枉路。</p>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

type DiffName = "初级" | "中级" | "高级";
const DIFFS: Record<DiffName, number> = { 初级: 10, 中级: 16, 高级: 24 };

// 方向：0 北 1 东 2 南 3 西
const DX = [0, 1, 0, -1];
const DY = [-1, 0, 1, 0];
const WALLBIT = [1, 2, 4, 8];
const OPP = [2, 3, 0, 1];

const BEST_KEY = "maze-bfs-best-v1";
type Best = { steps: number; ms: number };

const difficulty = ref<DiffName>("中级");
const phase = ref<"idle" | "playing" | "demo" | "won">("idle");
const steps = ref(0);
const shortest = ref(0);
const demoVisited = ref(0);
const demoDone = ref(false);
const winCard = ref(false);
const newRecord = ref(false);
const timeText = ref("0:00");

const canvasRef = ref<HTMLCanvasElement | null>(null);
const wrapRef = ref<HTMLDivElement | null>(null);

let ctx: CanvasRenderingContext2D | null = null;
let n = DIFFS["中级"];
let walls = new Uint8Array(0);
let dist = new Int32Array(0);
let bfsPath: number[] = [];
let goalIdx = 0;
let maxDist = 1;

// 玩家
let fx = 0, fy = 0; // 插值浮点坐标（格子单位）
let pcx = 0, pcy = 0; // 逻辑格子
let tx = 0, ty = 0; // 补间目标
let tweening = false;
let pendingDir = -1;
let facing = 1;
let bump = 0, bumpDir = 0;
let trail: number[] = [];

// 计时 / 动画时间线
let phaseStart = 0;
let startTime = 0;
let finalMs = 0;
let movedOnce = false;

let canvasW = 0;
let cell = 0;
let raf = 0;
let logicTimer = 0;
let lastLogic = 0;
let resizeObserver: ResizeObserver | null = null;
let themeObserver: MutationObserver | null = null;

const mqlReduce =
  typeof window !== "undefined" && window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : null;
const reducedMotion = () => !!mqlReduce?.matches;

// ---------- 纪录 ----------
const bests = reactive<Partial<Record<DiffName, Best>>>({});
try {
  Object.assign(bests, JSON.parse(localStorage.getItem(BEST_KEY) || "{}"));
} catch {
  /* 隐私模式下静默失败 */
}
const bestText = computed(() => {
  const b = bests[difficulty.value];
  return b ? `${b.steps} 步` : "";
});

// ---------- 评分 ----------
const rating = computed<"S" | "A" | "B" | "C">(() => {
  if (!shortest.value) return "S";
  const r = steps.value / shortest.value;
  return r <= 1 ? "S" : r <= 1.15 ? "A" : r <= 1.5 ? "B" : "C";
});
const ratingTitle = computed(
  () =>
    ({
      S: "完美最短路！你就是人形 BFS",
      A: "几乎最优，路感极佳",
      B: "顺利到达，略有绕路",
      C: "到达了，虽然绕了点远路 😄",
    })[rating.value],
);

// ---------- 主题色 ----------
const theme = { wall: "#666", accent: "#3eaf7c" };
function refreshTheme() {
  const s = getComputedStyle(document.documentElement);
  theme.wall = s.getPropertyValue("--vp-c-text").trim() || "#555";
  theme.accent = s.getPropertyValue("--vp-c-accent").trim() || "#3eaf7c";
}

// ---------- 迷宫生成（随机化 DFS / 递归回溯） ----------
function genMaze() {
  n = DIFFS[difficulty.value];
  const total = n * n;
  walls = new Uint8Array(total).fill(15);
  const visited = new Uint8Array(total);
  const stack: number[] = [0];
  visited[0] = 1;
  const order = [0, 1, 2, 3];
  while (stack.length) {
    const cur = stack[stack.length - 1];
    const cx = cur % n;
    const cy = (cur / n) | 0;
    for (let i = 3; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      [order[i], order[j]] = [order[j], order[i]];
    }
    let advanced = false;
    for (const d of order) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
      const ni = ny * n + nx;
      if (visited[ni]) continue;
      walls[cur] &= ~WALLBIT[d];
      walls[ni] &= ~WALLBIT[OPP[d]];
      visited[ni] = 1;
      stack.push(ni);
      advanced = true;
      break;
    }
    if (!advanced) stack.pop();
  }
  goalIdx = total - 1;
  runBfs();
}

// ---------- BFS 最短路 ----------
function runBfs() {
  const total = n * n;
  dist = new Int32Array(total).fill(-1);
  const parent = new Int32Array(total).fill(-1);
  const q = new Int32Array(total);
  let head = 0;
  let tail = 0;
  dist[0] = 0;
  q[tail++] = 0;
  maxDist = 0;
  while (head < tail) {
    const cur = q[head++];
    const cx = cur % n;
    const cy = (cur / n) | 0;
    for (let d = 0; d < 4; d++) {
      if (walls[cur] & WALLBIT[d]) continue;
      const ni = (cy + DY[d]) * n + cx + DX[d];
      if (dist[ni] !== -1) continue;
      dist[ni] = dist[cur] + 1;
      if (dist[ni] > maxDist) maxDist = dist[ni];
      parent[ni] = cur;
      q[tail++] = ni;
    }
  }
  shortest.value = dist[goalIdx];
  bfsPath = [];
  let c = goalIdx;
  while (c !== -1) {
    bfsPath.push(c);
    c = parent[c];
  }
  bfsPath.reverse();
}

// ---------- 回合控制 ----------
function resetRun(autoStart: boolean) {
  pcx = 0;
  pcy = 0;
  fx = 0;
  fy = 0;
  tx = 0;
  ty = 0;
  tweening = false;
  pendingDir = -1;
  facing = 1;
  trail = [0];
  steps.value = 0;
  movedOnce = false;
  startTime = 0;
  finalMs = 0;
  demoDone.value = false;
  winCard.value = false;
  newRecord.value = false;
  phase.value = autoStart ? "playing" : "idle";
  timeText.value = "0:00";
}

function setDifficulty(name: string) {
  difficulty.value = name as DiffName;
  genMaze();
  resetRun(false);
  resizeCanvas();
  focusWrap();
}

function newMaze() {
  genMaze();
  resetRun(false);
  resizeCanvas();
  focusWrap();
}

function retryMaze() {
  resetRun(true);
  focusWrap();
}

function startPlay() {
  if (phase.value === "won" || phase.value === "demo") resetRun(true);
  else phase.value = "playing";
  focusWrap();
}

function startDemo() {
  resetRun(false);
  phase.value = "demo";
  phaseStart = performance.now();
  let cnt = 0;
  for (let i = 0; i < dist.length; i++) if (dist[i] >= 0) cnt++;
  demoVisited.value = cnt;
  focusWrap();
}

function finishWin() {
  finalMs = movedOnce ? performance.now() - startTime : 0;
  phase.value = "won";
  winCard.value = false;
  phaseStart = performance.now();
  pendingDir = -1;
  const b = bests[difficulty.value];
  const better =
    !b || steps.value < b.steps || (steps.value === b.steps && finalMs < b.ms);
  newRecord.value = better;
  if (better) {
    bests[difficulty.value] = { steps: steps.value, ms: Math.round(finalMs) };
    try {
      localStorage.setItem(BEST_KEY, JSON.stringify(bests));
    } catch {
      /* 隐私模式下静默失败 */
    }
  }
}

// ---------- 移动 ----------
function tryMove(d: number) {
  if (phase.value === "idle") phase.value = "playing";
  if (phase.value !== "playing") return;
  facing = d;
  if (tweening) {
    pendingDir = d;
    return;
  }
  const cur = pcy * n + pcx;
  if (walls[cur] & WALLBIT[d]) {
    bump = 1;
    bumpDir = d;
    return;
  }
  pcx += DX[d];
  pcy += DY[d];
  tx = pcx;
  ty = pcy;
  tweening = true;
  steps.value++;
  trail.push(pcy * n + pcx);
  if (!movedOnce) {
    movedOnce = true;
    startTime = performance.now();
  }
}

function stepTween(dt: number) {
  if (!tweening) return;
  const speed = reducedMotion() ? 1 : 0.016; // 格子 / 毫秒
  const dx = tx - fx;
  const dy = ty - fy;
  const len = Math.hypot(dx, dy);
  const step = speed * dt;
  if (len <= step || len === 0) {
    fx = tx;
    fy = ty;
    tweening = false;
  } else {
    fx += (dx / len) * step;
    fy += (dy / len) * step;
    return;
  }
  if (pcy * n + pcx === goalIdx) {
    finishWin();
    return;
  }
  if (pendingDir >= 0) {
    const d = pendingDir;
    pendingDir = -1;
    tryMove(d);
  }
}

// ---------- 输入 ----------
const KEYMAP: Record<string, number> = {
  ArrowUp: 0,
  KeyW: 0,
  ArrowRight: 1,
  KeyD: 1,
  ArrowDown: 2,
  KeyS: 2,
  ArrowLeft: 3,
  KeyA: 3,
};

function onKey(e: KeyboardEvent) {
  if (document.activeElement !== wrapRef.value) return;
  const d = KEYMAP[e.code];
  if (d === undefined) return;
  e.preventDefault();
  tryMove(d);
}

let downX = 0;
let downY = 0;
function onPointerDown(e: PointerEvent) {
  focusWrap();
  downX = e.clientX;
  downY = e.clientY;
}
function onPointerUp(e: PointerEvent) {
  const dx = e.clientX - downX;
  const dy = e.clientY - downY;
  if (Math.hypot(dx, dy) < 24) return; // 视为点击而非滑动
  const d =
    Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : dy > 0 ? 2 : 0;
  tryMove(d);
}
function dpad(d: number) {
  focusWrap();
  tryMove(d);
}

function focusWrap() {
  wrapRef.value?.focus({ preventScroll: true });
}

// ---------- 画布 ----------
function resizeCanvas() {
  const canvas = canvasRef.value;
  const wrap = wrapRef.value;
  if (!canvas || !wrap) return;
  const w = Math.max(240, Math.min(560, wrap.clientWidth - 2));
  const dpr = window.devicePixelRatio || 1;
  canvasW = w;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${w}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(w * dpr);
  ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cell = (w - 2) / n;
  }
}

function rr(x: number, y: number, w: number, h: number, r: number) {
  if (!ctx) return;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function circle(x: number, y: number, r: number) {
  if (!ctx) return;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

function fmt(ms: number) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function drawGoldPath(reveal: number) {
  if (!ctx || bfsPath.length < 2) return;
  const ox = 1;
  const oy = 1;
  const total = bfsPath.length;
  const seg = reveal * (total - 1);
  ctx.strokeStyle = "#f2a93b";
  ctx.lineWidth = Math.max(3, cell * 0.22);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(242, 169, 59, 0.55)";
  ctx.shadowBlur = cell * 0.35;
  ctx.beginPath();
  const full = Math.min(Math.floor(seg), total - 1);
  for (let k = 0; k <= full; k++) {
    const idx = bfsPath[k];
    const px = ox + ((idx % n) + 0.5) * cell;
    const py = oy + (((idx / n) | 0) + 0.5) * cell;
    if (k === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  if (full < total - 1) {
    const a = bfsPath[full];
    const b = bfsPath[full + 1];
    const frac = seg - full;
    ctx.lineTo(
      ox + ((a % n) + 0.5 + ((b % n) - (a % n)) * frac) * cell,
      oy + (((a / n) | 0) + 0.5 + (((b / n) | 0) - ((a / n) | 0)) * frac) * cell,
    );
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

// 逻辑时钟：补间推进、时间线状态切换、计时文本。
// 与渲染解耦——rAF 被浏览器节流（后台标签 / 省电模式）时逻辑仍要前进。
function tickLogic(now: number) {
  const dt = Math.min(50, lastLogic ? now - lastLogic : 16);
  lastLogic = now;
  stepTween(dt);
  if (bump > 0) bump = Math.max(0, bump - dt / 140);

  if (phase.value === "demo") {
    const t = now - phaseStart;
    const revealDur = reducedMotion()
      ? 0
      : Math.min(2600, Math.max(900, maxDist * 70));
    const pt = (t - revealDur - 250) / 900;
    if (pt >= 1 && !demoDone.value) demoDone.value = true;
  } else if (phase.value === "won") {
    const t = now - phaseStart;
    if (t > 1500 && !winCard.value) winCard.value = true;
  }

  if (phase.value === "playing" && movedOnce)
    timeText.value = fmt(performance.now() - startTime);
  else if (phase.value === "won") timeText.value = fmt(finalMs);
}

function draw(now: number) {
  if (!ctx) return;

  let revealLevel = -1;
  let pathReveal = 0;
  if (phase.value === "demo") {
    const t = now - phaseStart;
    const revealDur = reducedMotion()
      ? 0
      : Math.min(2600, Math.max(900, maxDist * 70));
    revealLevel = revealDur
      ? maxDist * Math.min(1, t / revealDur)
      : maxDist + 1;
    pathReveal = Math.max(0, Math.min(1, (t - revealDur - 250) / 900));
  } else if (phase.value === "won") {
    const t = now - phaseStart;
    pathReveal = Math.max(0, Math.min(1, (t - 450) / 900));
  }

  const ox = 1;
  const oy = 1;
  ctx.clearRect(0, 0, canvasW, canvasW);

  // BFS 扩散层
  if (revealLevel >= 0) {
    for (let i = 0; i < n * n; i++) {
      const d = dist[i];
      if (d < 0 || d > revealLevel) continue;
      const t = d / maxDist;
      ctx.fillStyle = `hsla(${195 + 105 * t}, 78%, 56%, ${(0.1 + 0.3 * t).toFixed(3)})`;
      ctx.fillRect(ox + (i % n) * cell, oy + ((i / n) | 0) * cell, cell, cell);
    }
  }

  // 玩家足迹
  if (trail.length > 1) {
    ctx.strokeStyle = theme.accent;
    ctx.globalAlpha = 0.38;
    ctx.lineWidth = Math.max(3, cell * 0.3);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let k = 0; k < trail.length; k++) {
      const px = ox + ((trail[k] % n) + 0.5) * cell;
      const py = oy + (((trail[k] / n) | 0) + 0.5) * cell;
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // 金色最短路（演示 / 通关对照）
  if (pathReveal > 0) drawGoldPath(pathReveal);

  // 墙
  ctx.strokeStyle = theme.wall;
  ctx.globalAlpha = 0.75;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const w = walls[y * n + x];
      const px = ox + x * cell;
      const py = oy + y * cell;
      if (w & 1) {
        ctx.moveTo(px, py);
        ctx.lineTo(px + cell, py);
      }
      if (w & 8) {
        ctx.moveTo(px, py);
        ctx.lineTo(px, py + cell);
      }
    }
  }
  ctx.moveTo(ox + n * cell, oy);
  ctx.lineTo(ox + n * cell, oy + n * cell);
  ctx.moveTo(ox, oy + n * cell);
  ctx.lineTo(ox + n * cell, oy + n * cell);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // 起点
  ctx.strokeStyle = theme.wall;
  ctx.globalAlpha = 0.3;
  ctx.beginPath();
  ctx.arc(ox + 0.5 * cell, oy + 0.5 * cell, cell * 0.2, 0, Math.PI * 2);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // 终点旗
  const gx = ox + ((goalIdx % n) + 0.5) * cell;
  const gy = oy + (((goalIdx / n) | 0) + 0.5) * cell;
  ctx.globalAlpha = 0.22 + 0.14 * Math.sin(now / 300);
  ctx.fillStyle = "#f2a93b";
  circle(gx, gy, cell * 0.42);
  ctx.globalAlpha = 1;
  ctx.font = `${Math.floor(cell * 0.52)}px "Segoe UI Emoji", "Noto Color Emoji", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("🏁", gx, gy + cell * 0.04);

  // 玩家角色
  let px = ox + (fx + 0.5) * cell;
  let py = oy + (fy + 0.5) * cell;
  if (bump > 0) {
    const k = Math.sin(bump * Math.PI) * cell * 0.15;
    px += DX[bumpDir] * k;
    py += DY[bumpDir] * k;
  }
  const s =
    cell * 0.62 * (tweening ? 1 + 0.07 * Math.sin(now / 40) : 1);
  ctx.fillStyle = theme.accent;
  rr(px - s / 2, py - s / 2, s, s, s * 0.32);
  ctx.fill();
  ctx.strokeStyle = "rgba(0, 0, 0, 0.25)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  const ex = DX[facing];
  const ey = DY[facing];
  const e1x = px + ex * s * 0.12 - ey * s * 0.14;
  const e1y = py + ey * s * 0.12 + ex * s * 0.14;
  const e2x = px + ex * s * 0.12 + ey * s * 0.14;
  const e2y = py + ey * s * 0.12 - ex * s * 0.14;
  ctx.fillStyle = "#fff";
  circle(e1x, e1y, s * 0.11);
  circle(e2x, e2y, s * 0.11);
  ctx.fillStyle = "#333";
  circle(e1x + ex * s * 0.045, e1y + ey * s * 0.045, s * 0.055);
  circle(e2x + ex * s * 0.045, e2y + ey * s * 0.045, s * 0.055);
}

function loop(now: number) {
  draw(now);
  raf = requestAnimationFrame(loop);
}

// ---------- 生命周期 ----------
onMounted(() => {
  refreshTheme();
  genMaze();
  resetRun(false);
  resizeCanvas();
  raf = requestAnimationFrame(loop);
  logicTimer = window.setInterval(
    () => tickLogic(performance.now()),
    16,
  );
  resizeObserver = new ResizeObserver(() => resizeCanvas());
  if (wrapRef.value) resizeObserver.observe(wrapRef.value);
  themeObserver = new MutationObserver(refreshTheme);
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class", "data-theme"],
  });
  window.addEventListener("keydown", onKey);
});

onBeforeUnmount(() => {
  cancelAnimationFrame(raf);
  clearInterval(logicTimer);
  window.removeEventListener("keydown", onKey);
  resizeObserver?.disconnect();
  themeObserver?.disconnect();
});
</script>

<style scoped>
.maze-game {
  max-width: 580px;
  margin: 1rem auto;
}

.maze-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.maze-seg {
  display: inline-flex;
  padding: 2px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 9px;
}

.maze-seg button {
  padding: 4px 12px;
  border: none;
  border-radius: 7px;
  background: transparent;
  color: var(--vp-c-text-mute);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}

.maze-seg button.active {
  background: var(--vp-c-accent);
  color: #fff;
}

.maze-btn {
  padding: 5px 12px;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}

.maze-btn:hover:not(:disabled) {
  border-color: var(--vp-c-accent);
  color: var(--vp-c-accent);
}

.maze-btn:active:not(:disabled) {
  transform: translateY(1px);
}

.maze-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.maze-btn.primary {
  background: var(--vp-c-accent);
  border-color: var(--vp-c-accent);
  color: #fff;
}

.maze-btn.primary:hover:not(:disabled) {
  color: #fff;
  filter: brightness(1.08);
}

.maze-hud {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--vp-c-text-mute);
  font-variant-numeric: tabular-nums;
}

.maze-hud b {
  color: var(--vp-c-text);
}

.maze-wrap {
  position: relative;
  max-width: 562px;
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  background: var(--vp-c-bg-elv);
  overflow: hidden;
  outline: none;
}

.maze-wrap:focus-visible {
  box-shadow: 0 0 0 2px var(--vp-c-accent-soft, var(--vp-c-accent));
}

.maze-wrap canvas {
  display: block;
  touch-action: none;
  cursor: pointer;
}

.maze-chip {
  position: absolute;
  top: 10px;
  left: 10px;
  padding: 3px 10px;
  border-radius: 999px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-border);
  font-size: 12px;
  color: var(--vp-c-text-mute);
  pointer-events: none;
}

.maze-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
}

.maze-overlay::before {
  content: "";
  position: absolute;
  inset: 0;
  background: var(--vp-c-bg);
  opacity: 0.62;
}

.maze-card {
  position: relative;
  max-width: 320px;
  padding: 20px 24px;
  border: 1px solid var(--vp-c-border);
  border-radius: 14px;
  background: var(--vp-c-bg-elv);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.14);
  text-align: center;
  animation: maze-pop 0.28s ease;
}

@keyframes maze-pop {
  from {
    transform: scale(0.9);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

.maze-card-title {
  margin-bottom: 8px;
  font-size: 17px;
  font-weight: 600;
}

.maze-card-text {
  margin: 0 0 14px;
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--vp-c-text-mute);
}

.maze-card-text b {
  color: var(--vp-c-accent);
}

.maze-card-sub {
  margin: 10px 0 0;
  font-size: 11.5px;
  color: var(--vp-c-text-mute);
  opacity: 0.75;
}

.maze-card-actions {
  display: flex;
  justify-content: center;
  gap: 10px;
}

.maze-record {
  margin: -6px 0 12px;
  font-size: 13px;
  color: #f2a93b;
  font-weight: 600;
}

.maze-rating {
  width: 60px;
  height: 60px;
  margin: 0 auto 10px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  font-size: 30px;
  font-weight: 700;
  animation: maze-pop 0.4s ease 0.1s backwards;
}

.maze-rating[data-r="S"] {
  background: rgba(242, 169, 59, 0.16);
  color: #e0961f;
}

.maze-rating[data-r="A"] {
  background: rgba(62, 175, 124, 0.16);
  color: #2f9e6b;
}

.maze-rating[data-r="B"] {
  background: rgba(76, 139, 245, 0.16);
  color: #4c8bf5;
}

.maze-rating[data-r="C"] {
  background: rgba(128, 128, 128, 0.16);
  color: var(--vp-c-text-mute);
}

.maze-dpad {
  display: none;
  grid-template-columns: repeat(3, 54px);
  gap: 6px;
  justify-content: center;
  margin-top: 14px;
}

@media (pointer: coarse) {
  .maze-dpad {
    display: grid;
  }
}

.maze-dpad button {
  height: 46px;
  border: 1px solid var(--vp-c-border);
  border-radius: 10px;
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text);
  font-size: 15px;
  cursor: pointer;
}

.maze-dpad button:active {
  background: var(--vp-c-accent-soft, var(--vp-c-bg-soft));
  border-color: var(--vp-c-accent);
}

.maze-tip {
  margin-top: 10px;
  font-size: 12px;
  color: var(--vp-c-text-mute);
  text-align: center;
}
</style>
