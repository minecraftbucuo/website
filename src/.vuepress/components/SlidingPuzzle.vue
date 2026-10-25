<template>
  <div class="slide-game">
    <div class="slide-toolbar">
      <div class="slide-seg" role="group" aria-label="棋盘大小">
        <button
          v-for="k in SIZES"
          :key="k"
          :class="{ active: n === k }"
          @click="setSize(k)"
        >
          {{ k }}×{{ k }}<span v-if="bests[k]" class="slide-cleared">✓</span>
        </button>
      </div>
      <button class="slide-btn" @click="scramble">🔀 重新打乱</button>
      <button class="slide-btn" @click="phase === 'demo' ? skipDemo() : startDemo()">
        {{ phase === 'demo' ? '⏭ 跳过演示' : '🤖 A* 求解演示' }}
      </button>
    </div>

    <div class="slide-hud">
      <span>👣 步数 <b>{{ moves }}</b></span>
      <span>{{ n === 3 ? '🎯 最少' : '🤖 参考' }} <b>{{ par }}</b> 步</span>
      <span>⏱ {{ timeText }}</span>
      <span v-if="bestText">🏆 最佳 {{ bestText }}</span>
    </div>

    <div
      ref="boardRef"
      class="slide-board"
      :class="{ won: phase === 'won' }"
      tabindex="0"
      aria-label="数字华容道棋盘，点击后可用方向键操作"
      @keydown="onKey"
    >
      <div
        v-for="t in tiles"
        :key="t"
        class="slide-tile"
        :class="{ correct: isCorrect(t) }"
        :style="tileStyle(t)"
        @click="tileClick(t)"
      >{{ t }}</div>

      <div v-if="phase === 'demo'" class="slide-chip">🤖 A* 演示中… {{ demoIdx }}/{{ demoLen }}</div>

      <div v-if="phase === 'demoDone'" class="slide-overlay">
        <div class="slide-card">
          <div class="slide-card-title">🤖 演示完毕</div>
          <p class="slide-card-text">
            从刚才的局面复原共 <b>{{ demoLen }}</b> 步<br />
            {{ n === 3 ? '这就是 A* 找到的最优解' : '这是 A* 给出的参考解' }}
          </p>
          <div class="slide-card-actions">
            <button class="slide-btn primary" @click="scramble">我自己来</button>
          </div>
        </div>
      </div>

      <div v-else-if="phase === 'won' && winCard" class="slide-overlay">
        <div class="slide-card">
          <div class="slide-rating" :data-r="rating">{{ rating }}</div>
          <div class="slide-card-title">{{ ratingTitle }}</div>
          <p class="slide-card-text">
            用了 <b>{{ moves }}</b> 步 · {{ n === 3 ? '最少' : '参考' }} {{ par }} 步 · 用时 {{ timeText }}
          </p>
          <p v-if="newRecord" class="slide-record">🎉 新纪录！</p>
          <div class="slide-card-actions">
            <button class="slide-btn primary" @click="scramble">再来一局</button>
          </div>
        </div>
      </div>
    </div>

    <p class="slide-tip">点击与空格同行/同列的方块即可滑动（可一次推多块），也可以点一下棋盘后用方向键。</p>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

const SIZES = [3, 4];
const BEST_KEY = "slide-puzzle-best-v1";
type Best = { steps: number; ms: number };

const n = ref(3);
const cells = ref<number[]>([]);
const phase = ref<"playing" | "demo" | "demoDone" | "won">("playing");
const moves = ref(0);
const par = ref(0);
const timeText = ref("0:00");
const winCard = ref(false);
const newRecord = ref(false);
const demoIdx = ref(0);
const demoLen = ref(0);

let startTime = 0;
let finalMs = 0;
let movedOnce = false;
let timerId = 0;
let demoTimer = 0;
let cardTimer = 0;
let resizeObserver: ResizeObserver | null = null;

const boardRef = ref<HTMLElement | null>(null);
const boardW = ref(400);

// 方块列表固定按编号升序（与棋盘状态无关），保证 keyed diff 不搬家 DOM 节点，
// 否则节点重排会打断 CSS 过渡，纵向滑动会变成瞬移
const tiles = computed(() => Array.from({ length: n.value * n.value - 1 }, (_, i) => i + 1));

// ---------- 纪录 ----------
const bests = reactive<Record<number, Best>>({});
try {
  Object.assign(bests, JSON.parse(localStorage.getItem(BEST_KEY) || "{}"));
} catch {
  /* 隐私模式下静默失败 */
}
const bestText = computed(() => {
  const b = bests[n.value];
  return b ? `${b.steps} 步` : "";
});

// ---------- 评分 ----------
const rating = computed<"S" | "A" | "B" | "C">(() => {
  const r = moves.value / Math.max(1, par.value);
  return r <= 1 ? "S" : r <= 1.25 ? "A" : r <= 1.6 ? "B" : "C";
});
const ratingTitle = computed(
  () =>
    ({
      S: n.value === 3 ? "贴着最优解通关！A* 都挑不出毛病" : "比 A* 的参考解还漂亮！",
      A: "非常接近参考解，稳！",
      B: "通关！路线还能再紧凑点",
      C: "通关啦～完成就是胜利",
    })[rating.value],
);

// ---------- 工具 ----------
function solved(n: number) {
  return Array.from({ length: n * n }, (_, i) => (i + 1) % (n * n));
}
function isSolved(arr: number[]) {
  return arr.every((v, i) => v === (i + 1) % arr.length);
}
function isCorrect(t: number) {
  return cells.value.indexOf(t) === t - 1;
}
function neighbors(i: number, n: number) {
  const x = i % n;
  const y = (i / n) | 0;
  const out: number[] = [];
  if (y > 0) out.push(i - n);
  if (x < n - 1) out.push(i + 1);
  if (y < n - 1) out.push(i + n);
  if (x > 0) out.push(i - 1);
  return out;
}

// ---------- 打乱（随机游走，保证可解） ----------
function scrambled(n: number) {
  const arr = solved(n);
  let blank = arr.length - 1;
  let last = -1;
  const steps = n === 3 ? 120 : 300;
  for (let s = 0; s < steps; s++) {
    const opts = neighbors(blank, n).filter((i) => i !== last);
    const pick = opts[(Math.random() * opts.length) | 0];
    arr[blank] = arr[pick];
    arr[pick] = 0;
    last = blank;
    blank = pick;
  }
  if (isSolved(arr)) return scrambled(n);
  return arr;
}

// ---------- A* 求解器 ----------
function manhattan(arr: number[], n: number) {
  let h = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v === 0) continue;
    const goal = v - 1;
    h += Math.abs((i % n) - (goal % n)) + Math.abs(((i / n) | 0) - ((goal / n) | 0));
  }
  return h;
}

// 小顶堆
class Heap {
  a: { f: number; g: number; key: string }[] = [];
  push(x: { f: number; g: number; key: string }) {
    const a = this.a;
    a.push(x);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p].f <= a[i].f) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop()!;
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && a[l].f < a[m].f) m = l;
        if (r < a.length && a[r].f < a[m].f) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
  get size() {
    return this.a.length;
  }
}

// 返回空白格的移动序列（下标），失败返回 null
function aStar(start: number[], n: number): number[] | null {
  const goalKey = solved(n).join(",");
  const w = n === 3 ? 1 : 2.5; // 4×4 用加权 A* 换速度
  const CAP = 500000;
  const startKey = start.join(",");
  if (startKey === goalKey) return [];
  const h0 = manhattan(start, n);
  const open = new Heap();
  open.push({ f: w * h0, g: 0, key: startKey });
  const gScore = new Map<string, number>([[startKey, 0]]);
  const parent = new Map<string, { key: string; blank: number }>();
  let expanded = 0;
  while (open.size) {
    const cur = open.pop();
    if (cur.g !== gScore.get(cur.key)) continue; // 过期堆项
    if (cur.key === goalKey) {
      // 回溯
      const path: number[] = [];
      let k = cur.key;
      while (k !== startKey) {
        const p = parent.get(k)!;
        path.unshift(p.blank);
        k = p.key;
      }
      return path;
    }
    if (++expanded > CAP) return null;
    const arr = cur.key.split(",").map(Number);
    const blank = arr.indexOf(0);
    const g2 = cur.g + 1;
    for (const nb of neighbors(blank, n)) {
      arr[blank] = arr[nb];
      arr[nb] = 0;
      const key2 = arr.join(",");
      if (g2 < (gScore.get(key2) ?? Infinity)) {
        gScore.set(key2, g2);
        parent.set(key2, { key: cur.key, blank: nb });
        open.push({ f: g2 + w * manhattan(arr, n), g: g2, key: key2 });
      }
      arr[nb] = arr[blank];
      arr[blank] = 0;
    }
  }
  return null;
}

// ---------- 流程 ----------
function scramble() {
  stopDemo();
  cells.value = scrambled(n.value);
  moves.value = 0;
  movedOnce = false;
  finalMs = 0;
  timeText.value = "0:00";
  winCard.value = false;
  newRecord.value = false;
  phase.value = "playing";
  const plan = aStar(cells.value, n.value);
  par.value = plan ? plan.length : manhattan(cells.value, n.value) * 2;
}

function setSize(k: number) {
  if (phase.value === "demo") stopDemo();
  n.value = k;
  scramble();
}

function slideTo(i: number) {
  const blank = cells.value.indexOf(0);
  const dist = Math.abs((i % n.value) - (blank % n.value)) + Math.abs(((i / n.value) | 0) - ((blank / n.value) | 0));
  if (i === blank || (i % n.value !== blank % n.value && ((i / n.value) | 0) !== ((blank / n.value) | 0))) return false;
  const arr = cells.value;
  const step = i > blank ? 1 : -1;
  const horiz = i % n.value !== blank % n.value;
  let cur = blank;
  for (let k = 0; k < dist; k++) {
    const next = horiz ? cur + step : cur + step * n.value;
    arr[cur] = arr[next];
    arr[next] = 0;
    cur = next;
    moves.value++;
  }
  return true;
}

function tileClick(t: number) {
  if (phase.value !== "playing") return;
  const i = cells.value.indexOf(t);
  if (!slideTo(i)) return;
  if (!movedOnce) {
    movedOnce = true;
    startTime = performance.now();
    startTimer();
  }
  if (isSolved(cells.value)) finishWin();
}

function onKey(e: KeyboardEvent) {
  if (phase.value !== "playing") return;
  const map: Record<string, number> = { ArrowUp: -n.value, ArrowDown: n.value, ArrowLeft: -1, ArrowRight: 1 };
  const d = map[e.key];
  if (d === undefined) return;
  e.preventDefault();
  const blank = cells.value.indexOf(0);
  const x = blank % n.value;
  if ((d === -1 && x === 0) || (d === 1 && x === n.value - 1)) return;
  const nb = blank + d;
  if (nb < 0 || nb >= cells.value.length) return;
  const arr = cells.value;
  arr[blank] = arr[nb];
  arr[nb] = 0;
  moves.value++;
  if (!movedOnce) {
    movedOnce = true;
    startTime = performance.now();
    startTimer();
  }
  if (isSolved(cells.value)) finishWin();
}

function finishWin() {
  phase.value = "won";
  finalMs = movedOnce ? performance.now() - startTime : 0;
  timeText.value = fmt(finalMs);
  stopTimer();
  const b = bests[n.value];
  const better =
    !b || moves.value < b.steps || (moves.value === b.steps && finalMs < b.ms);
  newRecord.value = better;
  if (better) {
    bests[n.value] = { steps: moves.value, ms: Math.round(finalMs) };
    try {
      localStorage.setItem(BEST_KEY, JSON.stringify(bests));
    } catch {
      /* 隐私模式下静默失败 */
    }
  }
  clearTimeout(cardTimer);
  cardTimer = window.setTimeout(() => (winCard.value = true), 600);
}

// ---------- 演示 ----------
function startDemo() {
  const plan = aStar(cells.value, n.value);
  if (!plan || !plan.length) return;
  phase.value = "demo";
  demoLen.value = plan.length;
  demoIdx.value = 0;
  const interval = Math.max(170, Math.min(280, 2600 / plan.length));
  demoTimer = window.setInterval(() => {
    if (demoIdx.value >= demoLen.value) {
      stopDemoTimer();
      phase.value = "demoDone";
      return;
    }
    const nb = plan[demoIdx.value++];
    const blank = cells.value.indexOf(0);
    cells.value[blank] = cells.value[nb];
    cells.value[nb] = 0;
  }, interval);
}

function stopDemoTimer() {
  clearInterval(demoTimer);
  demoTimer = 0;
}

function stopDemo() {
  stopDemoTimer();
}

function skipDemo() {
  // 直接把剩余演示步走完（不逐帧动画）
  stopDemo();
  phase.value = "playing";
  scramble();
}

// ---------- 计时 ----------
function startTimer() {
  stopTimer();
  timerId = window.setInterval(() => {
    if (phase.value === "playing" && movedOnce)
      timeText.value = fmt(performance.now() - startTime);
  }, 500);
}
function stopTimer() {
  clearInterval(timerId);
  timerId = 0;
}

function fmt(ms: number) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// ---------- 渲染 ----------
function tileStyle(t: number) {
  const i = cells.value.indexOf(t);
  const k = 100 / n.value;
  return {
    left: `calc(${(i % n.value) * k}% + 3px)`,
    top: `calc(${((i / n.value) | 0) * k}% + 3px)`,
    width: `calc(${k}% - 6px)`,
    height: `calc(${k}% - 6px)`,
    fontSize: `${Math.max(13, Math.round(boardW.value / n.value / 3.2))}px`,
    zIndex: 5,
  };
}

// 初始化（同步，走完整打乱流程，首帧就有参考步数）
scramble();

onMounted(() => {
  if (boardRef.value) {
    boardW.value = boardRef.value.clientWidth;
    resizeObserver = new ResizeObserver(() => {
      if (boardRef.value) boardW.value = boardRef.value.clientWidth;
    });
    resizeObserver.observe(boardRef.value);
  }
});

onBeforeUnmount(() => {
  stopTimer();
  stopDemoTimer();
  clearTimeout(cardTimer);
  resizeObserver?.disconnect();
});
</script>

<style scoped>
.slide-game {
  max-width: 520px;
  margin: 1rem auto;
}

.slide-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.slide-seg {
  display: inline-flex;
  padding: 2px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 9px;
}

.slide-seg button {
  padding: 4px 12px;
  border: none;
  border-radius: 7px;
  background: transparent;
  color: var(--vp-c-text-mute);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}

.slide-seg button.active {
  background: var(--vp-c-accent);
  color: #fff;
}

.slide-cleared {
  margin-left: 2px;
  font-size: 11px;
}

.slide-btn {
  padding: 5px 12px;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}

.slide-btn:hover:not(:disabled) {
  border-color: var(--vp-c-accent);
  color: var(--vp-c-accent);
}

.slide-btn:active:not(:disabled) {
  transform: translateY(1px);
}

.slide-btn.primary {
  background: var(--vp-c-accent);
  border-color: var(--vp-c-accent);
  color: #fff;
}

.slide-btn.primary:hover {
  color: #fff;
  filter: brightness(1.08);
}

.slide-hud {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--vp-c-text-mute);
  font-variant-numeric: tabular-nums;
}

.slide-hud b {
  color: var(--vp-c-text);
}

.slide-board {
  position: relative;
  aspect-ratio: 1 / 1;
  max-width: 520px;
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  background: var(--vp-c-bg-elv);
  overflow: hidden;
  outline: none;
  touch-action: manipulation;
}

.slide-board:focus-visible {
  box-shadow: 0 0 0 2px var(--vp-c-accent-soft, var(--vp-c-accent));
}

.slide-board.won .slide-tile {
  animation: slide-glow 0.9s ease;
}

@keyframes slide-glow {
  0%,
  100% {
    filter: brightness(1);
  }
  40% {
    filter: brightness(1.35);
  }
}

.slide-tile {
  position: absolute;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: var(--vp-c-bg-soft);
  border: 1.5px solid var(--vp-c-border);
  color: var(--vp-c-text);
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  cursor: pointer;
  user-select: none;
  transition: left 0.16s ease, top 0.16s ease, border-color 0.2s, color 0.2s;
}

.slide-tile.correct {
  border-color: var(--vp-c-accent);
  color: var(--vp-c-accent);
}

.slide-chip {
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
  z-index: 30;
}

.slide-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  z-index: 40;
}

.slide-overlay::before {
  content: "";
  position: absolute;
  inset: 0;
  background: var(--vp-c-bg);
  opacity: 0.62;
}

.slide-card {
  position: relative;
  max-width: 320px;
  padding: 20px 24px;
  border: 1px solid var(--vp-c-border);
  border-radius: 14px;
  background: var(--vp-c-bg-elv);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.14);
  text-align: center;
  animation: slide-pop 0.28s ease;
}

@keyframes slide-pop {
  from {
    transform: scale(0.9);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

.slide-card-title {
  margin-bottom: 8px;
  font-size: 17px;
  font-weight: 600;
}

.slide-card-text {
  margin: 0 0 14px;
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--vp-c-text-mute);
}

.slide-card-text b {
  color: var(--vp-c-accent);
}

.slide-card-actions {
  display: flex;
  justify-content: center;
  gap: 10px;
}

.slide-record {
  margin: -6px 0 12px;
  font-size: 13px;
  color: #f2a93b;
  font-weight: 600;
}

.slide-rating {
  width: 60px;
  height: 60px;
  margin: 0 auto 10px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  font-size: 30px;
  font-weight: 700;
  animation: slide-pop 0.4s ease 0.1s backwards;
}

.slide-rating[data-r="S"] {
  background: rgba(242, 169, 59, 0.16);
  color: #e0961f;
}

.slide-rating[data-r="A"] {
  background: rgba(62, 175, 124, 0.16);
  color: #2f9e6b;
}

.slide-rating[data-r="B"] {
  background: rgba(76, 139, 245, 0.16);
  color: #4c8bf5;
}

.slide-rating[data-r="C"] {
  background: rgba(128, 128, 128, 0.16);
  color: var(--vp-c-text-mute);
}

.slide-tip {
  margin-top: 10px;
  font-size: 12px;
  color: var(--vp-c-text-mute);
  text-align: center;
}

@media (prefers-reduced-motion: reduce) {
  .slide-tile,
  .slide-card,
  .slide-rating {
    transition: none;
    animation: none;
  }
}
</style>
