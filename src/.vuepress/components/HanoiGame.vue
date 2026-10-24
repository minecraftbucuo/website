<template>
  <div class="hanoi-game">
    <div class="hanoi-toolbar">
      <div class="hanoi-seg" role="group" aria-label="盘数（关卡）">
        <button
          v-for="k in LEVELS"
          :key="k"
          :class="{ active: n === k }"
          @click="setLevel(k)"
        >
          {{ k }} 层<span v-if="bests[k]" class="hanoi-cleared">✓</span>
        </button>
      </div>
      <button class="hanoi-btn" :disabled="!canUndo" @click="undo">↩ 撤销</button>
      <button class="hanoi-btn" @click="reset(true)">↺ 重置</button>
      <button class="hanoi-btn" @click="phase === 'demo' ? skipDemo() : startDemo()">
        {{ phase === 'demo' ? '⏭ 跳过演示' : '🎬 最优解演示' }}
      </button>
    </div>

    <div class="hanoi-hud">
      <span>🎯 目标：全部搬到 C 柱</span>
      <span>👣 步数 <b>{{ moves }}</b> / 最少 {{ minSteps }}</span>
      <span>⏱ {{ timeText }}</span>
      <span v-if="bestText">🏆 最佳 {{ bestText }}</span>
    </div>

    <div class="hanoi-board" :class="{ won: phase === 'won' }" :style="{ height: boardH + 'px' }">
      <div
        v-for="(peg, i) in pegs"
        :key="i"
        class="hanoi-peg"
        :class="{ picked: picked === i, shake: shakePeg === i }"
        role="button"
        :aria-label="`柱子 ${'ABC'[i]}，${peg.length} 个盘`"
        @click="pegClick(i)"
      >
        <div class="hanoi-rod" :style="{ height: rodH + 'px' }"></div>
        <div class="hanoi-base"></div>
        <span class="hanoi-label">{{ 'ABC'[i] }}</span>
      </div>

      <template v-if="ready">
        <div
          v-for="d in n"
          :key="d"
          class="hanoi-disk"
          :class="{ lifted: isLifted(d) }"
          :style="diskStyle(d)"
        >{{ d }}</div>
      </template>

      <div v-if="phase === 'demo'" class="hanoi-chip">🎬 递归演示中…</div>

      <div v-if="phase === 'demoDone'" class="hanoi-overlay">
        <div class="hanoi-card">
          <div class="hanoi-card-title">🎬 演示完毕</div>
          <p class="hanoi-card-text">
            {{ n }} 层汉诺塔最少需要 <b>{{ minSteps }}</b> 步<br />
            刚才就是完整的递归流程
          </p>
          <div class="hanoi-card-actions">
            <button class="hanoi-btn primary" @click="reset(true)">我自己来</button>
          </div>
        </div>
      </div>

      <div v-else-if="phase === 'won' && winCard" class="hanoi-overlay">
        <div class="hanoi-card">
          <div class="hanoi-rating" :data-r="rating">{{ rating }}</div>
          <div class="hanoi-card-title">{{ ratingTitle }}</div>
          <p class="hanoi-card-text">
            {{ n }} 层用了 <b>{{ moves }}</b> 步 · 最少 {{ minSteps }} 步 · 用时 {{ timeText }}
          </p>
          <p v-if="newRecord" class="hanoi-record">🎉 新纪录！</p>
          <div class="hanoi-card-actions">
            <button v-if="n < MAX_LEVEL" class="hanoi-btn primary" @click="setLevel(n + 1)">
              下一关 →
            </button>
            <button class="hanoi-btn" @click="reset(true)">再来一次</button>
          </div>
        </div>
      </div>
    </div>

    <p class="hanoi-tip">点柱子拿起最上面的盘，再点目标柱放下；大盘不能压小盘。</p>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

const MAX_LEVEL = 8;
const MIN_LEVEL = 3;
const LEVELS = Array.from(
  { length: MAX_LEVEL - MIN_LEVEL + 1 },
  (_, i) => i + MIN_LEVEL,
);
const BEST_KEY = "hanoi-best-v1";
type Best = { steps: number; ms: number };

const n = ref(3);
const ready = ref(false);
const phase = ref<"playing" | "demo" | "demoDone" | "won">("playing");
const pegs = reactive<number[][]>([[], [], []]);
const picked = ref(-1);
const shakePeg = ref(-1);
const moves = ref(0);
const timeText = ref("0:00");
const winCard = ref(false);
const newRecord = ref(false);

const history = reactive<[number, number][]>([]);
let startTime = 0;
let finalMs = 0;
let movedOnce = false;
let timerId = 0;
let shakeTimer = 0;
let demoTimer = 0;
let demoList: [number, number][] = [];
let demoIdx = 0;
let cardTimer = 0;

const minSteps = computed(() => 2 ** n.value - 1);
const canUndo = computed(
  () => phase.value === "playing" && history.length > 0,
);

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
  const r = moves.value / minSteps.value;
  return r <= 1 ? "S" : r <= 1.25 ? "A" : r <= 1.6 ? "B" : "C";
});
const ratingTitle = computed(
  () =>
    ({
      S: "最少步数通关！递归大师",
      A: "非常接近最优，稳！",
      B: "通关！还有优化空间",
      C: "通关啦～路是走出来的",
    })[rating.value],
);

// ---------- 几何 ----------
const diskH = computed(() => (n.value <= 4 ? 34 : n.value <= 6 ? 28 : 24));
const rodH = computed(() => diskH.value * (n.value + 1) + 12);
const boardH = computed(() => rodH.value + 56);

// 每个盘当前所在（柱，槽位）
const pos = computed(() => {
  const map = new Map<number, { peg: number; slot: number }>();
  pegs.forEach((stack, pi) =>
    stack.forEach((d, si) => map.set(d, { peg: pi, slot: si })),
  );
  return map;
});

function isLifted(d: number) {
  const p = pos.value.get(d);
  return !!p && picked.value === p.peg && p.slot === pegs[p.peg].length - 1;
}

function diskStyle(d: number) {
  const { peg, slot } = pos.value.get(d)!;
  const width = (100 / 3) * (0.3 + (0.6 * d) / n.value);
  return {
    left: `${(peg + 0.5) * (100 / 3)}%`,
    top: `${rodH.value + 14 - (slot + 1) * diskH.value}px`,
    width: `${width}%`,
    height: `${diskH.value - 4}px`,
    background: `hsl(${198 + ((d - 1) * 150) / Math.max(1, n.value - 1)}, 68%, 58%)`,
    fontSize: `${Math.max(11, Math.round(diskH.value * 0.45))}px`,
    zIndex: isLifted(d) ? 20 : 5,
  };
}

// ---------- 流程 ----------
function resetBoard(autoStart = true) {
  pegs[0] = Array.from({ length: n.value }, (_, i) => n.value - i);
  pegs[1] = [];
  pegs[2] = [];
  picked.value = -1;
  moves.value = 0;
  history.length = 0;
  movedOnce = false;
  startTime = 0;
  finalMs = 0;
  timeText.value = "0:00";
  winCard.value = false;
  newRecord.value = false;
  ready.value = true;
  phase.value = autoStart ? "playing" : "demo";
}

function reset(autoStart = true) {
  stopDemo();
  resetBoard(autoStart);
}

function setLevel(k: number) {
  if (phase.value === "demo") stopDemo();
  n.value = k;
  resetBoard(true);
}

function pegClick(i: number) {
  if (phase.value !== "playing") return;
  if (picked.value === -1) {
    if (pegs[i].length) picked.value = i;
    return;
  }
  if (picked.value === i) {
    picked.value = -1;
    return;
  }
  const disk = pegs[picked.value][pegs[picked.value].length - 1];
  const target = pegs[i][pegs[i].length - 1];
  if (target !== undefined && target < disk) {
    // 大盘压小盘：拒绝并抖动提示
    shakePeg.value = i;
    clearTimeout(shakeTimer);
    shakeTimer = window.setTimeout(() => (shakePeg.value = -1), 350);
    return;
  }
  pegs[picked.value].pop();
  pegs[i].push(disk);
  history.push([picked.value, i]);
  picked.value = -1;
  moves.value++;
  if (!movedOnce) {
    movedOnce = true;
    startTime = performance.now();
    startTimer();
  }
  if (pegs[2].length === n.value) finishWin();
}

function undo() {
  if (phase.value !== "playing" || !history.length) return;
  const [from, to] = history.pop()!;
  const disk = pegs[to].pop();
  if (disk !== undefined) pegs[from].push(disk);
  picked.value = -1;
  moves.value = Math.max(0, moves.value - 1);
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
  cardTimer = window.setTimeout(() => (winCard.value = true), 650);
}

// ---------- 最优解演示 ----------
function optimalMoves(k: number, from: number, to: number, via: number, out: [number, number][]) {
  if (k === 0) return;
  optimalMoves(k - 1, from, via, to, out);
  out.push([from, to]);
  optimalMoves(k - 1, via, to, from, out);
}

function startDemo() {
  resetBoard(false);
  phase.value = "demo";
  demoList = [];
  optimalMoves(n.value, 0, 2, 1, demoList);
  demoIdx = 0;
  const interval = Math.max(70, Math.min(350, 2600 / demoList.length));
  demoTimer = window.setInterval(() => {
    if (demoIdx >= demoList.length) {
      stopDemoTimer();
      phase.value = "demoDone";
      return;
    }
    const [from, to] = demoList[demoIdx++];
    const disk = pegs[from].pop();
    if (disk !== undefined) pegs[to].push(disk);
    moves.value++;
  }, interval);
}

function stopDemoTimer() {
  clearInterval(demoTimer);
  demoTimer = 0;
}

function stopDemo() {
  stopDemoTimer();
  demoList = [];
  demoIdx = 0;
}

function skipDemo() {
  stopDemo();
  resetBoard(true);
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

onMounted(() => {
  resetBoard(true);
});
onBeforeUnmount(() => {
  stopTimer();
  stopDemo();
  clearTimeout(shakeTimer);
  clearTimeout(cardTimer);
});
</script>

<style scoped>
.hanoi-game {
  max-width: 580px;
  margin: 1rem auto;
}

.hanoi-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.hanoi-seg {
  display: inline-flex;
  padding: 2px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 9px;
}

.hanoi-seg button {
  padding: 4px 10px;
  border: none;
  border-radius: 7px;
  background: transparent;
  color: var(--vp-c-text-mute);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}

.hanoi-seg button.active {
  background: var(--vp-c-accent);
  color: #fff;
}

.hanoi-cleared {
  margin-left: 2px;
  font-size: 11px;
}

.hanoi-btn {
  padding: 5px 12px;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}

.hanoi-btn:hover:not(:disabled) {
  border-color: var(--vp-c-accent);
  color: var(--vp-c-accent);
}

.hanoi-btn:active:not(:disabled) {
  transform: translateY(1px);
}

.hanoi-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.hanoi-btn.primary {
  background: var(--vp-c-accent);
  border-color: var(--vp-c-accent);
  color: #fff;
}

.hanoi-btn.primary:hover:not(:disabled) {
  color: #fff;
  filter: brightness(1.08);
}

.hanoi-hud {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--vp-c-text-mute);
  font-variant-numeric: tabular-nums;
}

.hanoi-hud b {
  color: var(--vp-c-text);
}

.hanoi-board {
  position: relative;
  max-width: 562px;
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  background: var(--vp-c-bg-elv);
  overflow: hidden;
}

.hanoi-peg {
  position: absolute;
  top: 0;
  bottom: 0;
  cursor: pointer;
}

.hanoi-peg:nth-child(1) {
  left: 0;
  width: 33.333%;
}

.hanoi-peg:nth-child(2) {
  left: 33.333%;
  width: 33.333%;
}

.hanoi-peg:nth-child(3) {
  left: 66.666%;
  width: 33.334%;
}

.hanoi-peg.picked::before,
.hanoi-peg:hover::before {
  content: "";
  position: absolute;
  inset: 8px 6px 40px;
  border-radius: 10px;
  background: var(--vp-c-accent-soft, var(--vp-c-bg-soft));
  pointer-events: none;
}

.hanoi-rod {
  position: absolute;
  bottom: 40px;
  left: 50%;
  width: 8px;
  transform: translateX(-50%);
  border-radius: 4px 4px 0 0;
  background: var(--vp-c-border);
}

.hanoi-base {
  position: absolute;
  bottom: 36px;
  left: 12%;
  right: 12%;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-border);
}

.hanoi-label {
  position: absolute;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-mute);
}

.hanoi-disk {
  position: absolute;
  transform: translateX(-50%);
  display: grid;
  place-items: center;
  font-weight: 700;
  color: rgba(255, 255, 255, 0.92);
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
  user-select: none;
  pointer-events: none;
  border-radius: 999px;
  box-shadow: inset 0 -3px 0 rgba(0, 0, 0, 0.18), 0 1px 3px rgba(0, 0, 0, 0.2);
  transition: left 0.28s ease, top 0.28s ease, box-shadow 0.2s;
}

.hanoi-disk.lifted {
  top: 14px !important;
  box-shadow: inset 0 -3px 0 rgba(0, 0, 0, 0.18),
    0 8px 18px rgba(0, 0, 0, 0.35);
}

.hanoi-board.won .hanoi-disk {
  animation: hanoi-glow 0.9s ease;
}

@keyframes hanoi-glow {
  0%,
  100% {
    filter: brightness(1);
  }
  40% {
    filter: brightness(1.45) saturate(1.3);
  }
}

.hanoi-peg.shake .hanoi-disk:last-child {
  animation: hanoi-shake 0.3s ease;
}

@keyframes hanoi-shake {
  0%,
  100% {
    transform: translateX(-50%);
  }
  30% {
    transform: translateX(calc(-50% - 5px));
  }
  60% {
    transform: translateX(calc(-50% + 5px));
  }
}

.hanoi-chip {
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

.hanoi-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  z-index: 40;
}

.hanoi-overlay::before {
  content: "";
  position: absolute;
  inset: 0;
  background: var(--vp-c-bg);
  opacity: 0.62;
}

.hanoi-card {
  position: relative;
  max-width: 320px;
  padding: 20px 24px;
  border: 1px solid var(--vp-c-border);
  border-radius: 14px;
  background: var(--vp-c-bg-elv);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.14);
  text-align: center;
  animation: hanoi-pop 0.28s ease;
}

@keyframes hanoi-pop {
  from {
    transform: scale(0.9);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

.hanoi-card-title {
  margin-bottom: 8px;
  font-size: 17px;
  font-weight: 600;
}

.hanoi-card-text {
  margin: 0 0 14px;
  font-size: 13.5px;
  line-height: 1.7;
  color: var(--vp-c-text-mute);
}

.hanoi-card-text b {
  color: var(--vp-c-accent);
}

.hanoi-card-actions {
  display: flex;
  justify-content: center;
  gap: 10px;
}

.hanoi-record {
  margin: -6px 0 12px;
  font-size: 13px;
  color: #f2a93b;
  font-weight: 600;
}

.hanoi-rating {
  width: 60px;
  height: 60px;
  margin: 0 auto 10px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  font-size: 30px;
  font-weight: 700;
  animation: hanoi-pop 0.4s ease 0.1s backwards;
}

.hanoi-rating[data-r="S"] {
  background: rgba(242, 169, 59, 0.16);
  color: #e0961f;
}

.hanoi-rating[data-r="A"] {
  background: rgba(62, 175, 124, 0.16);
  color: #2f9e6b;
}

.hanoi-rating[data-r="B"] {
  background: rgba(76, 139, 245, 0.16);
  color: #4c8bf5;
}

.hanoi-rating[data-r="C"] {
  background: rgba(128, 128, 128, 0.16);
  color: var(--vp-c-text-mute);
}

.hanoi-tip {
  margin-top: 10px;
  font-size: 12px;
  color: var(--vp-c-text-mute);
  text-align: center;
}

@media (prefers-reduced-motion: reduce) {
  .hanoi-disk,
  .hanoi-card,
  .hanoi-rating {
    transition: none;
    animation: none;
  }
}
</style>
