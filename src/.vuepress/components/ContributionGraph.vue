<template>
  <div class="cgraph-wrap">
    <!-- 左侧：外卡片（大标题 + 内嵌图表面板），GitHub 同款结构 -->
    <div class="cgraph-main" :style="cardVars">
      <div class="cgraph-head">
        <span class="cgraph-title">{{ total }} 次提交{{ rangeText }}</span>
        <select v-model="selectedYear" class="cgraph-select" aria-label="选择年份">
          <option v-for="y in years" :key="y" :value="y">{{ y }}</option>
        </select>
      </div>

      <div class="cgraph-panel">
        <div ref="graphEl" class="cgraph-scroll">
          <div class="cgraph-stack">
            <!-- 月份标签 -->
            <div class="cgraph-months">
              <span
                v-for="(m, i) in monthLabels"
                :key="i"
                :style="{ gridColumn: `${m.col + 1} / span ${m.span}` }"
              >
                {{ m.text }}
              </span>
            </div>

            <!-- 热力图主体 -->
            <div class="cgraph-body">
              <div class="cgraph-days" aria-hidden="true">
                <span v-for="(d, i) in dayLabels" :key="i" :style="{ gridRow: d.row + 1 }">
                  {{ d.text }}
                </span>
              </div>
              <div class="cgraph-grid">
                <div
                  v-for="(cell, i) in cells"
                  :key="i"
                  class="cgraph-cell"
                  :class="[`lv${cell.level}`, { blank: cell.future }]"
                  @mouseenter="showTip($event, cell)"
                  @mouseleave="hideTip"
                ></div>
              </div>
            </div>
          </div>
        </div>

        <!-- 面板底部：说明 + 图例 -->
        <div class="cgraph-foot">
          <span class="cgraph-note">统计基于本站 Git 提交记录</span>
          <div class="cgraph-legend">
            <span>少</span>
            <i class="cgraph-cell lv0"></i>
            <i class="cgraph-cell lv1"></i>
            <i class="cgraph-cell lv2"></i>
            <i class="cgraph-cell lv3"></i>
            <i class="cgraph-cell lv4"></i>
            <span>多</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 悬浮提示 -->
    <div
      v-if="tip"
      class="cgraph-tip"
      :style="{ left: tip.x + 'px', top: tip.y + 'px' }"
    >
      {{ tip.text }}
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onBeforeUnmount } from "vue";
import contributions from "./contributions.json";

const counts = contributions.counts ?? contributions;
const TODAY = new Date();
const CURRENT_YEAR = TODAY.getFullYear();

const years = [...new Set(Object.keys(counts).map((k) => Number(k.slice(0, 4))))]
  .filter((y) => !Number.isNaN(y))
  .sort((a, b) => b - a);
if (!years.includes(CURRENT_YEAR)) years.unshift(CURRENT_YEAR);

const selectedYear = ref(years[0]);

const pad = (n) => String(n).padStart(2, "0");
const fmt = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

// 构建周列表（列 = 周，行 = 周日~周六）
// 当前年份用「过去一年」滚动窗口（53 周，GitHub 同款）；往年用完整自然年
const weeks = computed(() => {
  const year = selectedYear.value;
  let start;
  let end;

  if (year === CURRENT_YEAR) {
    end = new Date(TODAY);
    start = new Date(TODAY);
    start.setDate(start.getDate() - 364);
  } else {
    end = new Date(year, 11, 31);
    start = new Date(year, 0, 1);
  }
  start.setDate(start.getDate() - start.getDay()); // 对齐到周日

  const result = [];
  const cur = new Date(start);
  while (cur <= end) {
    const week = [];
    for (let i = 0; i < 7; i++) {
      const d = new Date(cur.getFullYear(), cur.getMonth(), cur.getDate());
      const future = d > end;
      const key = fmt(d);
      week.push({
        key,
        count: future ? null : counts[key] ?? 0,
        future,
        month: d.getMonth() + 1,
        day: d.getDate(),
        year: d.getFullYear(),
      });
      cur.setDate(cur.getDate() + 1);
    }
    result.push(week);
  }
  return result;
});

// 按当前视图内提交量的四分位数划分颜色等级
const thresholds = computed(() => {
  const positive = weeks.value
    .flat()
    .filter((d) => !d.future && d.count > 0)
    .map((d) => d.count)
    .sort((a, b) => a - b);
  const pick = (p) => positive[Math.floor(positive.length * p)] ?? 1;
  return [pick(0.25), pick(0.5), pick(0.75)];
});

const cells = computed(() =>
  weeks.value.flat().map((d) => {
    const [q1, q2, q3] = thresholds.value;
    const level =
      d.future || d.count === null
        ? 0
        : d.count === 0
          ? 0
          : d.count <= q1
            ? 1
            : d.count <= q2
              ? 2
              : d.count <= q3
                ? 3
                : 4;
    return { ...d, level };
  }),
);

const total = computed(() =>
  cells.value.reduce((sum, c) => sum + (c.future ? 0 : c.count ?? 0), 0),
);

const rangeText = computed(() =>
  selectedYear.value === CURRENT_YEAR
    ? " · 过去一年"
    : ` · ${selectedYear.value} 年`,
);

// 月份标签：出现在包含某月 1 日的列上；首列补一个月名，与 GitHub 一致
const monthLabels = computed(() => {
  const list = [];
  const seen = new Set();
  weeks.value.forEach((week, col) => {
    for (const d of week) {
      if (d.future) continue;
      if (d.day === 1 && !seen.has(d.month) && col > 0) {
        seen.add(d.month);
        list.push({ col, month: d.month });
        break;
      }
    }
  });

  // 窗口起始列若不是某月 1 日，补上该列所属月份
  if (list.length === 0 || list[0].col > 0) {
    const firstDays = weeks.value[0].filter((d) => !d.future);
    const first = firstDays[firstDays.length - 1] ?? weeks.value[0][0];
    list.unshift({ col: 0, month: first.month });
  }

  return list.map((m, i) => ({
    ...m,
    text: `${m.month}月`,
    span: (list[i + 1]?.col ?? weeks.value.length) - m.col,
  }));
});

// 左侧星期标签（周一 / 周三 / 周五）
const dayLabels = [
  { row: 1, text: "一" },
  { row: 3, text: "三" },
  { row: 5, text: "五" },
];

// 悬浮提示
const tip = ref(null);
const showTip = (event, cell) => {
  if (cell.future) return;
  const text =
    cell.count === 0
      ? `${cell.year}年${cell.month}月${cell.day}日：无提交`
      : `${cell.year}年${cell.month}月${cell.day}日：${cell.count} 次提交`;
  tip.value = {
    text,
    x: Math.min(event.clientX + 12, window.innerWidth - 180),
    y: event.clientY - 36,
  };
};
const hideTip = () => {
  tip.value = null;
};

// 自适应尺寸：格子大小与间距按面板可用宽度精确分配，铺满整行
const graphEl = ref(null);
const avail = ref(800); // 可用宽度（不含左侧星期标签列），挂载前给桌面端典型值
const LABEL_SPACE = 26; // 左侧星期标签列宽度（22px 标签 + 4px 间距）

const updateAvail = () => {
  const el = graphEl.value;
  if (!el) return;
  avail.value = el.clientWidth - LABEL_SPACE;
};

let resizeObserver;
onMounted(() => {
  updateAvail();
  resizeObserver = new ResizeObserver(updateAvail);
  if (graphEl.value) resizeObserver.observe(graphEl.value);
});
onBeforeUnmount(() => resizeObserver?.disconnect());

const cardVars = computed(() => {
  const n = weeks.value.length || 53;
  // 格子固定 10px，不随文字或窗口大小变化
  const cell = 10;
  // 间距只用剩余空间微调（2~6px），多余宽度由容器居中吸收
  const raw = n > 1 ? (avail.value - n * cell) / (n - 1) : 2;
  const gap = Math.min(6, Math.max(2, Math.round(raw * 10) / 10));
  return { "--cg-cell": `${cell}px`, "--cg-gap": `${gap}px` };
});
</script>

<style lang="scss" scoped>
.cgraph-wrap {
  display: flex;
  align-items: stretch;
  gap: 16px;
  margin: 1rem 0;
}

/* 外卡片：大标题 + 内嵌面板 */
.cgraph-main {
  --cg-cell: 10px;
  --cg-gap: 6px;

  position: relative;
  flex: 1;
  min-width: 0;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  background: #ffffff;
  padding: 16px 16px 12px;
}

.cgraph-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.cgraph-title {
  font-size: 17px;
  font-weight: 500;
  color: #1f2328;
}

/* 年份下拉框 */
.cgraph-select {
  appearance: none;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  background: #f6f8fa
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path d='M1 1l4 4 4-4' fill='none' stroke='%23656d76' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>")
    no-repeat right 8px center;
  color: #1f2328;
  font-size: 12px;
  padding: 3px 26px 3px 10px;
  cursor: pointer;
  outline: none;
  transition: border-color 0.15s ease;

  &:hover {
    border-color: #afb8c1;
  }

  &:focus {
    border-color: #0969da;
  }

  option {
    background: #ffffff;
    color: #1f2328;
  }
}

/* 内嵌图表面板（GitHub 的双层边框结构） */
.cgraph-panel {
  border: 1px solid #d8dee4;
  border-radius: 6px;
  padding: 12px 12px 8px;
}

.cgraph-scroll {
  overflow-x: auto;
  scrollbar-width: none;

  &::-webkit-scrollbar {
    display: none;
  }
}

.cgraph-stack {
  width: fit-content;
  margin: 0 auto;
}

.cgraph-months {
  display: grid;
  grid-template-rows: 16px;
  grid-auto-columns: var(--cg-cell);
  gap: var(--cg-gap);
  margin-left: 26px;
  margin-bottom: 6px;
  font-size: 11px;
  color: #656d76;
  white-space: nowrap;
}

.cgraph-body {
  display: flex;
  gap: 4px;
}

.cgraph-days {
  display: grid;
  grid-template-rows: repeat(7, var(--cg-cell));
  gap: var(--cg-gap);
  font-size: 10px;
  color: #656d76;
  min-width: 22px;

  span {
    display: flex;
    align-items: center;
  }
}

.cgraph-grid {
  display: grid;
  grid-template-rows: repeat(7, var(--cg-cell));
  grid-auto-flow: column;
  grid-auto-columns: var(--cg-cell);
  gap: var(--cg-gap);
}

.cgraph-cell {
  width: var(--cg-cell);
  height: var(--cg-cell);
  border-radius: 2px;
  background-color: #ebedf0;
  outline: 1px solid rgba(27, 31, 35, 0.04);
  outline-offset: -1px;
  transition: transform 0.1s ease;

  &:hover:not(.blank) {
    transform: scale(1.25);
  }

  &.lv1 { background-color: #9be9a8; }
  &.lv2 { background-color: #40c463; }
  &.lv3 { background-color: #30a14e; }
  &.lv4 { background-color: #216e39; }

  &.blank {
    visibility: hidden;
  }
}

.cgraph-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.cgraph-note {
  font-size: 11px;
  color: #656d76;
}

.cgraph-legend {
  display: flex;
  align-items: center;
  gap: 3px;
  font-size: 11px;
  color: #656d76;

  span {
    margin: 0 4px;
  }

  i.cgraph-cell {
    display: inline-block;

    &:hover {
      transform: none;
    }
  }
}

.cgraph-tip {
  position: fixed;
  z-index: 1000;
  pointer-events: none;
  background: #24292f;
  color: #fff;
  font-size: 12px;
  line-height: 1.6;
  padding: 4px 10px;
  border-radius: 6px;
  white-space: nowrap;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

/* 深色模式（GitHub 官方配色），跟随主题的 data-theme 属性切换 */
[data-theme="dark"] .cgraph-main {
  border-color: #30363d;
  background: #0d1117;
}

[data-theme="dark"] .cgraph-panel {
  border-color: #21262d;
}

[data-theme="dark"] .cgraph-title {
  color: #e6edf3;
}

[data-theme="dark"] .cgraph-select {
  border-color: #30363d;
  background-color: #21262d;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path d='M1 1l4 4 4-4' fill='none' stroke='%238b949e' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
  color: #e6edf3;

  &:hover {
    border-color: #484f58;
  }

  &:focus {
    border-color: #1f6feb;
  }

  option {
    background: #161b22;
    color: #e6edf3;
  }
}

[data-theme="dark"] .cgraph-months,
[data-theme="dark"] .cgraph-days,
[data-theme="dark"] .cgraph-note,
[data-theme="dark"] .cgraph-legend {
  color: #8b949e;
}

[data-theme="dark"] .cgraph-cell {
  background-color: #161b22;
  outline-color: rgba(255, 255, 255, 0.02);

  &.lv1 { background-color: #0e4429; }
  &.lv2 { background-color: #006d32; }
  &.lv3 { background-color: #26a641; }
  &.lv4 { background-color: #39d353; }
}

[data-theme="dark"] .cgraph-tip {
  background: #6e7681;
  color: #fff;
}
</style>
