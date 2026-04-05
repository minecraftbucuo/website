# VP到的一个有趣题（2026.4.5）

## 前言：
今天清明节，我们愉快地 VP 了一场 CCPC 邀请赛，整个基地一下午只有我们三个人。好了这些废话就不多说了，比赛链接：[2024 National Invitational of CCPC (Zhengzhou)](https://codeforces.com/gym/105158)。 这里所谓的有趣题就是其中的 D 题，我猜排个序答案一定在相邻不远的两个点中取到，没想到真过了，而且其实答案一定能在排序完相邻的两个点中取到。不会证啊，于是问 Gemini 给出了一个巧妙的证明，记录于此。

## 题目描述：
### Problem D. 距离之比

对于 $\mathbb{R}^2$ 平面上的两个点 $P(x_P, y_P)$ 与 $Q(x_Q, y_Q)$，$PQ$ 之间的曼哈顿距离定义为

$$
||PQ||_1 = |x_P - x_Q| + |y_P - y_Q|
$$

而 $PQ$ 之间的欧几里得距离定义为

$$
||PQ||_2 = \sqrt{(x_P - x_Q)^2 + (y_P - y_Q)^2}
$$

现在给出平面上互不重合的 $n$ 个点 $P_1, P_2, \dots, P_n$，请求出

$$
\max_{1 \le i < j \le n} \frac{||P_i P_j||_1}{||P_i P_j||_2}
$$

### 输入格式

本题测试点包含多组数据。
第一行，一个正整数 $T$（$1 \le T \le 10^5$），表示数据组数。

对于每组数据：
第一行，一个正整数 $n$（$2 \le n \le 2 \times 10^5$），表示平面上的点数。
接下来 $n$ 行，每行两个整数 $x_i, y_i$（$-10^5 \le x_i \le 10^5$，$-10^5 \le y_i \le 10^5$），表示点 $P_i(x_i, y_i)$。
保证对于单个测试点有 $\sum n \le 2 \times 10^5$。

### 输出格式

对于每组数据，输出一行，一个实数，表示点对之间曼哈顿距离与欧几里得距离之比的最大值。当你的答案与标准答案的相对误差或绝对误差不超过 $10^{-9}$ 时将视为正确答案。

## AI 给出的解答
这是一道非常经典的计算几何与数学结合的题目。考虑到 $\sum n \le 2 \times 10^5$，我们不能使用 $O(n^2)$ 的暴力枚举，而是需要寻找一种 $O(n \log n)$ 的解法，确保**不会出现超时 (TLE)** 的情况。

以下是具体的推导思路和 AC 代码：

### 💡 思路分析

**1. 公式变形与坐标系转换**
对于任意两点 $i$ 和 $j$，设 $\Delta x = x_i - x_j$，$\Delta y = y_i - y_j$。
题目要求最大化的比值为：
$$R = \frac{|\Delta x| + |\Delta y|}{\sqrt{\Delta x^2 + \Delta y^2}}$$

利用曼哈顿距离和切比雪夫距离的转换关系，我们引入新的坐标：
设 $u = x + y$，$v = x - y$。
那么有：
$$|\Delta x| + |\Delta y| = \max(|\Delta u|, |\Delta v|)$$
同时可以推导出：
$$\Delta x^2 + \Delta y^2 = \frac{\Delta u^2 + \Delta v^2}{2}$$

将这两项代入原比值公式中：
$$R = \frac{\max(|\Delta u|, |\Delta v|)}{\sqrt{\frac{\Delta u^2 + \Delta v^2}{2}}} = \sqrt{2} \frac{\max(|\Delta u|, |\Delta v|)}{\sqrt{\Delta u^2 + \Delta v^2}}$$

**2. 极值推导**
为了最大化上述 $R$，我们将分子分母同除以 $\min(|\Delta u|, |\Delta v|)$。
令 $t = \frac{\max(|\Delta u|, |\Delta v|)}{\min(|\Delta u|, |\Delta v|)}$ （显然 $t \ge 1$），则公式可化为：
$$R = \sqrt{2} \frac{t}{\sqrt{t^2 + 1}}$$

通过求导或观察可知，$f(t) = \frac{t}{\sqrt{t^2 + 1}}$ 在 $t \ge 1$ 时是一个**单调递增**函数。
因此，**最大化比值 $R$，就等价于最大化 $t$**，即最大化：
$$\max\left( \frac{|\Delta u|}{|\Delta v|}, \frac{|\Delta v|}{|\Delta u|} \right)$$

**3. 利用“糖水不等式 / 中位分数定理”降维打击**
现在的目标是找到一对点，使得 $\frac{|\Delta u|}{|\Delta v|}$ 或 $\frac{|\Delta v|}{|\Delta u|}$ 的值尽可能大。
根据中位分数定理（即 $\frac{\sum A_i}{\sum B_i} \le \max(\frac{A_i}{B_i})$），我们有一个极其优美的结论：
* 如果我们把所有点**按照 $v$ 坐标升序排序**，那么 $\frac{|\Delta u|}{|\Delta v|}$ 的全局最大值，**必定在相邻的两点之间产生**。
* 同理，如果我们把所有点**按照 $u$ 坐标升序排序**，那么 $\frac{|\Delta v|}{|\Delta u|}$ 的全局最大值，**也必定在相邻的两点之间产生**。

**4. 算法结论**
1. 对于每个点计算 $u_i = x_i + y_i$ 和 $v_i = x_i - y_i$。
2. 将所有点按照 $u$ 坐标排序，计算所有相邻点对的原始距离比值，取最大值。
3. 将所有点按照 $v$ 坐标排序，再次计算所有相邻点对的原始距离比值，取最大值。
4. 两种排序遍历后的最大值，即为最终答案。时间复杂度为非常优秀的 $O(n \log n)$，可以轻松通过样例及大数据范围。

---

### 💻 C++ 代码实现

代码中采用了 `long double` 来保证极高的精度，避免计算 $dx^2+dy^2$ 时产生浮点误差导致答案错误。

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

struct Point {
    long long x, y;
    long long u, v;
};

// 计算真实比值的函数，使用 long double 防治精度丢失
long double get_ratio(const Point& p1, const Point& p2) {
    long double dx = abs(p1.x - p2.x);
    long double dy = abs(p1.y - p2.y);
    return (dx + dy) / sqrt(dx * dx + dy * dy);
}

void solve() {
    int n;
    cin >> n;
    vector<Point> pts(n);
    for (int i = 0; i < n; i++) {
        cin >> pts[i].x >> pts[i].y;
        pts[i].u = pts[i].x + pts[i].y;
        pts[i].v = pts[i].x - pts[i].y;
    }

    long double max_r = 0.0;

    // 1. 按照 u 坐标排序并遍历相邻点
    sort(pts.begin(), pts.end(), [](const Point& a, const Point& b) {
        if (a.u != b.u) return a.u < b.u;
        return a.v < b.v; // 当 u 相同时，以 v 为第二关键字以保证严格稳定性
    });
    for (int i = 0; i < n - 1; i++) {
        max_r = max(max_r, get_ratio(pts[i], pts[i+1]));
    }

    // 2. 按照 v 坐标排序并遍历相邻点
    sort(pts.begin(), pts.end(), [](const Point& a, const Point& b) {
        if (a.v != b.v) return a.v < b.v;
        return a.u < b.u;
    });
    for (int i = 0; i < n - 1; i++) {
        max_r = max(max_r, get_ratio(pts[i], pts[i+1]));
    }

    // 按要求输出高精度实数
    cout << fixed << setprecision(12) << (double)max_r << "\n";
}

int main() {
    // 开启快读以应对极大数据量，防止 TLE
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}
```

## 为什么全局最大值一定在排序后相邻两点之间产生？

其实，这个结论是**排序带来的单调性**、**绝对值不等式（三角不等式）**以及**中位分数定理**三者巧妙结合的结果。我们来详细拆解一下这个推导过程。

### 1. 前置知识：中位分数定理（糖水不等式扩展）

假设有一组分数 $\frac{A_1}{B_1}, \frac{A_2}{B_2}, \dots, \frac{A_m}{B_m}$，且分母**全部为正数**（即 $B_k > 0$），分子非负（$A_k \ge 0$）。

那么，把它们的分子分母分别加起来得到的新分数，一定**不会大于**这些分数中最大的那一个。即：
$$\frac{A_1 + A_2 + \dots + A_m}{B_1 + B_2 + \dots + B_m} \le \max_{1 \le k \le m} \left( \frac{A_k}{B_k} \right)$$
*（直观理解：几杯不同甜度的糖水混在一起，混合后的甜度绝不可能超过原本最甜的那一杯。）*

### 2. 开始推导：为什么全局最大值只在相邻点产生？

我们以求 $\frac{|\Delta u|}{|\Delta v|}$ 的最大值为例。

**第一步：排序带来的性质（分母拆解）**
我们把所有点按照 $v$ 坐标升序排序，使得：
$$v_1 \le v_2 \le v_3 \le \dots \le v_n$$

现在，我们任取两个**不相邻**的点 $P_i$ 和 $P_j$（假设 $i < j$）。
它们在 $v$ 坐标上的差值可以拆解为相邻点差值之和：
$$v_j - v_i = (v_{i+1} - v_i) + (v_{i+2} - v_{i+1}) + \dots + (v_j - v_{j-1})$$
因为我们已经排过序了，所以上面括号里的**每一项 $(v_{k+1} - v_k)$ 都是大于等于 0 的**。这就是我们要的 $B_k$！

**第二步：三角不等式（分子拆解）**
再来看这两点在 $u$ 坐标上的差值的绝对值 $|u_j - u_i|$。
同样地，我们把它插入中间点：
$$u_j - u_i = (u_{i+1} - u_i) + (u_{i+2} - u_{i+1}) + \dots + (u_j - u_{j-1})$$
根据绝对值的**三角不等式**（即 $|x+y| \le |x| + |y|$），我们可以得到：
$$|u_j - u_i| \le |u_{i+1} - u_i| + |u_{i+2} - u_{i+1}| + \dots + |u_j - u_{j-1}|$$
这就构成了我们要的 $A_k$。

**第三步：见证奇迹的合并**
现在，我们把任意两点 $P_i$ 和 $P_j$ 之间的比值写出来，并应用上面的两个结论：

$$\frac{|\Delta u|}{|\Delta v|} = \frac{|u_j - u_i|}{|v_j - v_i|} = \frac{|u_j - u_i|}{v_j - v_i}$$
*(因为排序过了，分母 $v_j - v_i \ge 0$，可以去掉绝对值)*

套用第二步的三角不等式放大分子：
$$\le \frac{|u_{i+1} - u_i| + |u_{i+2} - u_{i+1}| + \dots + |u_j - u_{j-1}|}{(v_{i+1} - v_i) + (v_{i+2} - v_{i+1}) + \dots + (v_j - v_{j-1})}$$

套用第一步的中位分数定理：
$$\le \max_{i \le k < j} \left( \frac{|u_{k+1} - u_k|}{v_{k+1} - v_k} \right)$$

### 3. 总结

上面的公式证明了极其重要的一点：
**任意两个距离较远的点 $P_i$ 和 $P_j$ 所构成的比值，一定小于或等于它们之间某一对“相邻点”构成的比值。**

既然任何跨越多个点的组合，都无法超越中间某对相邻点的表现，那么我们在寻找**全局最大值**时，就完全不需要去枚举那些不相邻的点对（$O(n^2)$），只需要把点排序后，遍历比较所有的相邻点对（$O(n)$）就足够了。

同理，对于 $\frac{|\Delta v|}{|\Delta u|}$，我们只需要按照 $u$ 坐标排序，再用完全相同的逻辑推导一遍即可。这样整体时间复杂度就被完美压缩到了排序所需的 $O(n \log n)$。

## 感想
还是要大胆猜想啊。

<p align="right">最后更新于：2026-03-25</p>