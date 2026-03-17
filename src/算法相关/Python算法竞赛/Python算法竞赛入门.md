# Python 算法竞赛入门与常用算法模板

---

## 1. 常用库

```python
import sys
import math
import heapq
import bisect
from collections import deque, defaultdict, Counter
from functools import lru_cache
```

说明：

- `sys`：快读快写
- `math`：数学函数
- `heapq`：优先队列，默认小根堆
- `bisect`：二分查找
- `deque`：双端队列
- `defaultdict`：带默认值的字典
- `Counter`：计数器
- `lru_cache`：记忆化搜索

---

```python
# Python 里和 C++ 差别很大的几个点

# 1. 没有自增自减运算符
# ++i, i++, --i, i-- 都没有
# 统一写成
i += 1
i -= 1

# 2. 没有标准库有序 set / multiset / map
# Python 标准库里：
# dict / set 本质更像 unordered_map / unordered_set
# 没有 set<int, cmp>、multiset、map 这种树形有序容器
# 也没有 lower_bound / upper_bound 这种直接挂在容器上的接口

# 3. 没有 const
# 所谓常量只能靠命名习惯，比如
MOD = 998244353
INF = 10**18

# 4. 没有 C++ 那种静态数组 / 静态区写法
# 但有全局变量和局部变量
# 文件顶层定义的变量可以近似理解成“全局”
N = 200005
a = [0] * N

# 5. 递归深度默认很小，深搜时常要手动开大
import sys
sys.setrecursionlimit(1 << 20)

# 6. 常用容器主要是这些
# list         -> 动态数组 / 栈 / 邻接表
# tuple        -> pair / 多元组
# dict         -> 哈希映射，像 unordered_map
# set          -> 哈希集合，像 unordered_set
# defaultdict  -> 带默认值的 dict
# Counter      -> 计数哈希表
# deque        -> 双端队列 / 普通队列 
# heapq        -> 堆，默认小根堆

# 7. 这些容器大多不是有序容器
# 真正常用且“按顺序维护”的，标准库里主要靠：
# list + sort / bisect
# heapq
# 但都不是 set<int, cmp> 那种树结构

# 8. Python 代码按模块从上到下执行
# 没有必须存在的 main
# 但仍然有全局变量和函数局部变量的区别

# 9. 标准库基本没有“传比较器声明容器”的接口
# 例如不能写类似 set<int, cmp> s;
# 也不能像 priority_queue<T, vector<T>, cmp> 那样直接传 cmp
# 通常做法是：
# - 排序：sort(key=...)
# - 堆：手动改关键字 / 塞 tuple / 必要时写 __lt__

# 10. 没有指针，也没有 .begin() / .end()
# 很多遍历直接写：
for x in a:
    pass

# 11. 缩进就是代码块
# 不能像 C++ 靠大括号控制作用域
# 压行可以用分号，但一般不推荐乱压
a = 0; b = 1

# 12. 容器接口和 C++ 有些像，但名字常不一样
# list.append(x)      对应 push_back
# list.pop()          对应 pop_back
# heapq.heappush(q,x)
# heapq.heappop(q)
# deque.append(x)
# deque.popleft()
```

## 2. 输入输出

### 2.1 快读

```python
# 方式 1
import sys
input = sys.stdin.readline
# 方式 2 (自动调用strip())
import sys
input = lambda : sys.stdin.redline().strip()
```

### 2.2 基本输入

```python
n = int(input())
a, b = map(int, input().split())
s = input().strip()
a = list(map(int, input().split()))
```

说明：

- `input()` 默认读到的是字符串
- 字符串通常配合 `strip()` 去掉末尾换行

### 2.3 多测模板

```python
def solve():
    pass

T = int(input())
for _ in range(T):
    solve()
```

### 2.4 输出

```python
print(x)
print(a, b)
print(*res)
print(ans, end=' ')
print(' '.join(res))
```

---

## 3. 作用域与常见写法

Python 没有大括号，靠缩进控制作用域。

```python
def gcd(a, b):
    if b:
        return gcd(b, a % b)
    return a
```

单行赋值：

```python
a, b = 1, 2
x = y = 0
```

数组初始化：

```python
inf = 10 ** 18
dis = [inf] * (n + 1)
dis[s] = 0
```

二维数组初始化：

```python
g = [[] for _ in range(n + 1)]
```

不要写成：

```python
g = [[]] * (n + 1)
```

因为python没有静态数组所以写定大小只能写成下面这样

```python
# int f[N][M];
f = [[0] * M for _ in range(N)]
```

也不要写成这样

```python
f = [[0] * M for _ in range(N)]
```

后果是这样的。

```python
f = [[0] * 3] * 3
f[0][1] = 7
print(f)
# [[0, 7, 0], [0, 7, 0], [0, 7, 0]]
```

```python
# 初始化方法
inf = 10**18
dis = [[inf] * (m + 1) for _ in range(n + 1)]
```

```python
# 三维 [n][m][k]
f = [[[0] * k for _ in range(m)] for _ in range(n)]
```



---

## 4. list

Python 的 `list` 可以理解成动态数组。

### 4.1 初始化

```python
a = []
# 这里对应的就是静态的形式
a = [0] * n
# vector<vector>g(n + 1);
g = [[] for _ in range(n + 1)]
```

### 4.2 常用操作

```python
# 类似a.push_back(x)
a.append(x)
# a.pop_back()
a.pop()
# a.back()类似的写法
a[-1]
# a.size()
n = len(a)
# 在pos处插入val，这里
a.insert(pos, val)
a.reverse()
a.clear()
sum(a)
# 统计容器中的x的cnt
a.count(x)
n = max(a)
n = min(a)
```

判空：

```python
if not a:
    # 填充pass部分就行，当占位符了,除此之外这个关键字没有任何作用
    pass
```

### 4.3 排序

```python
a.sort()
a.sort(reverse=True)
b = sorted(a)
```

自定义排序：

```python
# 0,1都按升序排序
a.sort(key=lambda x: (x[0], x[1]))
# 0按升序，1按降序
a.sort(key=lambda x: (x[0], -x[1]))
# 优先1按降序排
a.sort(key=lambda x: (-x[1], x[0]))
```

### 4.4 输出 list

```python
print(a)
print(*a)
for x in a:
    print(x)
```

### 4.5 多个参数打包

`append` 一次只能加一个对象，所以多个量要打包成元组：

```python
g[u].append((v, w))
g[u].append((v, w, idx))
```

遍历：

```python
for v, w in g[u]:
    pass

for v, w, idx in g[u]:
    # 这里pass是用来占位的，对应的是大括号内的内容
    pass
```

---

## 5. 字符串

```python
s = input().strip()
len(s)
s[i]
s[::-1]   # 反转序列
s.split()
```

转字符数组：

```python
s = list(input().strip())
```

拼接字符串：

```python
res = []
res.append('a')
res.append('b')
print(' '.join(res))
```

---

## 6. deque

需要队列 / 双端队列时用 `deque`。

```python
from collections import deque

q = deque()
q.append(x)
q.appendleft(x)
q.pop()
q.popleft()
```

BFS 常用：

```python
q = deque([s])
vis[s] = 1
while q:
    u = q.popleft()
    for v in g[u]:
        if not vis[v]:
            vis[v] = 1
            q.append(v)
```

---

## 7. heapq

注意：`heapq` 默认是小根堆。

### 7.1 基本操作

```python
import heapq

# 将现有列表转化为堆 (原地操作，O(N))
nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)
print(f"堆化后: {nums}") 
# 输出: [1, 1, 2, 3, 5, 9, 4, 6] (注意：列表内部顺序不一定是完全排序的，但 nums[0] 一定是最小值)

q = []
heapq.heappush(q, x)
x = heapq.heappop(q)
```

### 7.2 存二元组

```python
heapq.heappush(q, (dis[v], v))
d, u = heapq.heappop(q)
```

按元组字典序比较，先比第一个，再比第二个。

### 7.3 大根堆

通常把值取反：

```python
heapq.heappush(q, -x)
x = -heapq.heappop(q)
```

---

## 8. dict / set / defaultdict / Counter

### 8.1 dict

```python
# 本质哈希表unordered_map<int,int>，
# 先声明一下容器 mp = {}
# 当然也可以存字符 mp['a'] = 1
mp = {}
mp[x] = mp.get(x, 0) + 1
if x in mp:
    pass
for k, v in mp.items():
    pass
# dict的问题在于，如果你没定义mp[15]的值的话访问是报错的，也就是这里要提前定义才有对应的值存储不是默认0
```

```python
mp = {}
s = "abac"

for c in s:
    if c in mp:
        mp[c] += 1
    else:
        # 一定得声明
        mp[c] = 1
```

### 8.2 defaultdict

```python
from collections import defaultdict

mp = defaultdict(int)
mp[x] += 1

g = defaultdict(list)
g[u].append(v)
```

### 8.3 Counter

```python
from collections import Counter

cnt = Counter(a)
print(cnt[x])
```

### 8.4 set

```python
st = set()
# set.insert(x)
st.add(x)
# set.erase(x)
st.discard(x)
# 这里类似st.find(x) != st.end()
if x in st:
    pass
```

---

## 9. bisect 二分库

```python
import bisect
#lower_bound,upper_bound
l = bisect.bisect_left(a, x)
r = bisect.bisect_right(a, x)
```

要求 `a` 有序。

- `bisect_left(a, x)`：第一个 `>= x` 的位置
- `bisect_right(a, x)`：第一个 `> x` 的位置

---

## 10. math 库

```python
math.gcd(a, b)
math.lcm(a, b)
#sqrt是浮点，isqrt是整型
math.sqrt(x)
math.isqrt(x)
math.factorial(n)
math.comb(n, k)   # 从 n 个里面选 k 个
math.perm(n, k)   
abs(x)

#math库里comb对应的是组合数，perm对应排列数，factorial对应的是阶乘
```

计算几何常用：

```python
math.sin(x)
math.cos(x)
math.tan(x)
math.asin(x)
math.acos(x)
math.atan(x)
math.atan2(y, x)
```

常量：

```python
math.pi
math.e
# 没啥用写10**18更直观
math.inf
#NaN 表示非法的计算值比如1 / 0
math.nan
```

---

## 11. 常见循环

```python
# 0 到 n - 1遍历
for i in range(n):
    pass
# l 到 r
for i in range(l, r + 1):
    pass
# 起点，终点，步骤
# 对应for(int i = n - 1;i >= 0;--i){}
for i in range(n - 1, -1, -1):
    pass
```

---

## 12. 图论存图模板

### 12.1 无权图

```python
g = [[] for _ in range(n + 1)]
for _ in range(m):
    u, v = map(int, input().split())
    g[u].append(v)
    g[v].append(u)
```

### 12.2 带权图

```python
g = [[] for _ in range(n + 1)]
for _ in range(m):
    u, v, w = map(int, input().split())
    g[u].append((v, w))
    g[v].append((u, w))
```

### 12.3 链式前向星风格（Python 不常用）

Python 里通常直接用邻接表 list，不太会像 C++ 一样硬写数组链式前向星。

---

## 13. 基础算法模板

## 13.1 gcd / lcm

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a // gcd(a, b) * b
```

## 13.2 快速幂

```python
def qpow(a, b, mod):
    res = 1
    while b:
        if b & 1:
            res = res * a % mod
        a = a * a % mod
        b >>= 1
    return res
```

## 13.3 扩展欧几里得

```python
def exgcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = exgcd(b, a % b)
    x = y1
    y = x1 - a // b * y1
    return g, x, y
```

## 13.4 模逆元

质数模：

```python
def inv(x, mod):
    return qpow(x, mod - 2, mod)
```

一般情况：

```python
# 扩展欧几里得定理求逆元
def inv_exgcd(a, mod):
    g, x, y = exgcd(a, mod)
    if g != 1:
        return -1
    return x % mod
```

---

## 14. 前缀和 / 差分

### 14.1 一维前缀和

```python
pre = [0] * (n + 1)
for i in range(1, n + 1):
    pre[i] = pre[i - 1] + a[i]
```

区间和：

```python
s = pre[r] - pre[l - 1]
```

### 14.2 二维前缀和

```python
pre = [[0] * (m + 1) for _ in range(n + 1)]
for i in range(1, n + 1):
    for j in range(1, m + 1):
        pre[i][j] = pre[i - 1][j] + pre[i][j - 1] - pre[i - 1][j - 1] + a[i][j]
```

### 14.3 一维差分

```python
d = [0] * (n + 2)
d[l] += v
d[r + 1] -= v
```

还原：

```python
for i in range(1, n + 1):
    d[i] += d[i - 1]
```

---

## 15. 二分答案模板

```python

def check(x):
    # function contents
    return True

l, r = 0, 10 ** 18
while l < r:
    mid = (l + r) >> 1
    if check(mid):
        r = mid
    else:
        l = mid + 1
print(l)
```

找最后一个满足条件的位置：

```python
l, r = 0, 10 ** 18
while l < r:
    mid = (l + r + 1) >> 1
    if check(mid):
        l = mid
    else:
        r = mid - 1
print(l)
```

## 16. 排序去重 / 离散化

### 16.1 去重

```python
a = sorted(set(a))
```

### 16.2 离散化

```python
b = sorted(set(a))
mp = {x: i + 1 for i, x in enumerate(b)}
res = [mp[x] for x in a]
```

---

## 17. 双指针模板

### 17.1 同向双指针

```python
j = 0
for i in range(n):
    while j < n and check(i, j):
        j += 1
    # 当前处理 [i, j)
```

### 17.2 相向双指针

```python
l, r = 0, n - 1
while l < r:
    if cond:
        l += 1
    else:
        r -= 1
```

---

## 18. 滑动窗口模板

```python
def check(cnt):
    return #//判定的方法

cnt = defaultdict(int)
l = 0
for r in range(n):
    cnt[a[r]] += 1
    while not check(cnt):
        cnt[a[l]] -= 1
        l += 1
    # 当前窗口 [l, r]
```

---

## 19. 枚举子集模板

### 19.1 枚举所有子集

```python
for s in range(1 << n):
    pass
```

### 19.2 枚举 s 的子集

```python
t = s
while t:
    # 处理 t
    t = (t - 1) & s
# 如果还要处理空集，再单独补一次
```

---

## 20. DFS / BFS 模板

### 20.1 DFS

```python
def dfs(u, fa):
    for v in g[u]:
        if v == fa:
            continue
        dfs(v, u)
```

如果带边权：

```python
def dfs(u, fa):
    for v, w in g[u]:
        if v == fa:
            continue
        dfs(v, u)
```

### 20.2 BFS

```python
from collections import deque

def bfs(s):
    dis = [-1] * (n + 1)
    dis[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in g[u]:
            if dis[v] == -1:
                dis[v] = dis[u] + 1
                q.append(v)
    return dis
```

---

## 21. 最短路模板

### 21.1 Dijkstra

```python
import heapq

inf = 10 ** 18

def dijkstra(s, n, g):
    dis = [inf] * (n + 1)
    dis[s] = 0
    q = [(0, s)]
    while q:
        d, u = heapq.heappop(q)
        if d != dis[u]:
            continue
        for v, w in g[u]:
            if dis[v] > d + w:
                dis[v] = d + w
                heapq.heappush(q, (dis[v], v))
    return dis
```

### 21.2 01 BFS

边权只有 `0/1` 时：

```python
from collections import deque

inf = 10 ** 18

def bfs01(s, n, g):
    dis = [inf] * (n + 1)
    dis[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v, w in g[u]:
            if dis[v] > dis[u] + w:
                dis[v] = dis[u] + w
                if w == 0:
                    q.appendleft(v)
                else:
                    q.append(v)
    return dis
```

### 21.3 Floyd

```python
for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])
```

---

## 22. 并查集模板

```python
# 等效于 iota(fa.begin(),fa.end(),0);,把所有点从0开始初始化成0,1,2,3,...,n的形式
fa = list(range(n + 1))
siz = [1] * (n + 1)

def find(x):
    if fa[x] != x:
        fa[x] = find(fa[x])
    return fa[x]

def merge(x, y):
    fx = find(x)
    fy = find(y)
    if fx == fy:
        return False
    if siz[fx] < siz[fy]:
        fx, fy = fy, fx
    fa[fy] = fx
    siz[fx] += siz[fy]
    return True
```

---

## 23. 拓扑排序模板

```python
from collections import deque

ind = [0] * (n + 1)
q = deque()
for i in range(1, n + 1):
    if ind[i] == 0:
        q.append(i)

order = []
while q:
    u = q.popleft()
    order.append(u)
    for v in g[u]:
        ind[v] -= 1
        if ind[v] == 0:
            q.append(v)
```

---

## 24. 二分图染色模板

```python
col = [0] * (n + 1)

def dfs(u, c):
    col[u] = c
    for v in g[u]:
        if col[v] == 0:
            if not dfs(v, 3 - c):
                return False
        elif col[v] == c:
            return False
    return True
```

---

## 25. 树上基础模板

### 25.1 树大小 / 深度 / 父亲

```python
fa = [0] * (n + 1)
dep = [0] * (n + 1)
siz = [0] * (n + 1)

def dfs(u, p):
    fa[u] = p
    dep[u] = dep[p] + 1
    siz[u] = 1
    for v in g[u]:
        if v == p:
            continue
        dfs(v, u)
        siz[u] += siz[v]
```

### 25.2 LCA 倍增

```python
LOG = 20
fa = [[0] * (n + 1) for _ in range(LOG)]
dep = [0] * (n + 1)

def dfs(u, p):
    fa[0][u] = p
    dep[u] = dep[p] + 1
    for k in range(1, LOG):
        fa[k][u] = fa[k - 1][fa[k - 1][u]]
    for v in g[u]:
        if v == p:
            continue
        dfs(v, u)

def lca(x, y):
    if dep[x] < dep[y]:
        # 等价于 swap(x,y)
        x, y = y, x
    d = dep[x] - dep[y]
    # 严格二进制dep倍增
    for k in range(LOG):
        if d >> k & 1:
            x = fa[k][x]
    if x == y:
        return x
    for k in range(LOG - 1, -1, -1):
        if fa[k][x] != fa[k][y]:
            x = fa[k][x]
            y = fa[k][y]
    return fa[0][x]
```

---

## 26. DP 常用模板

### 26.1 线性 DP

```python
f = [0] * (n + 1)
f[0] = 1
for i in range(1, n + 1):
    f[i] = f[i - 1]
```

### 26.2 背包 01

```python
f = [0] * (m + 1)
for i in range(1, n + 1):
    for j in range(m, w[i] - 1, -1):
        f[j] = max(f[j], f[j - w[i]] + v[i])
```

### 26.3 完全背包

```python
f = [0] * (m + 1)
for i in range(1, n + 1):
    for j in range(w[i], m + 1):
        f[j] = max(f[j], f[j - w[i]] + v[i])
```

### 26.4 区间 DP

```python
f = [[0] * (n + 1) for _ in range(n + 1)]
for length in range(2, n + 1):
    for l in range(1, n - length + 2):
        r = l + length - 1
        for k in range(l, r):
            f[l][r] = max(f[l][r], f[l][k] + f[k + 1][r])
```

---

## 27. 单调栈模板

### 27.1 求每个点左边第一个更小值位置

```python
st = []
L = [0] * n
for i in range(n):
    while st and a[st[-1]] >= a[i]:
        st.pop()
    L[i] = st[-1] if st else -1
    st.append(i)
```

### 27.2 求每个点右边第一个更小值位置

```python
st = []
R = [n] * n
for i in range(n - 1, -1, -1):
    while st and a[st[-1]] >= a[i]:
        st.pop()
    R[i] = st[-1] if st else n
    st.append(i)
```

---

## 28. 单调队列模板

维护区间最小值：

```python
from collections import deque

q = deque()
for i in range(n):
    while q and a[q[-1]] >= a[i]:
        q.pop()
    q.append(i)
    while q and q[0] <= i - k:
        q.popleft()
    if i >= k - 1:
        print(a[q[0]])
```

---

## 29. KMP 模板

### 29.1 求 next 数组

```python
def kmp_nxt(s):
    n = len(s)
    nxt = [0] * n
    j = 0
    for i in range(1, n):
        while j and s[i] != s[j]:
            j = nxt[j - 1]
        if s[i] == s[j]:
            j += 1
        nxt[i] = j
    return nxt
```

### 29.2 模式匹配

```python
def kmp_match(s, t):
    nxt = kmp_nxt(t)
    j = 0
    pos = []
    for i in range(len(s)):
        while j and s[i] != t[j]:
            j = nxt[j - 1]
        if s[i] == t[j]:
            j += 1
        if j == len(t):
            pos.append(i - len(t) + 1)
            j = nxt[j - 1]
    return pos
```

---

## 30. Trie 模板

```python
ch = [[0] * 26]
cnt = [0]

def insert(s):
    u = 0
    for c in s:
        x = ord(c) - 97
        if ch[u][x] == 0:
            ch[u][x] = len(ch)
            ch.append([0] * 26)
            cnt.append(0)
        u = ch[u][x]
    cnt[u] += 1


def find(s):
    u = 0
    for c in s:
        x = ord(c) - 97
        if ch[u][x] == 0:
            return 0
        u = ch[u][x]
    return cnt[u]
```

---

## 31. 矩阵快速幂模板

```python
def mat_mul(a, b, mod):
    n = len(a)
    m = len(b[0])
    t = len(b)
    c = [[0] * m for _ in range(n)]
    for i in range(n):
        for k in range(t):
            if a[i][k] == 0:
                continue
            for j in range(m):
                c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % mod
    return c


def mat_pow(a, b, mod):
    n = len(a)
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        res[i][i] = 1
    while b:
        if b & 1:
            res = mat_mul(res, a, mod)
        a = mat_mul(a, a, mod)
        b >>= 1
    return res
```

---

## 32. 质数筛 / 分解质因数

### 32.1 埃氏筛

```python
def eratosthenes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return is_prime
```

### 32.2 线性筛

```python
def linear_sieve(n):
    vis = [0] * (n + 1)
    pri = []
    for i in range(2, n + 1):
        if not vis[i]:
            pri.append(i)
        for p in pri:
            if i * p > n:
                break
            vis[i * p] = 1
            if i % p == 0:
                break
    return pri
```

### 32.3 试除分解质因数

```python
def divide(x):
    res = []
    i = 2
    while i * i <= x:
        if x % i == 0:
            c = 0
            while x % i == 0:
                x //= i
                c += 1
            res.append((i, c))
        i += 1
    if x > 1:
        res.append((x, 1))
    return res
```

---

## 33. 组合数模板

### 33.1 杨辉三角

```python
C = [[0] * (n + 1) for _ in range(n + 1)]
for i in range(n + 1):
    C[i][0] = C[i][i] = 1
    for j in range(1, i):
        C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
```

### 33.2 阶乘逆元预处理

```python
mod = 998244353
fac = [1] * (n + 1)
ifac = [1] * (n + 1)
for i in range(1, n + 1):
    fac[i] = fac[i - 1] * i % mod
ifac[n] = qpow(fac[n], mod - 2, mod)
for i in range(n, 0, -1):
    ifac[i - 1] = ifac[i] * i % mod

def C(n, m):
    if m < 0 or m > n:
        return 0
    return fac[n] * ifac[m] % mod * ifac[n - m] % mod
```

---

## 34. 记忆化搜索模板

```python
from functools import lru_cache

@lru_cache(None)
def dfs(x):
    if x == 0:
        return 1
    return dfs(x - 1)
```

用完可以清缓存：

```python
dfs.cache_clear()
```

---

## 35. 常用基础模板总板

```python
import sys
import math
import heapq
import bisect
from collections import deque, defaultdict, Counter
from functools import lru_cache

input = sys.stdin.readline
inf = 10 ** 18


def solve():
    pass


T = 1
# T = int(input())
for _ in range(T):
    solve()
```

---

## 36. 需要特别注意的点

1. `heapq` 默认是小根堆，不是大根堆。
2. `input()` 默认返回字符串，不是整数。
3. 二维数组不要写成 `[[0] * m] * n` 这种会共享引用的形式。
4. `sum(res)` 只能用于元素可加时，通常要求都是数字。
5. `append` 一次只能加一个对象，多个量要打包成元组。
6. Python 递归深度默认较小，深搜很深时要小心，必要时加：

```python
# 一定放在最前的地方写
sys.setrecursionlimit(1 << 20)
```

7. 竞赛里频繁字符串拼接优先写成 list 后 `' '.join(res)`。
8. Python 常用 `0-index`，但图论和树题你也可以统一写成 `1-index`。



