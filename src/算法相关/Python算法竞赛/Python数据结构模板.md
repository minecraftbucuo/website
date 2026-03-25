# Python 高级数据结构模版（2026.3.17）

## 1. Python中实现丐版multiset
### 1.1 初始版(已有题目测试通过)
```Python
import heapq
from collections import Counter

class MultiSet:
    """模拟multiset，可快速获取最小值、最大值，删除任意值。"""
    def __init__(self):
        self.min_heap = []  # 最小堆
        self.max_heap = []  # 最大堆（存负数）
        self.min_pending = Counter()  # 最小堆中待删除的元素
        self.max_pending = Counter()  # 最大堆中待删除的元素（存负数的计数）
        self.mp = Counter()  # 元素出现的次数
        self._size = 0

    def insert(self, x):
        heapq.heappush(self.min_heap, x)
        heapq.heappush(self.max_heap, -x)
        self.mp[x] += 1
        self._size += 1

    def erase(self, x):
        if self._size <= 0:
            return
        if self.mp[x] == 0:
            return
        self.min_pending[x] += 1
        self.max_pending[-x] += 1
        self.mp[x] -= 1
        self._size -= 1
        self._clean()

    def get_min(self):
        self._clean()
        return self.min_heap[0] if self.min_heap else None

    def get_max(self):
        self._clean()
        return -self.max_heap[0] if self.max_heap else None

    def _clean(self):
        # 清理两个堆顶的无效元素
        while self.min_heap and self.min_pending[self.min_heap[0]] > 0:
            self.min_pending[self.min_heap[0]] -= 1
            heapq.heappop(self.min_heap)
        while self.max_heap and self.max_pending[self.max_heap[0]] > 0:
            self.max_pending[self.max_heap[0]] -= 1
            heapq.heappop(self.max_heap)

    def __len__(self):
        return self._size
```
### 1.2 叫 AI 优化的版本(没有用题目测试过)
```Python
import heapq
from collections import Counter

class MultiSet:
    """高性能 multiset：支持重复元素、O(1) amortized 极值查询、严格内存管理"""
    __slots__ = ('min_heap', 'max_heap', 'min_pending', 'max_pending', 'count', '_size')
    
    def __init__(self):
        self.min_heap = []      # 小顶堆：原始值
        self.max_heap = []      # 大顶堆：存储负值
        self.min_pending = Counter()  # min_heap 待删计数（原始值）
        self.max_pending = Counter()  # max_heap 待删计数（负值）
        self.count = Counter()        # 有效元素计数
        self._size = 0                # 有效元素总数（O(1) __len__）

    def insert(self, x):
        heapq.heappush(self.min_heap, x)
        heapq.heappush(self.max_heap, -x)
        self.count[x] += 1
        self._size += 1

    def erase(self, x):
        if self.count[x] <= 0:  # Counter 自动处理缺失键（返回0）
            return
        self.min_pending[x] += 1
        self.max_pending[-x] += 1
        self.count[x] -= 1
        self._size -= 1
        # 延迟清理：仅在查询极值时处理堆顶

    def get_min(self):
        self._clean_min_heap()
        return self.min_heap[0] if self.min_heap else None

    def get_max(self):
        self._clean_max_heap()
        return -self.max_heap[0] if self.max_heap else None

    def _clean_min_heap(self):
        """清理 min_heap 堆顶无效元素，并移除 pending 中的 0 计数键"""
        while self.min_heap:
            top = self.min_heap[0]
            if self.min_pending.get(top, 0) > 0:
                heapq.heappop(self.min_heap)
                self.min_pending[top] -= 1
                if self.min_pending[top] == 0:
                    del self.min_pending[top]  # 防止内存泄漏
            else:
                break

    def _clean_max_heap(self):
        """清理 max_heap 堆顶无效元素，并移除 pending 中的 0 计数键"""
        while self.max_heap:
            top = self.max_heap[0]  # top 为负值
            if self.max_pending.get(top, 0) > 0:
                heapq.heappop(self.max_heap)
                self.max_pending[top] -= 1
                if self.max_pending[top] == 0:
                    del self.max_pending[top]  # 防止内存泄漏
            else:
                break

    def __len__(self):
        return self._size

    def __bool__(self):
        return self._size > 0

    # 可选：提供只读 size 属性（符合 Python 习惯）
    @property
    def size(self):
        return self._size
```

## 2. 丐版线段树(已有题测试通过)
```Python
class Node:
    def __init__(self, l, r, mx):
        self.l = l
        self.r = r
        self.mx = mx

    def set(self, l, r):
        self.l = l
        self.r = r

    def set_max(self, mx):
        self.mx = mx

class SegTree:
    def __init__(self, vec):
        self.tr = [Node(0, 0, 0) for _ in range(len(vec) << 2)]
        self.build(vec, 1, 1, len(vec) - 1)

    def pushup(self, u):
        self.tr[u].set_max(max(self.tr[u << 1].mx, self.tr[u << 1 | 1].mx))

    def build(self, vec, u, l, r):
        self.tr[u].set(l, r)
        if l == r:
            self.tr[u].set_max(vec[l])
            return
        mid = (l + r) // 2
        self.build(vec, u << 1, l, mid)
        self.build(vec, u << 1 | 1, mid + 1, r)
        self.pushup(u)

    def query(self, l, r):
        return self.query_(1, l, r)

    def query_(self, u, l, r):
        if l <= self.tr[u].l and self.tr[u].r <= r:
            return self.tr[u].mx
        mid = (self.tr[u].l + self.tr[u].r) // 2
        if r <= mid:
            return self.query_(u << 1, l, r)
        if l > mid:
            return self.query_(u << 1 | 1, l, r)
        return max(self.query_(u << 1, l, mid), self.query_(u << 1 | 1, mid + 1, r))
```

## 3. 树状数组(已有题测试通过)
```Python
class BIT:
    __slots__ = ("n", "tr")
    def __init__(self, nums : list):
        """nums需要以1为起始索引"""
        self.n = len(nums)
        self.tr = [0] * self.n
        for i in range(1, self.n):
            self.add(i, nums[i])

    def add(self, idx : int, x_ : int) -> None:
        while idx < self.n:
            self.tr[idx] += x_
            idx += idx & -idx

    def query(self, l : int, r : int) -> int:
        return self._sum(r) - self._sum(l - 1)

    def _sum(self, idx : int) -> int:
        res_ = 0
        while idx != 0:
            res_ += self.tr[idx]
            idx -= idx & -idx
        return res_
```

