# Local HNSW Approximate Nearest-Neighbor Index

一个纯 Python（仅标准库）实现的本地向量近似最近邻搜索索引。所有向量、图结构、
索引元数据和快照只存在于**内存或本地文件**中——不依赖任何向量数据库、
Embedding API、云服务或网络访问。测试数据全部由代码按固定随机种子生成。

## 目录结构

```
ann_index/
  __init__.py        # 包导出
  distances.py       # cosine / L2 距离函数
  hnsw.py            # HNSW 图式 ANN 索引（插入/删除/查询/compact/快照）
  brute_force.py     # 暴力精确搜索，作为 recall 校验基准
tests/
  test_hnsw.py       # 30 个自动化测试，直接终端运行并打印 recall
```

## 索引结构

采用 **HNSW（Hierarchical Navigable Small World）** 多层图结构：

- 每个节点（向量）按指数衰减分布被随机指派一个层数 `level`（由索引内部
  带种子的 RNG 决定，相同 seed + 相同插入顺序 ⇒ 完全相同的图与查询结果）。
- 第 `l` 层中每个节点最多保留 `M` 条边（第 0 层为 `2*M`），边双向连接，
  插入时按"距离最近优先"剪枝。
- 查询从最高层入口点贪心下降，到第 0 层用宽度为 `ef_search` 的束搜索
  （beam search）找出候选集，过滤已删除节点后取 top-k。
- 删除采用**墓碑（tombstone）**标记：被删节点不再出现在结果中，但边保留
  以维持图的连通性；`compact()` 用存活向量确定性重建索引，物理清除无效
  节点（保留原 id）。

## 距离定义与排序规则

| metric    | 定义                                                        |
|-----------|-------------------------------------------------------------|
| `cosine`  | 余弦距离 `1 - cos(a, b)`，取值 `[0, 2]`；零范数向量约定距离为 1.0 |
| `l2`      | 欧氏距离 `sqrt(Σ (aᵢ - bᵢ)²)`                                |

所有查询结果按 **`(distance, id)` 升序**返回：距离越小越靠前，距离相等时
按 id 升序稳定 tie-break。暴力基准使用同一规则，两者可直接比较。

## 参数含义

| 参数               | 含义                                                         |
|--------------------|--------------------------------------------------------------|
| `dim`              | 固定向量维度；插入/查询维度不符抛 `ValueError`                |
| `metric`           | `'cosine'` 或 `'l2'`                                          |
| `M`                | 每节点每层最大连接数（第 0 层为 `2*M`）；越大 recall 越高、内存越大 |
| `ef_construction`  | 建图时的束搜索宽度；越大图质量越高、构建越慢                   |
| `ef_search`        | 查询时的默认束搜索宽度（可逐次查询覆盖）；越大 recall 越高、越慢 |
| `seed`             | 内部 RNG 种子，保证构建与结果可复现                            |

输入校验：维度错误、NaN、±Inf、非数值分量、非序列输入、非法 `k` 均抛出
`TypeError` 或 `ValueError`。

## API 概览

```python
from ann_index import HNSWIndex, brute_force_search

idx = HNSWIndex(dim=128, metric="cosine", M=16,
                ef_construction=100, ef_search=50, seed=42)

vid = idx.insert(vector)          # -> 整数 id
idx.delete(vid)                   # 墓碑删除，结果中立即消失
res = idx.search(query, k=10)     # -> [(id, distance), ...] 按 (distance, id) 排序
exact = idx.brute_force_search(query, k=10)   # 精确基准
idx.compact()                     # 清理墓碑、确定性重建（保留 id）
idx.save("snapshot.json")         # 本地快照（向量+图+元数据）
idx2 = HNSWIndex.load("snapshot.json")        # 加载后查询结果与保存前完全一致
```

## 复杂度

设 `N` 为向量数、`d` 为维度：

| 操作            | 复杂度（期望）                                  |
|-----------------|-------------------------------------------------|
| 插入            | `O(log N · M · ef_construction · d)`            |
| ANN 查询        | `O(log N · M · ef_search · d)`，远小于全扫描     |
| 暴力精确查询    | `O(N · d)`                                       |
| 删除            | `O(1)`（打墓碑标记）                             |
| compact         | 一次全量重建，同插入总成本 `O(N log N · M · ef_construction · d)` |
| 空间            | `O(N · (d + M · 平均层数))`                      |

测试中用 `dist_computations` 计数器验证 ANN 查询不会退化为全扫描
（2000 向量下单次查询约 800 次距离计算，且断言 `< N/2`）。

## 运行测试

```bash
cd session_45/a
python -m unittest discover -s tests -v
```

测试覆盖：插入/删除、不同 k（含 k > N）、重复向量与 tie-break、快照保存/
加载结果一致性、维度/NaN/Inf/非法类型异常、cosine 与 L2 两种度量的
recall@k、固定随机种子的确定性、compact 清理、ANN 非全扫描验证。
recall 与距离计算计数会直接打印到终端，例如：

```
[recall] cosine recall@10 on 2000 vectors, 50 queries: 1.0000
[recall] l2 recall@10 on 1500 vectors, 30 queries: 1.0000
[compact] recall@10 after delete+compact: 1.0000
[scan] ANN query used 801 distance computations vs 2000 for a full scan
Ran 30 tests ... OK
```
