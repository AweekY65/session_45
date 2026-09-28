# local_vector_ann — 本地向量近似最近邻搜索索引

一个**完全本地、零第三方依赖**（仅使用 Python 3.8+ 标准库）的向量 ANN 索引实现。
所有向量、图结构、索引元数据和快照都只存在于**进程内存或本地文件**中：

- 不依赖任何向量数据库（FAISS / Milvus / pgvector 等）；
- 不调用 Embedding API、云服务或任何网络服务；
- 测试向量全部由代码用确定性随机种子生成。

核心算法为 **HNSW**（Hierarchical Navigable Small World，Malkov & Yashunin, 2018），
并附带 **brute-force 精确搜索**作为校验基准。

## 目录结构

```
.
├── local_vector_ann/
│   ├── __init__.py      # 包入口，导出 HNSWIndex / BruteForceIndex / 距离函数
│   ├── hnsw.py          # HNSW 图索引：插入、删除、搜索、compact、二进制持久化
│   ├── bruteforce.py    # 暴力精确搜索（ground-truth oracle）
│   ├── distances.py     # 距离定义、向量校验（维度 / NaN / 非法输入）
│   └── rng.py           # 内置确定性 RNG（xorshift64* + splitmix64 播种）
├── tests/
│   └── test_ann.py      # 自动化测试（自带轻量断言框架，终端直接运行）
└── README.md
```

## 快速开始

```python
from local_vector_ann import HNSWIndex, BruteForceIndex

idx = HNSWIndex(dim=128, metric="cosine", M=16,
                ef_construction=100, ef_search=50, seed=42)

idx.insert(0, [0.12, -0.03, ...])          # 支持 int 或 str 标签
idx.insert("doc-1", [ ... ])
results = idx.search(query, k=10)          # -> [(label, distance), ...]
idx.delete("doc-1")                        # 逻辑删除，立即从结果中消失
idx.compact()                              # 物理重建，清除墓碑节点
idx.save("index.bin")                      # 保存为单个本地二进制快照
idx2 = HNSWIndex.load("index.bin")         # 重新加载，查询结果与保存前一致

bf = BruteForceIndex(128, metric="cosine") # 精确基准，API 与 HNSWIndex 一致
```

## 索引结构

HNSW 是一个**多层近邻图**：

- 每个向量是一个节点，插入时随机分配一个层级 `L`（几何分布，
  `P(L=l) = exp(-l/mL)·(1-exp(-1/mL))`，`mL = 1/ln(M)`）；
- 第 0 层（base layer）包含全部节点，向上每层节点数指数级减少；
- 每个节点在自己所处的每一层维护一个近邻边表（0 层最多 `2M` 条边，
  高层最多 `M` 条），近邻用 HNSW 论文的 **heuristic neighbor selection**
  选择（优先保留互相之间不被更近节点“解释”的多样化近邻）；
- 全局有一个 entry point（图中层级最高的节点）。

**查询**：从 entry point 出发，在最高层做贪心下降（每层只走向更近的邻居），
到达第 0 层后用宽度为 `ef_search` 的 beam search（两个堆：待扩展候选的最小堆、
当前最优 `ef` 个结果的最大堆）收集近邻。

**删除模型**：删除只把节点标记为 **tombstone（墓碑）**。墓碑立即从搜索结果中
过滤掉，但仍保留在图中充当路由节点，因此删除是 O(1) 且不会破坏图连通性。
`compact()` 用原始种子重置 RNG，按原始插入顺序重建一张只含存活节点的新图——
其结构与“一开始就只插入存活向量”得到的图逐位一致。

## 距离定义

所有指标统一为**“越小越近”**：

| metric | 定义 | 取值范围 |
|---|---|---|
| `l2` | 平方欧氏距离 `Σ(aᵢ-bᵢ)²` | `[0, +∞)` |
| `cosine` | `1 − cosine_similarity`，即 `1 − a·b/(‖a‖·‖b‖)` | `[0, 2]` |

cosine 模式下，向量入库时被归一化为单位向量，之后距离等价于 `1 − 点积`。

**稳定 tie-break 规则**：结果按 `(距离量化桶, 标签)` 排序——距离差小于
`TIE_EPS = 1e-12` 视为相等（消除浮点末位抖动），距离相等时按标签确定序排序
（int 先于 str；int 按数值、str 按字典序）。brute-force 使用完全相同的排序规则。
因此重复向量、相等距离的场景结果也是确定、可复现的。

## 参数含义

| 参数 | 含义 | 增大的影响 |
|---|---|---|
| `dim` | 固定向量维度，插入/查询维度不符会抛 `ValueError` | — |
| `metric` | `"cosine"`（默认）或 `"l2"` | — |
| `M` | 构建连接数：每个节点每层的目标近邻数（0 层上限 `2M`） | 图更密、recall 更高、内存与构建/查询成本更高 |
| `ef_construction` | 构建时每层 beam search 的宽度 | 图质量更好、构建更慢 |
| `ef_search` | 查询搜索宽度（也可在 `search()` 中按次覆盖，自动取 `max(ef, k)`） | recall 更高、查询更慢；增大不应降低整体 recall |
| `seed` | 内置 RNG 的整数种子，决定层级分配 | 相同数据+相同种子 ⇒ 完全相同的图与结果 |

## 复杂度（n 个节点）

| 操作 | 期望复杂度 |
|---|---|
| 插入 | O(M·ef_construction·log n) 次距离计算 |
| 查询 | O(ef_search·log n) 次距离计算（亚线性，远小于 n） |
| 删除 | O(log n) 量级（标记墓碑 + 标签表更新） |
| compact | O(n·M·ef_construction·log n)，即全量重建 |
| 内存 | O(n·dim) 向量 + O(n·M·log n) 条边 |
| 快照 | O(n·(dim + M·log n)) 字节，单文件，原子写入（tmp + `os.replace`） |

测试会实际统计每次查询的距离计算次数（`idx.distance_count`），以证明 ANN
没有退化成全量扫描。

## 输入校验

插入和查询都会校验：

- 维度必须等于 `dim`（否则 `ValueError`）；
- 分量必须是 `int`/`float`，不能是 `NaN`/`±inf`（`ValueError`）、字符串等（`TypeError`）；
- 必须是有长度、可索引的序列（生成器等一次性迭代器会被拒绝）；
- cosine 模式禁止零向量；标签必须是 `int`（64 位有符号范围）或 `str`，
  且活节点标签唯一（重复插入抛 `ValueError`，需先 delete）。

## 持久化格式

`save(path)` 写出一个自包含二进制文件（小端）：

```
magic "LVANNH01" | version | metric | dim | M | ef_construction | ef_search
| mL(float64) | seed(uint64) | rng_state(uint64) | next_id | entry_point | node_count
| 每个节点: id, deleted 标记, level, 标签(int64 或 UTF-8 变长字符串),
           dim 个 float64 向量, 逐层的近邻 id 列表
```

向量按 **float64** 存储以保证序列化前后查询结果位级一致；快照同时保存 RNG
状态，因此 save → load → 继续插入的图与从不中断的进程完全一致。墓碑也会
被序列化，加载后删除状态保持不变。

## 运行测试

无需安装任何依赖，直接在终端运行：

```bash
python3 tests/test_ann.py
# 或
python3 -m tests.test_ann
```

输出内容包括：

- 每个检查项的 `[PASS]/[FAIL]`；
- 两种 metric、多种 k 下的 **recall@k** 统计（与 brute-force 对比）；
- 每次查询的平均距离计算次数及其占全量扫描的比例；
- 亚线性扩展测试（n 从 400 → 4000 时距离计算量的增长倍数）；
- 汇总行与失败列表，全部通过时退出码为 0，否则为 1。

覆盖场景：

1. 维度错误、NaN/inf、非法分量、生成器、错误标签、重复标签、空索引；
2. 小数据集上 HNSW 与 brute-force 在 k=1/3/10/25 下完全一致（cosine + l2）；
3. 重复向量与稳定 tie-break（按标签排序、重复查询结果不变）；
4. 删除后结果中不再出现、墓碑计数、compact 后结果不变且等价于全新构建；
5. save/load 后结果与距离完全一致、墓碑保持、RNG 状态延续、坏文件报错；
6. recall@1/@10/@50（默认配置下约 0.99–1.00）、ef_search 单调性、
   查询不做全量扫描；
7. 亚线性距离预算扩展（10× 数据只需约 1.5× 距离计算）；
8. 确定性种子：同种子图结构/层级/结果逐位一致，不同种子仍高质量；
9. 结果排序契约：距离非降序、tie 桶内按标签、距离非负、标签不重复。

典型输出（Python 3.12，纯标准库实现）：

```
metric=cosine n=2000 dim=32 ef_search=100  recall@10 = 1.0000
metric=l2     n=2000 dim=32 ef_search=100  recall@10 = 1.0000
=== sublinear distance-budget scaling ===
  n=  400  evals/query=    257  fraction of full scan=64.24%
  n= 1600  evals/query=    354  fraction of full scan=22.10%
  n= 4000  evals/query=    398  fraction of full scan= 9.95%
TOTAL: 63 passed, 0 failed, 63 checks
ALL TESTS PASSED
```

## 设计说明与限制

- 纯 Python 标准库实现，定位是可读、可测试、零依赖的参考实现；在数万向量、
  数百维以内的规模上使用没有问题，追求极致性能时可把距离热循环替换为
  numpy/原生扩展，算法层无需改动。
- 删除大量向量后建议适时 `compact()`：墓碑不影响正确性，但会占用内存并
  略微增加路由开销。
- HNSW 是近似索引；把 `ef_search` 调大（或取 `k=len(index)`）可以逼近精确
  结果，brute-force 索引始终提供精确基准用于离线评估 recall。
