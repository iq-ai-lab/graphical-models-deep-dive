# 02. Treewidth와 Inference Complexity

## 🎯 핵심 질문

- **Treewidth**는 graph의 어떤 성질을 측정하는가?
- 왜 inference 복잡도 $O(n \cdot d^{\text{tw} + 1})$이 정확히 treewidth에 의해 결정되는가?
- **Min treewidth** 계산이 왜 NP-hard인가 (Arnborg-Corneil-Proskurowski 1987)?
- Min-fill, min-weight 등의 heuristic은 얼마나 효과적인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Treewidth**는 PGM inference의 **근본적 복잡도 측도**. Low treewidth = tractable inference, high treewidth = intractable. **Grid MRF**(image segmentation)의 treewidth $O(\sqrt N)$이 exact inference를 불가능하게 하고, **chain/tree**의 treewidth 1이 HMM/linear CRF의 효율성을 보장. 더 나아가 SAT/CSP/database query optimization/quantum circuit simulation까지 treewidth가 결정적. **Parameterized complexity theory**의 대표 parameter. Treewidth를 이해하면 **어떤 문제가 근사 없이 풀 수 있는가**의 경계를 정확히 파악.

---

## 📐 수학적 선행 조건

- [Ch5-01 Variable Elimination Algorithm](./01-variable-elimination.md)
- [Ch2-04 Junction Tree Algorithm](../ch2-factor-graph/04-junction-tree.md): chordal graph
- Graph theory: tree decomposition

---

## 📖 직관적 이해

### Tree Decomposition

Graph $\mathcal{G}$의 **tree decomposition**은 tree $\mathcal{T}$와 각 tree node $t \in \mathcal{T}$에 subset $B_t \subseteq V(\mathcal{G})$ (**bag**)을 할당, 조건:

1. **Vertex coverage**: 모든 $v \in V$에 대해 어떤 $t$에서 $v \in B_t$
2. **Edge coverage**: 모든 edge $(u, v) \in E$에 대해 어떤 $t$에서 $u, v \in B_t$
3. **Running intersection**: 각 $v$에 대해 $\{t : v \in B_t\}$는 $\mathcal{T}$의 connected subtree

### Treewidth 정의

$$\text{tw}(\mathcal{G}) := \min_{\text{tree decomp } (\mathcal{T}, \{B_t\})} \max_t |B_t| - 1$$

**직관**: Tree-like structure의 "breadth". Tree = treewidth 1, cycle = treewidth 2, grid = $O(\sqrt N)$.

### Graph Examples

| Graph | Treewidth |
|-------|-----------|
| Tree | 1 |
| Cycle $C_n$ | 2 |
| $k$-tree | $k$ |
| Grid $L \times L$ | $L$ |
| Complete $K_n$ | $n - 1$ |
| Series-parallel | 2 |
| Planar graph | $O(\sqrt n)$ (worst case) |

### Inference Complexity

**Theorem**: Exact inference on graphical model with treewidth $\omega$는 $O(n \cdot d^{\omega + 1})$.

- Junction tree의 max clique size = $\omega + 1$
- 각 clique message가 $d^{\omega + 1}$ entries

**Exponential in $\omega$** — treewidth가 실용적 복잡도를 결정.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Tree Decomposition

$\mathcal{T} = (V_T, E_T)$: tree. 각 $t \in V_T$에 bag $B_t \subseteq V$ 할당. 조건:

**(T1) Coverage**: $\bigcup_t B_t = V$

**(T2) Edge**: 모든 $(u, v) \in E$에 대해 $\exists t : u, v \in B_t$

**(T3) Running Intersection**: 모든 $v \in V$에 대해 $T_v := \{t \in V_T : v \in B_t\}$가 $\mathcal{T}$에서 connected

### 정의 2.2 — Width and Treewidth

**Width**: $\text{width}(\mathcal{T}, \{B_t\}) := \max_t |B_t| - 1$

**Treewidth**: $\text{tw}(\mathcal{G}) := \min_{\text{all tree decomps}} \text{width}(\mathcal{T}, \{B_t\})$

### 정의 2.3 — Elimination Ordering and Induced Graph

Elimination order $\sigma$에서 **induced graph** $\mathcal{G}_\sigma$는:
- $\mathcal{G}$의 모든 edge 포함
- **Fill-in edges**: $x_i$ 제거 시 $x_i$의 남은 이웃들이 모두 서로 연결되도록 edge 추가

**Clique size**: Step $i$에서 $x_i$ + neighbors의 크기 = elimination 시의 clique 크기.

**Relation**: $\text{tw}(\mathcal{G}) = \min_\sigma \max_i |\{x_i\} \cup N_\sigma(x_i)| - 1$

### 정의 2.4 — Chordal Graph and Perfect Elimination Ordering

Graph $\mathcal{G}$가 **chordal** ⟺ **perfect elimination ordering (PEO)** 존재 ⟺ no cycle of length $\geq 4$ without chord.

**PEO**: $\sigma$ in which 각 $x_i$ 제거 시 **fill-in edge 없음** (이미 neighbors가 서로 연결).

**Fact**: Chordal graph의 max clique size = treewidth + 1.

---

## 🔬 정리와 증명

### 정리 2.1 — Tree Decomposition ⟺ Chordal Graph

**명제**: Graph $\mathcal{G}$의 treewidth = min triangulation의 max clique size - 1:
$$\text{tw}(\mathcal{G}) = \min_{\mathcal{G}' \supseteq \mathcal{G}, \mathcal{G}' \text{ chordal}} (\text{max clique}(\mathcal{G}') - 1)$$

**증명 개요**:

**(≤)** Triangulation $\mathcal{G}'$의 max clique를 각 tree decomposition bag으로 → valid decomposition with width = max clique - 1. 

**(≥)** Optimal tree decomposition 주어지면, bag들을 clique로 만드는 fill-in edges 추가 → chordal graph. Max clique ≤ max bag size. 따라서 $\leq$ treewidth + 1.

상세: Blair-Peyton 1993. $\square$

### 정리 2.2 — Inference Complexity $O(n \cdot d^{\text{tw}+1})$

**명제**: Graphical model with $n$ variables, cardinality $d$, treewidth $\omega$에서 exact inference (marginal, MAP)는 $O(n \cdot d^{\omega + 1})$.

**증명**:

1. Optimal tree decomposition $(\mathcal{T}, \{B_t\})$, width $= \omega$.
2. Junction tree에서 각 bag = clique, size $\leq \omega + 1$.
3. 각 clique의 potential: $d^{\omega + 1}$ entries.
4. Message passing: 각 tree edge에 message, size $d^{\omega + 1}$.
5. $\mathcal{T}$의 edge 수 $= |V_T| - 1 \leq n$ (bag 수 bounded by $n$).

Total: $O(n \cdot d^{\omega + 1})$. $\square$

### 정리 2.3 — Min Treewidth is NP-Hard

**명제** (Arnborg-Corneil-Proskurowski 1987): "Does $\text{tw}(\mathcal{G}) \leq k$?" decision problem는 NP-complete.

**증명 개요**:

**Reduction from 3-COLORING** (classic NP-complete):

Graph $\mathcal{G}$가 3-colorable ⟺ 수정된 graph $\mathcal{G}'$이 treewidth $\leq k$? (복잡한 gadget construction).

더 강한 결과: $\text{tw}(\mathcal{G}) \leq k$ check는 **fixed $k$에 대해 polynomial** (Bodlaender's algorithm, $O(n)$ for fixed $k$). 하지만 exponential in $k$.

**Practical implication**: 일반 그래프의 optimal treewidth 계산은 어렵지만, 작은 treewidth를 찾는 것이 목표라면 feasible.

$\square$

### 정리 2.4 — Heuristic Comparisons

**Empirical observation**: 다양한 heuristics (min-fill, min-weight, min-degree)가 서로 다른 graph class에서 다른 성능.

**Min-fill**: 매 step에서 least fill-in edges 추가하는 vertex 선택.
- 일반적으로 best in practice
- 복잡도 $O(n^3)$ per iteration (fill-in 계산)

**Min-degree**: 매 step에서 lowest-degree vertex 선택.
- 빠름, 자주 min-fill과 비슷
- Grid에서 terrible

**Maximum cardinality search (MCS)** (Tarjan-Yannakakis 1984):
- Output chordal graph with guaranteed "good" properties
- $O(n + m)$ time — fastest

**Nested dissection** (Lipton-Tarjan 1979): Planar graph에서 optimal treewidth ratio.

이들 모두 approximation ratio guarantees 없이 empirical하지만 실용적.

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def min_fill_triangulation(G):
    """Min-fill triangulation."""
    G = G.copy()
    order = []
    remaining = set(G.nodes())
    fill_edges = []
    
    while remaining:
        # Best vertex: lowest fill
        best_v = None
        best_fill = float('inf')
        best_neighbors = None
        for v in remaining:
            neighbors = [u for u in G.neighbors(v) if u in remaining]
            fill = 0
            for a, b in combinations(neighbors, 2):
                if not G.has_edge(a, b):
                    fill += 1
            if fill < best_fill:
                best_fill = fill
                best_v = v
                best_neighbors = neighbors
        
        # Add fill edges
        for a, b in combinations(best_neighbors, 2):
            if not G.has_edge(a, b):
                G.add_edge(a, b)
                fill_edges.append((a, b))
        
        order.append(best_v)
        remaining.remove(best_v)
    
    return G, order, fill_edges

def compute_treewidth_from_triangulation(G_tri):
    """Max clique size - 1 in chordal graph."""
    cliques = list(nx.find_cliques(G_tri))
    return max(len(c) for c in cliques) - 1

# 예시 1: Tree
T = nx.balanced_tree(2, 3)  # Binary tree depth 3
G_tri, order, fills = min_fill_triangulation(T)
print(f"Tree treewidth: {compute_treewidth_from_triangulation(G_tri)}")
print(f"Fill-in edges: {len(fills)}")

# 예시 2: Cycle
C = nx.cycle_graph(5)
G_tri, order, fills = min_fill_triangulation(C)
print(f"\n5-cycle treewidth: {compute_treewidth_from_triangulation(G_tri)}")
print(f"Fill-in edges: {len(fills)}")

# 예시 3: Complete graph
K5 = nx.complete_graph(5)
G_tri, order, fills = min_fill_triangulation(K5)
print(f"\nK_5 treewidth: {compute_treewidth_from_triangulation(G_tri)}")
print(f"Fill-in edges: {len(fills)}")  # 0 (already chordal)

# 예시 4: Grid
for L in [3, 4, 5, 6, 7]:
    G = nx.grid_2d_graph(L, L)
    G_tri, order, fills = min_fill_triangulation(G)
    tw = compute_treewidth_from_triangulation(G_tri)
    print(f"{L}x{L} grid: treewidth ≈ {tw}, fill-ins = {len(fills)}")

# Treewidth vs grid size
grid_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
tws = []
for L in grid_sizes:
    G = nx.grid_2d_graph(L, L)
    G_tri, order, fills = min_fill_triangulation(G)
    tws.append(compute_treewidth_from_triangulation(G_tri))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(grid_sizes, tws, 'o-')
axes[0].plot(grid_sizes, grid_sizes, 'k--', label='y = L (theoretical)')
axes[0].set_xlabel('Grid size L')
axes[0].set_ylabel('Treewidth (min-fill)')
axes[0].set_title('Grid MxM의 treewidth (grows as L)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Inference complexity (log scale)
d = 10  # cardinality
ns = [L*L for L in grid_sizes]
complexity = [n * d**(tw + 1) for n, tw in zip(ns, tws)]
axes[1].semilogy(grid_sizes, complexity, 'o-')
axes[1].set_xlabel('Grid size L')
axes[1].set_ylabel('Complexity O(n · d^(tw+1))')
axes[1].set_title(f'Inference complexity (d={d})')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('treewidth_complexity.png', dpi=120, bbox_inches='tight')
plt.show()

print("\n관찰: Grid treewidth는 선형 증가 → inference complexity exponential 증가")
print("이것이 이미지 분할 MRF의 exact inference가 불가능한 이유")
```

**출력 예시**:
```
Tree treewidth: 1
Fill-in edges: 0

5-cycle treewidth: 2
Fill-in edges: 2

K_5 treewidth: 4
Fill-in edges: 0

3x3 grid: treewidth ≈ 3, fill-ins = 5
4x4 grid: treewidth ≈ 4, fill-ins = 16
5x5 grid: treewidth ≈ 5, fill-ins = 35
6x6 grid: treewidth ≈ 6, fill-ins = 65
7x7 grid: treewidth ≈ 7, fill-ins = 108

관찰: Grid treewidth는 선형 증가 → inference complexity exponential 증가
이것이 이미지 분할 MRF의 exact inference가 불가능한 이유
```

Grid의 treewidth가 linear로 증가 → inference complexity가 $O(d^L)$ exponential.

---

## 🔗 AI/ML 연결

### Image Segmentation Intractability

$256 \times 256$ image에서 pixel-level MRF:
- Treewidth $\approx 256$
- Complexity $O(|V|^{257}) \approx$ 우주 나이 × 우주 원자 수
- **Approximate inference 필수**: mean-field, graph cut, neural

### Dependency Parsing Tractability

Linear chain dependency parse: treewidth 1 → $O(n d^2)$.  
Projective (nested tree): treewidth 2 → $O(n^3)$.  
Non-projective (arbitrary): treewidth O(n) → brute force $O(n^n)$ but **Matrix-Tree Theorem**으로 $O(n^3)$ (structural trick).

### Quantum Circuit Simulation

Sycamore (Google 2019) quantum supremacy:
- Circuit의 tensor network treewidth $\approx 50$+
- Classical simulation: $2^{50}$+ operations
- Google claim: 10000 years for best classical — "quantum supremacy"
- IBM counter: better contraction → 2.5 days (still shows treewidth sensitivity)

### Database Join Optimization

Query plan optimization:
- Each join = factor product
- Plan = elimination order
- **Min-weight** heuristic: minimize intermediate table size
- Actually same problem as PGM treewidth!

### SAT Solving

CNF formula → factor graph (each clause = factor). Treewidth small formulas → polynomial SAT (tractable). Generic SAT has treewidth $\Omega(n)$ → NP-hard.

**Parameterized complexity**: "SAT with treewidth $k$" is **FPT** (fixed parameter tractable) in $k$ — $O(2^k \cdot n)$.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Discrete variables | Continuous는 treewidth가 **$\infty$** 가능 (Gaussian 예외) |
| Min-fill heuristic | Approximation, optimal treewidth NP-hard |
| Static graph | Dynamic query는 multi-query setup에서 더 복잡 |
| Exact inference | Large treewidth면 approximate 필수 |

**주의**: Treewidth는 **graph structure**의 성질. 같은 graph에 다른 cardinality / potential가 있으면 **같은 treewidth, 같은 복잡도 bound**. 실제 runtime은 cardinality와 sparsity에 따라 훨씬 좋을 수도 있음.

---

## 📌 핵심 정리

$$\boxed{\text{tw}(\mathcal{G}) = \min_{\sigma} \max_i |\{x_i\} \cup N_\sigma(x_i)| - 1}$$

$$\boxed{\text{Inference complexity: } O(n \cdot d^{\text{tw} + 1})}$$

| Graph | Treewidth | Complexity |
|-------|-----------|------------|
| Tree | 1 | $O(n d^2)$ |
| Cycle | 2 | $O(n d^3)$ |
| Grid $L \times L$ | $L$ | $O(L^2 d^{L+1})$ |
| $K_n$ | $n - 1$ | $O(d^n)$ (brute force) |

**Key theorem** (Arnborg-Corneil-Proskurowski 1987): Min treewidth is NP-hard.  
**Practical heuristics**: min-fill, MCS, nested dissection.

---

## 🤔 생각해볼 문제

**문제 1** (기초): **Petersen graph** (유명한 non-planar graph, 10 vertices)의 treewidth를 추측하고 이유를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Petersen graph**: 10 vertices, 3-regular, highly symmetric, non-planar.

**실제 treewidth**: 4.

**이유**:
- 3-regular로 모든 vertex가 이웃 3 → min-degree heuristic은 treewidth $\geq 3$
- 5-cycle 포함 (outer, inner pentagon) → cycle이 triangulation에 chord 필요
- 복잡한 symmetric structure → 4 정도가 필요

**Inference 복잡도**: $10 \cdot d^5$ — still tractable for small $d$.

**실용적 함의**: Biological networks, social networks 같은 small-world graph는 종종 treewidth 5~10 → exact inference 가능하지만 경계선. Approximation (mean-field, loopy BP) 종종 사용.

</details>

**문제 2** (심화): Grid $L \times L$의 treewidth가 $\geq L$임을 증명하라 (Bramble argument).

<details>
<summary>힌트 및 해설</summary>

**Bramble** (Seymour-Thomas 1993): Graph의 "**brambles**"의 order가 treewidth의 lower bound.

**Bramble 정의**: $\mathcal{B} = \{B_1, B_2, \ldots\}$ — connected subgraphs such that 어떤 두 bramble도 touch (edge or common vertex).

**Order**: $\min$ hitting set size ($\mathcal{B}$의 모든 member를 intersect하는 최소 vertex set 크기).

**Grid에서의 bramble**:

$L \times L$ grid에서 column-row crossings 사용:
- 각 row와 column을 path로
- 모든 row와 column이 pairwise touching

Crossing bramble order $\geq L$ (각 row, column 다른 vertex로 hit 필요).

**따라서** $\text{tw} \geq \text{bramble number} - 1 \geq L - 1$.

**Upper bound** (matching): Row-by-row elimination → width $\leq L$. 따라서 $\text{tw}(\text{grid}_{L \times L}) = L$ (+/- 1 depending on exact definition).

**함의**: Grid의 treewidth는 structure에 intrinsic — 어떤 smart ordering도 이 lower bound를 피할 수 없음.

**Ising on grid**: Spin configuration의 exact partition function $\sum_s e^{\beta \sum s_i s_j}$ — treewidth-limited. $100 \times 100$ → $2^{100}$ brute → approximation 필수.

</details>

**문제 3** (AI 연결): Transformer의 self-attention은 complete graph ($K_n$). 이는 treewidth $n-1$로 exact inference가 불가능. 어떻게 Transformer는 이 문제를 우회하는가?

<details>
<summary>힌트 및 해설</summary>

**Transformer의 constraint**:
- Output을 "exact marginal" 아닌 **parametric distribution**으로 표현
- Inference = forward pass (not DP)
- $O(T^2)$ per layer (attention) × $L$ layers = $O(LT^2)$ - polynomial

**PGM 관점에서의 trick**:

1. **Fixed computation depth**: Transformer는 $L$ layers. Treewidth-exponential이 아닌 layer-linear.

2. **Amortized inference**: 한 번의 forward pass로 모든 variables의 marginal을 동시 계산 (conditional on input). PGM의 "compute one marginal at a time" 대비 훨씬 효율.

3. **Parametric approximation**: Factorized output $p(y | x) = \prod p(y_i | x)$ (causal LM) 또는 token-independent (BERT). 이는 **mean-field approximation**과 유사.

4. **Representation learning**: Attention weight matrix = "which token matters for which" — explicit factor structure 대신 **learned factor importance**.

**구체적 수학 비교**:

Exact PGM on $K_n$:
$$p(y) = \frac{1}{Z} \prod_{(i, j)} \phi_{ij}(y_i, y_j)$$
Inference: $O(K^n)$ brute force.

Transformer LM:
$$p(y) = \prod_t p(y_t | y_{<t})$$
Inference: $O(n \cdot \text{NN forward})$ per step, $O(n^2)$ total via parallel.

**핵심 차이**: Transformer는 **structured inference를 parameteric model로 대체**. Exact inference 포기하지만 scalable, learnable, flexible.

**Trade-off**:
- PGM: interpretable, exact, small data
- Transformer: large data, black-box, approximate

**최신 연구**:
- **Factor graph transformer** (Bai et al. 2023): Explicit factor structure + attention
- **Transformer as implicit MRF**: Attention as BP-like
- **Differentiable VE**: Smooth relaxation of exact inference

결론: Transformer = "deep learning era의 PGM approximation" — treewidth 문제를 "NN expressiveness + SGD"로 우회. Ch7-05에서 자세히.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Variable Elimination Algorithm](./01-variable-elimination.md) | [📚 README](../README.md) | [03. Clique Tree와 Junction Tree ▶](./03-clique-tree.md) |

</div>
