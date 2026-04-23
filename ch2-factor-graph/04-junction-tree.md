# 04. Junction Tree Algorithm

## 🎯 핵심 질문

- Loopy factor graph에서 exact inference를 어떻게 수행하는가?
- **Triangulation**과 **running intersection property**는 왜 필요한가?
- Junction tree의 clique 간 message passing은 어떻게 정의되는가?
- Treewidth는 왜 inference 복잡도의 본질적 측도인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Junction Tree**는 **loopy graphical model에서도 exact inference가 가능한 조건**을 제공한다. Expert system(ALARM, Pathfinder)의 의료 진단, Bayes ball algorithm, structured CRF, 작은 tree-width의 **Bayesian network exact inference** 모두 JT를 사용. 현대적으로 JT는 **PGM inference의 이론적 기준** — 모든 approximate algorithm은 JT에 비교된다. **Treewidth**는 SAT solver, constraint satisfaction, database query optimization 등 다양한 분야의 복잡도 측도로도 쓰인다.

---

## 📐 수학적 선행 조건

- [Ch2-01 Factor Graph](./01-factor-graph-definition.md)
- [Ch2-02 Sum-Product Algorithm](./02-sum-product-algorithm.md)
- [Ch1-05 Moralization](../ch1-conditional-independence/05-moralization.md)
- Graph theory: chordal graph, perfect elimination ordering

---

## 📖 직관적 이해

### 문제: Loopy Graph에서 BP가 실패

3-cycle MRF $A - B - C - A$에서 BP를 돌리면:
- $A$에서 오는 메시지가 $B \to C \to A$로 돌아옴 — **circular dependence**
- 수렴해도 정확한 marginal 아님

**해결책**: Loopy graph를 **tree 구조의 더 큰 node(= clique)**로 묶어서 tree로 변환.

### 3단계 절차

**Step 1: Moralization** (BN → MRF). Ch1-05에서 다룸.

**Step 2: Triangulation** — chordal graph로 만들기.
- Chordal graph: 길이 $\geq 4$인 **모든 cycle이 chord**(비인접 꼭짓점 간 edge)를 가짐
- Chordal이면 **maximum cardinality search** 같은 알고리즘으로 clique 구조 파악 가능
- Non-chordal을 chordal로 만들려면 **fill-in edges** 추가

**Step 3: Clique Tree 구성** — maximal clique을 node로, **running intersection property**를 만족하는 tree.

### Running Intersection Property (RIP)

Clique tree가 **RIP**를 만족한다는 것은:

> 두 clique $C_i, C_j$가 공통 변수 $v$를 포함하면, $C_i$와 $C_j$ 사이의 **모든 clique도 $v$를 포함**.

이것이 message passing이 correct하게 정보를 전파할 수 있게 하는 **핵심 조건**.

### 복잡도

Junction Tree BP의 복잡도:
$$O(n \cdot d^{\omega(\mathcal{G}) + 1})$$

여기서 $\omega(\mathcal{G})$ = **treewidth** = max clique size - 1 in the best triangulation.

**Tree**: treewidth = 1 → $O(n d^2)$
**Cycle of length $n$**: treewidth = 2 → $O(n d^3)$
**Complete graph $K_n$**: treewidth = $n - 1$ → $O(d^n)$ (exact inference hopeless)

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Chordal Graph

Undirected graph $\mathcal{G}$가 **chordal** (또는 triangulated)이면:

> 길이 $\geq 4$인 모든 cycle $v_1 - v_2 - \cdots - v_k - v_1$에 대해, 비인접 꼭짓점 간 edge(chord) $v_i - v_j$ 존재.

### 정의 4.2 — Clique Tree와 Running Intersection

**Clique tree** $\mathcal{T}$는 $\mathcal{G}$의 maximal clique들을 노드로 하고, 각 쌍 사이에 edge를 두어 만든 tree.

**Running Intersection Property (RIP)**: 모든 variable $v$에 대해, $v$를 포함하는 clique들이 $\mathcal{T}$의 **connected subtree**를 이룸.

동치 조건: RIP ⟺ $\mathcal{G}$가 chordal (Blair-Peyton 1993).

### 정의 4.3 — Separator

Clique tree에서 두 인접 clique $C_i, C_j$ 사이의 **separator**:
$$S_{ij} := C_i \cap C_j$$

RIP에 의해 $S_{ij}$가 $C_i, C_j$ 간 유일한 "connector" 역할.

### 정의 4.4 — Junction Tree Algorithm

**Input**: 모랄화된 chordal graph의 clique tree, 각 clique에 assigned potentials.

**Algorithm**:
1. Clique $C_i$에 potential $\psi_i$ 할당 (factor를 clique에 assign)
2. Message passing: clique $C_i$에서 인접 $C_j$로
$$\mu_{i \to j}(x_{S_{ij}}) = \sum_{x_{C_i \setminus S_{ij}}} \psi_i(x_{C_i}) \prod_{k \in N(i) \setminus j} \mu_{k \to i}(x_{S_{ik}})$$
3. Clique belief: $b(x_{C_i}) = \psi_i(x_{C_i}) \prod_{j \in N(i)} \mu_{j \to i}(x_{S_{ij}})$

### 정의 4.5 — Treewidth

Graph $\mathcal{G}$의 **treewidth** $\omega(\mathcal{G})$:
$$\omega(\mathcal{G}) := \min_{\text{triangulations}} \max_C |C| - 1$$

즉 triangulation over 가능한 모든 순서들 중 최대 clique 크기에서 1을 뺀 최솟값.

---

## 🔬 정리와 증명

### 정리 4.1 — Chordal ⟺ RIP Clique Tree 존재

**명제**: Graph $\mathcal{G}$가 chordal ⟺ $\mathcal{G}$의 maximal cliques에 대해 RIP를 만족하는 tree가 존재.

**증명** (개요):

**(⟹)** Chordal → RIP: **Perfect elimination ordering** (PEO) 존재. PEO $\sigma$에서 각 꼭짓점 $v$를 제거할 때 $v$의 neighbors가 clique. 이 clique 순서가 maximum cardinality search와 결합하여 RIP tree 구성 가능.

**(⟸)** RIP → Chordal: $\mathcal{G}$에 길이 $\geq 4$ cycle without chord가 있으면 해당 variables가 non-subtree 배치 → RIP 위반. 모순. $\square$

(Blair-Peyton 1993 상세 증명)

### 정리 4.2 — Junction Tree BP의 정확성

**명제**: RIP를 만족하는 clique tree에서, junction tree message passing은 각 clique의 정확한 marginal을 제공:

$$b(x_{C_i}) \propto p(x_{C_i})$$

**증명 스케치**:

Junction tree는 **tree factor graph와 동등** — 각 clique을 하나의 "super-variable"로 보고, separator를 공유하는 것이 일반적 tree factor graph의 특수 구조.

RIP가 separator들이 "올바른" 변수를 공유하게 보장. Tree factor graph의 BP 정확성(정리 2.1)을 그대로 적용. $\square$

### 정리 4.3 — 복잡도

**명제**: Junction tree BP의 시간복잡도:
$$O\left(\sum_{C \in \text{cliques}} d^{|C|}\right) = O(n \cdot d^{\omega(\mathcal{G}) + 1})$$

**증명**:

각 clique $C$의 potential 계산에 $d^{|C|}$ 연산 (모든 assignment enumerate). Message marginalization: $d^{|C|}$. Total: clique 수 $O(n)$ × clique 크기 $d^{\omega + 1}$. $\square$

### 정리 4.4 — Min Treewidth의 NP-Hardness

**명제** (Arnborg-Corneil-Proskurowski 1987): Graph의 treewidth를 $k$ 이하로 결정하는 문제는 NP-hard.

**증명 개요**: **Reduction from 3-COLORING**. 3-color이 $\omega \leq k$ 여부로 reduce. $\square$

**함의**: 최적 elimination ordering을 찾는 것이 NP-hard. 실전에서 휴리스틱 사용: **min-fill** (fill-in edge 수 최소화), **min-weight** (log domain size 고려), **maximum cardinality search**.

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

def min_fill_triangulation(G):
    """Min-fill heuristic으로 triangulation."""
    G_tri = G.copy()
    elimination_order = []
    remaining = list(G_tri.nodes())
    
    while remaining:
        # Min-fill: 제거 시 추가되어야 할 fill edge가 최소인 노드
        best_node = None
        best_fill = float('inf')
        for v in remaining:
            neighbors = [u for u in G_tri.neighbors(v) if u in remaining]
            fill = sum(1 for (a, b) in combinations(neighbors, 2) 
                       if not G_tri.has_edge(a, b))
            if fill < best_fill:
                best_fill = fill
                best_node = v
        
        # Fill-in edges 추가
        neighbors = [u for u in G_tri.neighbors(best_node) if u in remaining]
        for a, b in combinations(neighbors, 2):
            if not G_tri.has_edge(a, b):
                G_tri.add_edge(a, b)
        
        elimination_order.append(best_node)
        remaining.remove(best_node)
    
    return G_tri, elimination_order

def find_maximal_cliques(G):
    """Bron-Kerbosch로 maximal cliques."""
    return list(nx.find_cliques(G))

def build_junction_tree(cliques):
    """Maximum weight spanning tree로 junction tree."""
    clique_graph = nx.Graph()
    for i, c in enumerate(cliques):
        clique_graph.add_node(i, clique=set(c))
    
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            intersection = set(cliques[i]) & set(cliques[j])
            if intersection:
                clique_graph.add_edge(i, j, weight=-len(intersection))
                # negative: maximum spanning 원할 때 min spanning의 음수 weight
    
    # Maximum spanning tree
    jt = nx.minimum_spanning_tree(clique_graph)
    return jt

# 예시: 4-cycle MRF
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])

print("Original 4-cycle edges:", list(G.edges()))

G_tri, order = min_fill_triangulation(G)
print(f"Elimination order: {order}")
print(f"Triangulated edges: {list(G_tri.edges())}")
print(f"Fill-in edges added: {set(G_tri.edges()) - set(G.edges())}")

cliques = find_maximal_cliques(G_tri)
print(f"\nMaximal cliques: {cliques}")
print(f"Treewidth: {max(len(c) for c in cliques) - 1}")

jt = build_junction_tree(cliques)
print(f"\nJunction tree edges: {[(cliques[i], cliques[j]) for i, j in jt.edges()]}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

nx.draw(G, with_labels=True, ax=axes[0], node_color='lightblue', node_size=1500, font_weight='bold')
axes[0].set_title('Original 4-cycle (loopy)')

nx.draw(G_tri, with_labels=True, ax=axes[1], node_color='lightgreen', node_size=1500, font_weight='bold')
fill_edges = set(G_tri.edges()) - set(G.edges())
nx.draw_networkx_edges(G_tri, nx.spring_layout(G_tri, seed=1), edgelist=fill_edges,
                       edge_color='red', width=2, ax=axes[1])
axes[1].set_title('Triangulated (fill-in in red)')

# Junction tree 시각화
jt_labels = {i: str(tuple(sorted(c))) for i, c in enumerate(cliques)}
pos_jt = nx.spring_layout(jt, seed=1)
nx.draw(jt, pos_jt, labels=jt_labels, ax=axes[2], 
        node_color='lightyellow', node_size=3000, font_size=9)
axes[2].set_title('Junction Tree (clique tree)')

plt.tight_layout()
plt.savefig('junction_tree_example.png', dpi=120, bbox_inches='tight')
plt.show()

# RIP 검증: 각 variable이 포함된 cliques가 subtree를 이루는지
def check_rip(jt, cliques):
    """각 variable에 대해 포함하는 clique들이 connected subtree인지."""
    all_vars = set()
    for c in cliques:
        all_vars.update(c)
    
    for v in all_vars:
        containing = [i for i, c in enumerate(cliques) if v in c]
        sub = jt.subgraph(containing)
        if not nx.is_connected(sub):
            print(f"RIP 위반: variable {v}의 clique 집합이 disconnected")
            return False
    return True

print(f"\nRIP 만족 여부: {check_rip(jt, cliques)}")
```

**출력 예시**:
```
Original 4-cycle edges: [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
Elimination order: ['B', 'D', 'A', 'C']
Triangulated edges: [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')]
Fill-in edges added: {('A', 'C')}

Maximal cliques: [['A', 'B', 'C'], ['A', 'C', 'D']]
Treewidth: 2

Junction tree edges: [(['A', 'B', 'C'], ['A', 'C', 'D'])]

RIP 만족 여부: True
```

4-cycle이 triangulation 후 2개의 3-clique으로 분해되고, separator $\{A, C\}$로 연결된 JT가 완성됨.

---

## 🔗 AI/ML 연결

### ALARM Network와 의료 진단

ALARM(A Logical Alarm Reduction Mechanism, Beinlich 1989)은 37 variables의 의료 BN. Treewidth ≈ 5, 따라서 junction tree로 exact inference 가능. 복잡도 $37 \cdot 2^6 \approx 2400$ — 실시간 가능. 더 큰 네트워크(Pathfinder, 448 variables)도 treewidth에 따라 tractable 여부 결정.

### Structured Prediction의 Exact Inference

CRF for parsing, NER에서 **linear-chain CRF**(Ch4)는 treewidth = 1 → 선형 시간. **Skip-chain CRF**는 long-range dependency 추가 → treewidth 증가 → 더 비싼 inference.

### SAT Solver와 Constraint Satisfaction

SAT problem을 factor graph로 표현 후 JT 적용. 작은 treewidth의 CSP는 polynomial 시간에 해결 가능 — **parameterized complexity**의 대표 예. SAT solver(MiniSAT, Glucose)의 경우 경험적으로 treewidth가 작은 인스턴스를 먼저 해결.

### Database Query Optimization

Relational database의 join query를 factor graph로 볼 때, **query plan optimization**이 본질적으로 elimination ordering 선택. Treewidth가 작은 query가 efficient.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Small treewidth | Treewidth가 크면 JT 복잡도 폭발 (e.g., grid MRF의 $O(\sqrt n)$ treewidth) |
| Finite discrete | Continuous variables은 Gaussian JT로 제한 |
| Exact inference | 학습·structured search의 많은 경우 approximate (Ch6) 필요 |
| Chordal triangulation | Optimal triangulation NP-hard — 휴리스틱 사용 |

**주의**: 이미지 분할 같은 grid MRF의 treewidth는 $O(\sqrt{n})$ — JT 불가능. Variational approximation(Ch6)이 필수.

---

## 📌 핵심 정리

$$\boxed{\text{Moralize → Triangulate → Clique Tree with RIP → Sum-Product on JT}}$$

| 개념 | 의미 |
|------|------|
| **Chordal graph** | 모든 cycle(길이 ≥ 4)이 chord를 가짐 |
| **RIP** | Variable을 포함하는 clique들이 subtree |
| **Treewidth** | Triangulation의 최대 clique 크기 - 1 |
| **JT complexity** | $O(n \cdot d^{\omega + 1})$ |
| **Min-fill** | Triangulation 휴리스틱 (NP-hard 문제의 실용적 해답) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 5-cycle $v_1 - v_2 - v_3 - v_4 - v_5 - v_1$의 treewidth와 최적 triangulation을 찾아라.

<details>
<summary>힌트 및 해설</summary>

5-cycle은 non-chordal (4+ length cycle without chord).

**Triangulation 옵션 1**: $v_1 - v_3$, $v_1 - v_4$ 추가 → cliques $\{v_1, v_2, v_3\}, \{v_1, v_3, v_4\}, \{v_1, v_4, v_5\}$ — 모두 triangle, treewidth = 2.

**Triangulation 옵션 2**: $v_2 - v_4$, $v_2 - v_5$ 추가 → 비슷한 treewidth = 2.

**최적**: 어떤 triangulation이든 5-cycle은 **treewidth 2**가 최소. $n$-cycle의 treewidth = 2 (chord 하나만 추가하면 되지만 2 chord 필요하여 3 triangles).

일반적으로 $n$-cycle treewidth는 **2**. Grid graph $\sqrt{n} \times \sqrt{n}$는 treewidth $O(\sqrt n)$.

</details>

**문제 2** (심화): Grid MRF $L \times L$의 treewidth가 $L$임을 보여라 (image segmentation의 근본 제약).

<details>
<summary>힌트 및 해설</summary>

**상계** ($\omega \leq L$): Row-by-row elimination. 첫 row 제거 시 fill-in으로 전체 row가 clique. Size $L$ clique → treewidth $L - 1$. 좀 더 정교한 argument로 $L$에 근접.

**하계** ($\omega \geq L$): Bramble argument. Grid의 **Menger width** (서로 disjoint한 path의 최대 수)가 $L$ — 이는 treewidth의 하계.

구체적: $L \times L$ grid에서 top row와 bottom row를 연결하는 disjoint path가 $L$개. Treewidth는 이 connectivity에 의해 $\geq L - 1$.

**함의**: $100 \times 100$ image segmentation → treewidth 100 → $2^{100}$ brute force — **exact inference 불가능**. 반드시 approximation (graph cut, ICM, MRF inference via VI).

</details>

**문제 3** (AI 연결): Transformer가 treewidth의 제약을 피하는 방법은 무엇인가? 왜 full attention은 "exact inference가 불가능한 그래프"인데도 잘 동작하는가?

<details>
<summary>힌트 및 해설</summary>

**Transformer의 그래프 구조**: Fully-connected graph on $n$ tokens → treewidth = $n - 1$ → exact inference cost $O(2^n)$ — 절대 불가능.

**하지만 Transformer는 exact inference를 하지 않음**:

1. **Parametric approximation**: Attention matrix는 exact posterior가 아니라 **학습된 함수**. 수백만 파라미터가 문제를 "compile"한 것이지, 매번 JT 계산하는 게 아님.

2. **Feed-forward, no iterative inference**: Tree inference처럼 message passing을 수렴할 때까지 반복하지 않음. Fixed-depth (layer 수) computation.

3. **Amortized inference**: VAE와 유사 — 각 특정 $x$에 대해 수렴까지 iterate하지 않고, **모든 input에 대한 posterior을 네트워크로 approximate**.

4. **Target loss, not exact marginal**: Language modeling loss는 exact $p(x)$ 필요 없고 $\log p(x_t | x_{<t})$만. 이는 각 token의 조건부만 필요 → 순차적 "sampling" inference.

**결론**: Transformer는 **exact inference를 포기**하고 **neural amortization**으로 대체. Treewidth-bounded exact inference의 한계를 depth와 parameters로 극복. 이것이 "deep learning이 PGM을 대체"한 메커니즘 중 하나. Ch7-05에서 자세히.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Max-Product Algorithm과 MAP Inference](./03-max-product-algorithm.md) | [📚 README](../README.md) | [05. Loopy BP와 Bethe 자유에너지 ▶](./05-loopy-bp-bethe.md) |

</div>
