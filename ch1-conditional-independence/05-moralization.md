# 05. Moralization — DAG ↔ MRF 변환

## 🎯 핵심 질문

- DAG를 MRF로 변환할 때 왜 parents를 **moralize**(연결)해야 하는가?
- 이 변환이 **conditional independence를 보존하지 않는** 구체적 예시는?
- **I-map**, **P-map**, **Minimal I-map**은 어떻게 정의되고, 왜 minimal I-map은 unique한가?
- BN과 MRF는 서로를 완전히 표현할 수 있는가 — 아니라면 어떤 구조를 각자만 표현할 수 있는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Moralization**은 DAG의 추론 알고리즘(Junction Tree, Variable Elimination)이 **MRF 위에서 동작**하게 만드는 연결고리다. DAG의 d-separation을 MRF의 graph separation으로 번역하는 도구이기도 하다. **BN과 MRF의 표현력 차이**를 이해하면 **언제 어떤 모델을 써야 하는지**가 명확해진다 — causal semantics가 필요하면 BN, symmetric interaction이라면 MRF. **Chordal graph**(=moralize 후 cycle이 없는 것)는 exact inference가 가능한 그래프의 경계. DAG의 **Markov equivalence class** 역시 moralization과 v-structure로 특징지어진다.

---

## 📐 수학적 선행 조건

- [Ch1-02 Bayesian Network — DAG 기반 인수분해](./02-bayesian-network-factorization.md)
- [Ch1-03 d-separation](./03-d-separation.md)
- [Ch1-04 Markov Random Field와 Hammersley–Clifford](./04-markov-random-field.md)
- Graph theory: chordality, triangulation

---

## 📖 직관적 이해

### Moralization 절차

DAG $\mathcal{G}$가 주어지면 **moral graph** $\mathcal{M}(\mathcal{G})$을 다음과 같이 만든다:

1. **모든 edge의 방향 제거**
2. **공통 자식을 가진 모든 parents 쌍을 edge로 연결** (marry them!)

**이름의 유래**: 같은 자식의 부모들을 "결혼"시킨다는 농담 — Lauritzen & Spiegelhalter 1988의 조어.

### Moralization이 필요한 이유

DAG의 **v-structure** (collider) $X \to W \leftarrow Y$에서, $W$가 주어졌을 때 $X$와 $Y$는 **의존**. 만약 단순히 방향만 제거하면 undirected graph $X - W - Y$가 되는데, 이 MRF에서는 $W$가 separator가 되어 $X \perp\!\!\!\perp Y \mid W$가 강제됨 — **원래 DAG의 의도와 반대!**

따라서 $X$와 $Y$를 직접 연결하여, "$W$를 조건으로 주어도 $X - Y$가 직접 연결된 edge로 남아 있게" 한다. 이렇게 **collider dependency를 edge로 변환**.

```
원래 DAG (v-structure):           Moralized MRF:

    X ──► W ◄── Y       →           X ─────── Y
                                     \       /
                                      \     /
                                       \   /
                                        W
```

### CI 보존 여부

Moralization은 **I-map 관계**를 보존하지만 **일부 CI는 손실**. 구체적으로:

**보존되는 것**: DAG가 함의한 CI는 moral graph에서도 함의됨.

**손실되는 것**: DAG의 v-structure가 함의한 **unconditional independence** $X \perp\!\!\!\perp Y$는 moral graph에서 표현 불가 (edge가 추가되었으므로).

즉 moral graph는 DAG의 CI 구조의 **over-approximation** — 더 많은 edge, 더 적은 CI.

### BN vs MRF 표현력

어떤 CI 구조는 BN으로만 / MRF로만 표현 가능:

**BN-only**: v-structure $A \to C \leftarrow B$는 $A \perp\!\!\!\perp B$를 함의하지만 $A \not\perp\!\!\!\perp B \mid C$. **어떤 MRF도 이 조합을 표현 불가** (MRF에서 edge가 없으면 항상 CI; 있으면 항상 dependent).

**MRF-only**: 4-cycle $A - B - C - D - A$에서 $A \perp\!\!\!\perp C \mid B, D$와 $B \perp\!\!\!\perp D \mid A, C$를 동시 표현 가능. 어떤 **DAG도 이 두 CI를 동시에** 표현 불가 (어느 방향이든 v-structure가 추가됨).

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Moral Graph

DAG $\mathcal{G} = (V, E)$의 **moral graph** $\mathcal{M}(\mathcal{G}) = (V, E')$:

$$E' = \{(u, v) : (u, v) \in E \text{ or } (v, u) \in E\} \cup \{(u, v) : \exists w, (u, w) \in E \text{ and } (v, w) \in E\}$$

(방향 제거 + 공통 자식의 parents 연결, 모두 undirected)

### 정의 5.2 — I-map, D-map, P-map

분포 $P$와 그래프 $\mathcal{G}$(BN 또는 MRF)에 대해:

**I-map** (Independence map): $\mathcal{G}$가 $P$의 I-map ⟺ $\mathcal{G}$가 함의하는 모든 CI가 $P$에서 성립. (즉 $\mathcal{G}$는 $P$보다 제한적; $P$에 더 많은 CI가 있을 수 있음)

**D-map** (Dependence map): $P$의 모든 CI가 $\mathcal{G}$에서 함의됨. (즉 $\mathcal{G}$는 $P$보다 관대; $P$에 있는 dependency가 $\mathcal{G}$에도 있음)

**P-map** (Perfect map): I-map이자 D-map — $\mathcal{G}$의 CI와 $P$의 CI가 정확히 일치.

### 정의 5.3 — Minimal I-map

$\mathcal{G}$가 $P$의 **minimal I-map**이면 $\mathcal{G}$의 어떤 edge를 제거해도 I-map이 아니게 됨.

### 정의 5.4 — Markov Blanket

MRF에서 변수 $v$의 **Markov blanket** $\text{MB}(v) := N(v)$ (이웃).

BN에서 변수 $v$의 **Markov blanket** = $v$의 **parents** + **children** + **children의 다른 parents** (co-parents).

**성질**: $v \perp\!\!\!\perp V \setminus (\{v\} \cup \text{MB}(v)) \mid \text{MB}(v)$.

---

## 🔬 정리와 증명

### 정리 5.1 — Moralization이 I-map을 보존

**명제**: $\mathcal{G}$가 DAG이고 $P$가 $\mathcal{G}$에 대해 인수분해되면, $\mathcal{M}(\mathcal{G})$는 $P$의 MRF I-map이다.

**증명**: 

$P$의 DAG factorization $p(x) = \prod_v p(x_v | x_{\text{pa}(v)})$. 각 항 $p(x_v | x_{\text{pa}(v)})$은 $\{v\} \cup \text{pa}(v)$에만 의존하는 함수. $\mathcal{M}(\mathcal{G})$에서 $v$와 $\text{pa}(v)$의 모든 꼭짓점이 **서로 연결** (원래 edge $\text{pa}(v) \to v$ + moralization으로 $\text{pa}(v)$ 쌍끼리 연결):

이 $\{v\} \cup \text{pa}(v)$는 $\mathcal{M}(\mathcal{G})$에서 **clique**.

따라서 $\phi_v(x_{\{v\} \cup \text{pa}(v)}) := p(x_v | x_{\text{pa}(v)})$로 놓으면
$$p(x) = \prod_v \phi_v(x_{\{v\} \cup \text{pa}(v)})$$

이는 $\mathcal{M}(\mathcal{G})$의 clique potential 형태 (정규화는 자동). 따라서 $P$는 $\mathcal{M}(\mathcal{G})$의 MRF이고, Hammersley-Clifford(positive density 가정 하)에 의해 global Markov. 즉 $\mathcal{M}(\mathcal{G})$가 I-map. $\square$

### 정리 5.2 — Moralization은 일부 CI 손실

**명제**: V-structure $A \to W \leftarrow B$가 포함된 DAG $\mathcal{G}$의 moral graph에서는 $A \perp\!\!\!\perp B$가 **표현되지 않는다**.

**증명**: 

Moral graph에서 $A - W, B - W, A - B$가 모두 edge(마지막은 moralization으로 추가). $A$와 $B$ 사이에 직접 edge가 있으므로, MRF의 pairwise Markov property의 대우로, **어떤 조건 집합 $Z$에서도 $A \perp\!\!\!\perp B \mid Z$가 강제되지 않음**.

그러나 원래 DAG에서 $A \perp\!\!\!\perp B$ (marginal). Moral graph에서는 이 unconditional CI를 그래프 구조로 표현할 방법이 없음. $\square$

### 정리 5.3 — d-Separation ⟺ 적절한 Moral Graph의 Graph Separation

**명제** (Lauritzen 1996, Koller & Friedman 2009): DAG $\mathcal{G}$와 집합 $X, Y, Z$에 대해,

$$X \perp_{d, \mathcal{G}} Y \mid Z \iff X \perp_{m, \mathcal{M}(\mathcal{G}_{\text{An}(X \cup Y \cup Z)})} Y \mid Z$$

여기서 $\mathcal{G}_{\text{An}(X \cup Y \cup Z)}$는 $X \cup Y \cup Z$의 모든 ancestor만 포함한 subgraph, $\mathcal{M}$은 moralization, $\perp_m$은 MRF의 graph separation.

**증명 스케치**: 

**(⟹)** d-sep 판정의 세 패턴을 추적하면:
- Chain/Fork with $W \in Z$: moral graph에서 $W$를 제거하면 path 끊김
- Collider with $W \notin \text{An}(X \cup Y \cup Z)$: ancestral subgraph에서 $W$ 제거 → path 끊김
- Collider with $W \in \text{An}(X \cup Y \cup Z) \setminus Z$: 위험한 경우. 이 $W$의 parents가 moralization으로 연결되어 어디서 에서 path가 생김 — 이는 원래 d-sep되지 않은 경우와 대응

**(⟸)** 역방향: graph separation이면 d-sep. 구체적 path 변환으로 증명. $\square$

이 정리가 **d-sep 알고리즘의 핵심**. 실제 알고리즘은 먼저 ancestral subgraph를 만들고 moralize 후 BFS로 separation 판정 (복잡도 $O(|V| + |E|)$).

### 정리 5.4 — Minimal I-map의 Uniqueness (for BN)

**명제**: 고정된 topological order $\sigma$ 하에서, 분포 $P$의 **minimal I-map DAG**은 unique.

**증명**:

각 $i$에 대해, $\text{pa}_\sigma(v_i) := \{v_j : j < i, X_{v_i} \not\perp\!\!\!\perp X_{v_j} \mid X_{\{v_1, \ldots, v_{i-1}\} \setminus \{v_j\}}\}$로 정의. 즉 "$v_i$의 조건부 독립을 깨는 최소한의 조상".

이 $\text{pa}_\sigma$로 구성한 DAG이 unique minimal I-map (Koller & Friedman 2009 Theorem 3.3). 다른 parent 집합은 I-map이 아니거나 덜 minimal. $\square$

**주의**: Topological order가 바뀌면 minimal I-map도 바뀜. 전체 DAG의 Markov equivalence class 수준에서만 canonical.

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def moralize(dag):
    """DAG를 moralize한 undirected graph로 변환."""
    moral = nx.Graph()
    moral.add_nodes_from(dag.nodes())
    # 1. 방향 제거
    for u, v in dag.edges():
        moral.add_edge(u, v)
    # 2. 공통 자식의 parents 연결
    for v in dag.nodes():
        parents = list(dag.predecessors(v))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral.add_edge(parents[i], parents[j])
    return moral

# 예시 1: V-structure
dag1 = nx.DiGraph()
dag1.add_edges_from([('A', 'W'), ('B', 'W')])
moral1 = moralize(dag1)
print("V-structure moralization:")
print(f"  DAG edges: {list(dag1.edges())}")
print(f"  Moral edges: {list(moral1.edges())}")
# → A-B edge가 새로 생김

# 예시 2: Student Network
dag2 = nx.DiGraph()
dag2.add_edges_from([('D', 'G'), ('I', 'G'), ('I', 'S'), ('G', 'L')])
moral2 = moralize(dag2)
print(f"\nStudent Network moralization:")
print(f"  DAG edges: {sorted(dag2.edges())}")
print(f"  Moral edges: {sorted(moral2.edges())}")
# → D-I edge 추가 (G의 parents)

# 시각화
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# V-structure 원본
pos1 = {'A': (0, 1), 'W': (1, 0), 'B': (2, 1)}
nx.draw(dag1, pos1, with_labels=True, node_size=1500, node_color='lightblue',
        arrows=True, arrowsize=25, ax=axes[0])
axes[0].set_title('V-structure DAG')

# V-structure moralized
nx.draw(moral1, pos1, with_labels=True, node_size=1500, node_color='lightgreen', ax=axes[1])
axes[1].set_title('Moralized (A-B 추가!)')

# Student DAG
pos2 = {'D': (0, 1), 'I': (2, 1), 'G': (1, 0), 'S': (3, 0), 'L': (1, -1)}
nx.draw(dag2, pos2, with_labels=True, node_size=1500, node_color='lightblue',
        arrows=True, arrowsize=25, ax=axes[2])
axes[2].set_title('Student DAG')

# Student moralized
nx.draw(moral2, pos2, with_labels=True, node_size=1500, node_color='lightgreen', ax=axes[3])
axes[3].set_title('Student Moralized (D-I 추가!)')

plt.tight_layout()
plt.savefig('moralization_examples.png', dpi=120, bbox_inches='tight')
plt.show()

# CI 보존성 검증 (수치): v-structure에서 A ⊥ B가 moral에서 손실
# Toy distribution with v-structure: A, B ~ iid Bernoulli(0.5), W = A AND B
np.random.seed(0)
N = 100_000
A = np.random.randint(0, 2, N)
B = np.random.randint(0, 2, N)
W = A & B

# A ⊥ B? (original DAG: YES)
P_A = np.mean(A); P_B = np.mean(B); P_AB = np.mean(A * B)
print(f"\nA ⊥ B 검증: P(A=1, B=1) = {P_AB:.3f}, P(A=1)P(B=1) = {P_A*P_B:.3f} → 거의 동일 (독립)")

# A ⊥ B | W? (d-sep: NO; collider observed)
for w in [0, 1]:
    mask = W == w
    if mask.sum() < 100:
        continue
    P_A_given_W = np.mean(A[mask])
    P_B_given_W = np.mean(B[mask])
    P_AB_given_W = np.mean(A[mask] * B[mask])
    print(f"  W={w}: P(A=1,B=1|W) = {P_AB_given_W:.3f}, P(A=1|W)P(B=1|W) = {P_A_given_W*P_B_given_W:.3f}")

print("\n→ W=0에서는 A,B 모두 0 쪽으로 수렴 → 강한 dependency (explaining away)")
print("→ Moral graph에 A-B edge가 있어야 이 dependency 표현 가능")
```

**출력 예시**:
```
V-structure moralization:
  DAG edges: [('A', 'W'), ('B', 'W')]
  Moral edges: [('A', 'W'), ('A', 'B'), ('B', 'W')]

Student Network moralization:
  DAG edges: [('D', 'G'), ('G', 'L'), ('I', 'G'), ('I', 'S')]
  Moral edges: [('D', 'G'), ('D', 'I'), ('G', 'I'), ('G', 'L'), ('I', 'S')]

A ⊥ B 검증: P(A=1, B=1) = 0.251, P(A=1)P(B=1) = 0.250 → 거의 동일 (독립)
  W=0: P(A=1,B=1|W) = 0.000, P(A=1|W)P(B=1|W) = 0.111
  W=1: P(A=1,B=1|W) = 1.000, P(A=1|W)P(B=1|W) = 1.000

→ W=0에서는 A,B 모두 0 쪽으로 수렴 → 강한 dependency (explaining away)
→ Moral graph에 A-B edge가 있어야 이 dependency 표현 가능
```

V-structure에서 DAG와 moral graph의 차이가 명확: DAG에서만 표현되는 `A ⊥ B` (unconditional)가 moral graph에서 사라짐.

---

## 🔗 AI/ML 연결

### Junction Tree 알고리즘 (Ch2-04의 예고)

Exact inference on BN은 일반적으로 다음 절차:

1. BN을 moralize → MRF로 변환
2. Moral graph를 **triangulate** (chordal로 만듦) — fill-in edge 추가
3. Triangulated graph에서 **clique tree**(junction tree) 구성
4. Clique 간 message passing

이 과정에서 **treewidth**가 inference 복잡도를 결정 (Ch5-02). 따라서 moralization은 BN inference의 **첫 단계**.

### Conditional Random Field는 Moralize가 필요 없음

CRF(Ch4)는 이미 undirected (given input $x$, output $y$ is MRF). 따라서 moralization 없이 직접 message passing 가능. 이것이 CRF의 **inference 효율성** 중 하나.

### 혼합 모델 (Chain Graph)

BN과 MRF 모두를 포함하는 일반화: **chain graph** (Lauritzen & Wermuth 1989). 방향 edge와 undirected edge가 혼재. Moralization이 복잡하지만 여전히 정의 가능. 현대적으로 **deep generative + EBM**의 혼합 모델 분석에 응용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Positive density | Hammersley-Clifford 조건; 없으면 CI 판정 달라짐 |
| Ancestral subgraph 필요 | d-sep ↔ graph separation 동치는 ancestral restriction 후에만 |
| I-map 손실 | Moralization은 minimal I-map이 아닌 경우 많음 |
| BN-MRF 비호환성 | 일부 CI 구조는 어느 한쪽에만 표현 가능 |

**주의**: Moralization은 "**inference에 필요한 연결성을 보존**"하지만 "**CI 구조를 완전 보존**"하지는 않는다. 이 비대칭이 BN과 MRF의 본질적 차이.

---

## 📌 핵심 정리

$$\boxed{\mathcal{M}(\mathcal{G}) = \text{DAG의 edges} \cup \text{공통 자식을 가진 parent 쌍}}$$

| 개념 | 의미 |
|------|------|
| **Moralization** | 방향 제거 + parent pair 연결 |
| **I-map 보존** | DAG의 CI는 moral graph에서도 함의 |
| **CI 손실** | V-structure의 unconditional CI는 moral에서 표현 불가 |
| **Minimal I-map** | Topological order 고정 시 unique |
| **d-sep ⟺ m-sep** | Ancestral subgraph의 moral graph에서의 separation과 동치 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 DAG의 moral graph를 그려라.

```
A ──► B ──► D
│              ▲
│              │
└──► C ────────┘
```

edges: $A \to B, A \to C, B \to D, C \to D$.

<details>
<summary>힌트 및 해설</summary>

**원본 DAG의 edges**: $\{A-B, A-C, B-D, C-D\}$ (방향 제거 후).

**V-structure**: $D$의 parents = $\{B, C\}$. 따라서 $B-C$ edge 추가.

**Moral graph edges**: $\{A-B, A-C, B-C, B-D, C-D\}$.

결과: 4-완전그래프 $K_4$에서 $A-D$ edge만 빠진 형태 (즉 4-clique minus 1 edge).

흥미롭게, 이 moral graph는 **chordal** — 어떤 cycle도 길이 4 이상이 아님. 따라서 triangulation 불필요. Junction tree가 바로 구성 가능.

</details>

**문제 2** (심화): 4-cycle MRF $A - B - C - D - A$의 CI 구조 ($A \perp C \mid B, D$ and $B \perp D \mid A, C$)를 **어떤 DAG도 P-map으로 표현할 수 없다**를 보여라.

<details>
<summary>힌트 및 해설</summary>

4-cycle MRF의 두 CI:
(i) $A \perp C \mid B, D$  
(ii) $B \perp D \mid A, C$

이를 표현하는 DAG가 있다고 가정. 4-cycle이므로 edge 4개 또는 그 이상. 순환이 없는 DAG를 4개 변수로 만들려면 어떤 topological order를 택하든 **v-structure가 생김**.

구체적으로 $A \to B \to C \leftarrow D$ 같은 DAG을 생각하면:
- (ii) $B \perp D \mid A, C$? Trail $B \to C \leftarrow D$: collider $C$가 $\{A, C\}$에 있으므로 open. 그 다음 $C \leftarrow D$. path open → NOT d-sep. 즉 이 DAG는 $B \perp D \mid A, C$를 함의하지 **않음**. 따라서 D-map 실패.

다른 DAG 시도 — 모두 비슷하게 한 쪽 CI를 만족하면 다른 쪽을 어기거나, moral graph에 추가 edge로 인해 CI를 과소 표현.

**일반 사실**: 4-cycle의 symmetry (두 독립이 대칭적 역할)를 DAG가 재현하려면 두 chord 중 하나를 택해 v-structure로 전환해야 하는데, 이러면 대칭성이 깨짐. 따라서 어떤 DAG도 P-map 불가. $\square$

**결론**: MRF는 DAG로 표현 불가한 symmetric CI 구조가 있고, 반대로 DAG의 v-structure는 MRF로 표현 불가. **두 형식은 서로 enco할 수 없는 서로 다른 표현력**.

</details>

**문제 3** (AI 연결): Transformer의 causal self-attention은 DAG $\{1 \to 2, 1 \to 3, \ldots, 1 \to n, 2 \to 3, \ldots\}$ (fully ordered). 이 DAG의 moral graph는 무엇이고, 왜 이 그래프 구조가 "dense attention matrix"와 대응되는가?

<details>
<summary>힌트 및 해설</summary>

**Causal DAG의 구조**: 각 $i$에 대해 $\text{pa}(i) = \{1, 2, \ldots, i-1\}$ — 모든 이전 꼭짓점이 parent.

**Moralization**:
1. 모든 $(i, j)$ edge 방향 제거 ($i < j$): 이미 complete graph의 edge들을 포함 (모든 이전이 parent이므로).
2. Parents 쌍 연결: $\text{pa}(i) = \{1, \ldots, i-1\}$의 모든 쌍 — 이미 edge로 존재.

결과: **Complete undirected graph** $K_n$ (모든 pair가 연결).

**Attention matrix와 대응**:
- Causal attention mask: $A_{ij}$는 $j \leq i$일 때 비영(0이 아님). Lower triangular.
- Moral graph: undirected complete graph. Attention matrix로 보면 full matrix $A_{ij} \neq 0$ for all $i, j$.

차이: causal attention은 **방향 정보를 유지** (tril matrix), 반면 moral graph는 방향 정보를 잃음 (symmetric matrix).

**의미**: Transformer의 BN 표현력은 causal attention에서 나온다 — **v-structure의 전부 제거** (모든 parent 쌍이 연결되어 collider dependency가 항상 존재). 이것이 autoregressive language model이 **극도로 유연한 joint distribution**을 표현할 수 있는 이유 — 모든 complex CI pattern을 파라미터 (attention weight)로 학습.

**Dense vs sparse attention**: Sparse attention(Longformer, Reformer)은 이 complete graph에서 일부 edge를 제거 → 특정 CI 구조를 가정하는 restricted BN. 계산 효율 증가하지만 표현력 감소.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Markov Random Field와 Hammersley–Clifford](./04-markov-random-field.md) | [📚 README](../README.md) | [Ch2-01. Factor Graph의 정의와 통합 표현 ▶](../ch2-factor-graph/01-factor-graph-definition.md) |

</div>
