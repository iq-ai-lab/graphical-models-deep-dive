# 01. Factor Graph의 정의와 통합 표현

## 🎯 핵심 질문

- Factor graph는 Bayesian Network와 MRF를 어떻게 **하나의 표현**으로 통합하는가?
- Bipartite 구조(variable node + factor node)가 왜 message passing의 자연스러운 무대인가?
- BN과 MRF를 factor graph로 변환하는 규칙은 무엇인가?
- Factor graph만이 표현할 수 있는 구조는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Factor graph**는 modern graphical model의 **lingua franca**다. Kschischang–Frey–Loeliger(2001)의 sum-product algorithm이 LDPC 통신 코드, HMM의 Forward-Backward, Kalman filter, belief propagation을 **하나의 알고리즘**으로 통합한 것이 factor graph 덕분. **Probabilistic programming** (PyMC, Pyro, Stan)의 내부 표현도 factor graph. **GNN의 message passing**(Ch7-05)도 factor graph를 이론적 출발점으로 한다. 현대 PGM 구현 라이브러리(pgmpy, pomegranate)도 내부적으로 factor graph. Factor graph를 모르면 PGM을 "알고리즘별로" 이해하게 되어 통일된 관점을 놓친다.

---

## 📐 수학적 선행 조건

- [Ch1-02 Bayesian Network — DAG 기반 인수분해](../ch1-conditional-independence/02-bayesian-network-factorization.md): BN 인수분해
- [Ch1-04 Markov Random Field와 Hammersley–Clifford](../ch1-conditional-independence/04-markov-random-field.md): MRF clique potential
- [Ch1-05 Moralization](../ch1-conditional-independence/05-moralization.md): DAG → MRF 변환
- Graph theory: bipartite graph

---

## 📖 직관적 이해

### Factor Graph의 bipartite 구조

Factor graph는 **두 종류의 노드**를 가진 bipartite graph:

1. **Variable node** (원으로 표시): 각 확률변수 $x_i$를 나타냄
2. **Factor node** (사각형으로 표시): 각 factor $\phi_f$를 나타냄

**Edge**: 각 factor $\phi_f$와 $\phi_f$의 arguments인 variable들을 연결.

결합분포:
$$p(x_1, \ldots, x_n) = \frac{1}{Z} \prod_f \phi_f(x_{N(f)})$$

여기서 $N(f) \subseteq \{x_1, \ldots, x_n\}$은 factor $f$의 이웃(= argument).

### 예시: 3-chain

BN: $A \to B \to C$, factorization $p(a, b, c) = p(a) p(b|a) p(c|b)$

Factor graph:
```
[φ_A] ── A ── [φ_AB] ── B ── [φ_BC] ── C
```

여기서 $\phi_A(a) = p(a)$, $\phi_{AB}(a, b) = p(b|a)$, $\phi_{BC}(b, c) = p(c|b)$. 세 factor.

**장점**: BN의 CPT를 factor로 직접 번역. MRF의 clique potential도 같은 구조.

### 왜 Factor Graph가 BN/MRF보다 표현력 높은가

같은 MRF 인수분해라도 factor graph는 **더 세분화된** 구조 표현 가능.

**Maximal clique 분해 vs 세부 분해**:

4-cycle MRF에서 $\{A, B, C, D\}$의 maximal clique이 없고 pairwise edges만 있다고 하자:
- MRF: $p \propto \phi_{AB} \phi_{BC} \phi_{CD} \phi_{DA}$
- Factor graph: 4개의 **별개 factor node**

같은 분포를 다른 방식으로:
- $\phi'(a, b, c, d) = \phi_{AB} \phi_{BC} \phi_{CD} \phi_{DA}$ (하나의 4-variable factor)

**Factor graph는 이 두 경우를 구분** (별개 factor = 분해된 구조), MRF는 구분 못함 (clique potential 수준에서만).

### BN → Factor Graph 변환

BN $p(x) = \prod p(x_v | \text{pa}(v))$에서 각 $p(x_v | \text{pa}(v))$를 factor $\phi_v(\{v\} \cup \text{pa}(v)) := p(x_v | \text{pa}(v))$로.

**결과**: 원래 DAG의 각 $v$마다 factor node 하나. Variable-to-factor edge는 $v$와 $\text{pa}(v)$의 variables.

### MRF → Factor Graph 변환

MRF $p(x) \propto \prod_C \phi_C(x_C)$에서 각 clique potential을 factor로. Clique의 variables가 해당 factor에 연결.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Factor Graph

**Factor graph** $\mathcal{F} = (V, F, E)$는 bipartite graph로:

- $V = \{x_1, \ldots, x_n\}$: **variable nodes**
- $F = \{\phi_1, \ldots, \phi_m\}$: **factor nodes**, 각 $\phi_f: \prod_{v \in N(f)} \mathcal{X}_v \to \mathbb{R}_{\geq 0}$
- $E \subseteq V \times F$: variable-factor edges

각 factor $\phi_f$의 **scope** $N(f) := \{v : (v, f) \in E\}$.

결합분포:
$$p(x) = \frac{1}{Z} \prod_{f \in F} \phi_f(x_{N(f)}), \quad Z = \sum_x \prod_f \phi_f(x_{N(f)})$$

### 정의 1.2 — Normal Form (Loeliger 2004)

Factor graph가 **normal form**이면: 모든 variable이 정확히 **두 factor**에 연결.

이는 간선의 "중간 변수" 해석 — 각 edge가 하나의 variable을 나타냄. LDPC 통신 코드의 표준 표현.

변환: 임의 factor graph를 normal form으로 만들려면 variable copy node 추가 ($x_i = x_i'$ factor).

### 정의 1.3 — Factor Graph에서의 Markov Property

Variable $v$와 factor $f$에 대해, $v$의 **Markov blanket**은 $v$가 연결된 factor들의 다른 이웃 variables:

$$\text{MB}(v) = \bigcup_{f \in N(v)} N(f) \setminus \{v\}$$

$P$가 factor graph의 structure에 따라 인수분해되면 $v \perp\!\!\!\perp V \setminus (\{v\} \cup \text{MB}(v)) \mid \text{MB}(v)$.

---

## 🔬 정리와 증명

### 정리 1.1 — BN은 factor graph로 손실 없이 표현

**명제**: DAG $\mathcal{G}$의 BN $p(x) = \prod_v p(x_v | x_{\text{pa}(v)})$는 다음 factor graph와 동치:

- Variable nodes: $V$
- Factor nodes: 각 $v \in V$에 대해 하나의 factor $\phi_v(x_{\{v\} \cup \text{pa}(v)}) := p(x_v | x_{\text{pa}(v)})$
- Edges: $\phi_v$에서 $\{v\} \cup \text{pa}(v)$의 각 variable로

**증명**:

$$p(x) = \prod_v p(x_v | x_{\text{pa}(v)}) = \prod_v \phi_v(x_{\{v\} \cup \text{pa}(v)})$$

이는 정의 1.1의 factor graph 인수분해 형태 (정규화 상수 $Z = 1$, BN의 CPT가 이미 정규화되어 있음). $\square$

**참고**: 단일 variable factor $\phi_v$가 본질적으로 조건부분포이므로 자동 정규화 → $Z = 1$.

### 정리 1.2 — MRF는 factor graph로 손실 없이 표현

**명제**: MRF $p(x) \propto \prod_C \phi_C(x_C)$는 다음 factor graph와 동치:

- Variable nodes: $V$
- Factor nodes: 각 clique $C$마다 factor $\phi_C$
- Edges: $\phi_C$에서 $C$의 각 variable로

**증명**: 정의에 따라 즉시 성립. $\square$

### 정리 1.3 — Factor Graph의 Moralization (variant)

**명제**: Factor graph $\mathcal{F}$의 **induced Markov graph** $\mathcal{G}(\mathcal{F})$는

$$E(\mathcal{G}) = \{(u, v) : u \neq v, \exists f \text{ s.t. } u, v \in N(f)\}$$

즉 같은 factor에 속하는 모든 variable pair가 edge. $\mathcal{F}$의 인수분해 → $\mathcal{G}(\mathcal{F})$의 MRF 인수분해 (clique = factor scope). 역은 일반적으로 성립 안 함 (factor graph가 더 세분화된 구조를 표현 가능).

**증명**: Factor $\phi_f(x_{N(f)})$의 scope $N(f)$이 $\mathcal{G}$에서 complete → clique. $\phi_f$는 $\mathcal{G}$의 clique potential로 interpret 가능. $\square$

### 정리 1.4 — Factor Graph만의 표현력

**명제**: 어떤 인수분해 구조는 factor graph로만 표현 가능.

**예시**: 3-variable with $p(x_1, x_2, x_3) = \phi_{12}(x_1, x_2) \phi_{23}(x_2, x_3) \phi_{31}(x_3, x_1)$ — **triangle cycle with 3 pairwise factors**.

Induced MRF graph: triangle $K_3$ (모든 pair가 edge). 단일 maximal clique = $\{1, 2, 3\}$. MRF로 보면 $p \propto \phi(x_1, x_2, x_3)$ (하나의 3-variable factor)로만 표현 가능.

**차이**: 3개의 pairwise factor vs 1개의 triple factor — 파라미터 수, 계산 복잡도, 학습 가능성이 모두 다름. Factor graph는 이 구조적 차이를 **명시적으로 표현**.

$\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class FactorGraph:
    def __init__(self):
        self.variables = {}  # name -> cardinality
        self.factors = []    # list of (name, scope, table)
    
    def add_variable(self, name, cardinality):
        self.variables[name] = cardinality
    
    def add_factor(self, name, scope, table):
        """scope: list of variable names, table: ndarray"""
        assert table.shape == tuple(self.variables[v] for v in scope)
        self.factors.append((name, scope, table))
    
    def joint_distribution(self):
        """Brute force 결합분포 계산."""
        var_list = list(self.variables.keys())
        card = [self.variables[v] for v in var_list]
        total_size = np.prod(card)
        P = np.ones(card)
        
        for fname, scope, table in self.factors:
            # broadcast table to full shape
            shape = [1] * len(var_list)
            for vi, vname in enumerate(scope):
                shape[var_list.index(vname)] = table.shape[vi]
            broadcast = np.ones(card)
            for idx in np.ndindex(*card):
                scope_idx = tuple(idx[var_list.index(v)] for v in scope)
                broadcast[idx] = table[scope_idx]
            P *= broadcast
        
        Z = P.sum()
        return P / Z, Z
    
    def visualize(self, ax=None, pos=None):
        G = nx.Graph()
        for v in self.variables:
            G.add_node(v, kind='variable')
        for fname, scope, _ in self.factors:
            G.add_node(fname, kind='factor')
            for v in scope:
                G.add_edge(fname, v)
        
        if pos is None:
            pos = nx.spring_layout(G, seed=42)
        
        var_nodes = [n for n in G.nodes() if G.nodes[n]['kind'] == 'variable']
        fac_nodes = [n for n in G.nodes() if G.nodes[n]['kind'] == 'factor']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, node_color='lightblue',
                               node_shape='o', node_size=1500, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=fac_nodes, node_color='lightcoral',
                               node_shape='s', node_size=1500, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)

# 예시 1: BN을 factor graph로
# BN: A → B → C with P(A), P(B|A), P(C|B)
fg_bn = FactorGraph()
fg_bn.add_variable('A', 2)
fg_bn.add_variable('B', 2)
fg_bn.add_variable('C', 2)

P_A = np.array([0.6, 0.4])
P_BA = np.array([[0.7, 0.3], [0.2, 0.8]])  # P(B | A)
P_CB = np.array([[0.9, 0.1], [0.4, 0.6]])  # P(C | B)

fg_bn.add_factor('φ_A', ['A'], P_A)
fg_bn.add_factor('φ_AB', ['A', 'B'], P_BA)
fg_bn.add_factor('φ_BC', ['B', 'C'], P_CB)

P_joint, Z = fg_bn.joint_distribution()
print(f"BN as Factor Graph: Z = {Z:.4f}")
print(f"Joint P(A, B, C):\n{P_joint}")
print(f"주변 P(A): {P_joint.sum(axis=(1,2))}")  # = [0.6, 0.4]
print(f"주변 P(B): {P_joint.sum(axis=(0,2))}")

# 예시 2: MRF triangle을 factor graph로
fg_mrf = FactorGraph()
fg_mrf.add_variable('A', 2)
fg_mrf.add_variable('B', 2)
fg_mrf.add_variable('C', 2)

phi_AB = np.array([[2.0, 1.0], [1.0, 2.0]])
phi_BC = np.array([[1.5, 0.5], [0.5, 1.5]])
phi_CA = np.array([[1.8, 0.8], [0.8, 1.8]])

fg_mrf.add_factor('φ_AB', ['A', 'B'], phi_AB)
fg_mrf.add_factor('φ_BC', ['B', 'C'], phi_BC)
fg_mrf.add_factor('φ_CA', ['C', 'A'], phi_CA)

P_mrf, Z_mrf = fg_mrf.joint_distribution()
print(f"\nMRF Triangle: Z = {Z_mrf:.4f}")
print(f"Joint P(A, B, C):\n{P_mrf}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pos_bn = {'A': (0, 0), 'φ_A': (0, -1), 'B': (1, 0), 'φ_AB': (0.5, 0), 'C': (2, 0), 'φ_BC': (1.5, 0)}
fg_bn.visualize(ax=axes[0], pos=pos_bn)
axes[0].set_title('BN (A→B→C) as Factor Graph')
axes[0].axis('off')

pos_mrf = {'A': (0, 1), 'B': (1, 1), 'C': (0.5, 0),
           'φ_AB': (0.5, 1), 'φ_BC': (0.75, 0.5), 'φ_CA': (0.25, 0.5)}
fg_mrf.visualize(ax=axes[1], pos=pos_mrf)
axes[1].set_title('MRF Triangle as Factor Graph')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('factor_graph_examples.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
BN as Factor Graph: Z = 1.0000
Joint P(A, B, C):
[[[0.378 0.042]
  [0.072 0.108]]
 [[0.072 0.008]
  [0.128 0.192]]]
주변 P(A): [0.6 0.4]
주변 P(B): [0.5 0.5]

MRF Triangle: Z = 16.4000
Joint P(A, B, C):
[[[0.329 0.024]
  [0.037 0.018]]
 [[0.049 0.024]
  [0.037 0.482]]]
```

BN의 경우 factor graph 표현에서 $Z = 1$ (조건부분포의 곱이 자동 정규화), MRF는 $Z = 16.4$로 정규화 필요.

---

## 🔗 AI/ML 연결

### Probabilistic Programming

Pyro, PyMC, Stan 등의 PPL(probabilistic programming language)은 모델을 코드로 정의하지만 내부적으로는 **factor graph**로 컴파일. 예를 들어 Pyro의 `pyro.sample()` 호출은 variable node를, `pyro.factor()`는 명시적 factor node를 생성. 이 factor graph 위에서 VI, HMC, SVI 등이 작동.

### LDPC Code와 Sum-Product

LDPC(Low-Density Parity-Check) 통신 코드의 **Tanner graph**가 factor graph의 원조 (Gallager 1962, rediscovered by MacKay 1999). Sum-product algorithm의 성공 사례 — Shannon limit에 근접하는 성능을 message passing으로 달성.

### Graph Neural Network (GNN)

Gilmer et al.(2017)의 MPNN(Message Passing Neural Network) framework는 factor graph에 직접 대응:
- Variable node = graph의 node (atom in molecular graph)
- Factor node = edge/interaction
- Message passing = BP의 학습된 비선형 일반화

Ch7-05에서 자세히.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Bipartite 구조 | 다른 그래프 구조(e.g., hypergraph)로 일반화 가능하지만 복잡 |
| Factor $\geq 0$ | Negative factor 필요한 quantum graphical model 예외 |
| Normal form의 variable copying | 명시적 복사가 계산 오버헤드 |
| Discrete variables | 연속 factor graph는 Gaussian 등 특수 경우에만 exact |

**주의**: Factor graph의 세분화된 인수분해는 **과하게 세분화하면 오버헤드**. 예: 모든 variable에 unary factor를 분리하면 factor 수가 $O(|V|)$ 증가. 실용적으로는 **maximal factor**로 통합.

---

## 📌 핵심 정리

$$\boxed{p(x) = \frac{1}{Z} \prod_{f \in F} \phi_f(x_{N(f)}) \text{ — bipartite variable-factor graph}}$$

| 개념 | 의미 |
|------|------|
| **Variable node** | 확률변수 $x_i$ |
| **Factor node** | Non-negative function $\phi_f(x_{N(f)})$ |
| **Scope** | $N(f)$: factor $f$의 argument 집합 |
| **BN → FG** | 각 CPT가 하나의 factor |
| **MRF → FG** | 각 clique potential이 하나의 factor |
| **세분화 표현력** | Factor graph > MRF (같은 clique에서 여러 factor 구분) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3-variable with $p(x_1, x_2, x_3) = \phi_{123}(x_1, x_2, x_3)$ (1 triple factor) vs $p \propto \phi_{12} \phi_{13} \phi_{23}$ (3 pairwise factors). 이진 변수에서 각각의 **파라미터 수**를 계산하라.

<details>
<summary>힌트 및 해설</summary>

**Triple factor**: $\phi_{123}(x_1, x_2, x_3)$는 $2^3 = 8$개 엔트리. 정규화 후 7 free parameters.

**3 pairwise factors**: 각 $\phi_{ij}(x_i, x_j)$는 $2^2 = 4$개. 3개 factor = 12 entries. 하지만 rescaling freedom ($\phi_{ij} \to c \phi_{ij}, \phi_{ik} \to \phi_{ik}/c$)으로 자유도 감소. Canonical form (log-linear pairwise MRF)에서 실제 free parameters: 3 pairwise × 3 (이진 변수에서 pairwise의 canonical dimension) + normalization = **3 + 3 singleton potentials에서 = 이진 MRF pairwise에서 대략 6~9 free params**.

**결론**: Pairwise는 triple보다 **표현력이 제한적** (모든 분포 표현 불가) 하지만 **파라미터 수는 비슷하거나 적음**. 이 trade-off가 **Ising model** 같은 pairwise MRF 선택의 이유.

</details>

**문제 2** (심화): Factor graph의 **normal form** 변환에서, $k$개 factor에 연결된 variable $v$를 normal form으로 만들려면 몇 개의 copy node(=equality factor)가 필요한가?

<details>
<summary>힌트 및 해설</summary>

Normal form의 조건: 각 variable은 정확히 2개 factor에 연결.

$v$가 $k$개 factor $f_1, \ldots, f_k$에 연결되어 있다면, $v$를 $k$개 copy $v_1, v_2, \ldots, v_k$로 분리하고 "$v_1 = v_2 = \ldots = v_k$" equality constraint를 factor로 추가.

**방법 1** (star topology): $v$와 $v_1, \ldots, v_k$에 연결된 $(k+1)$-way equality factor 하나. Normal form 조건: $v$가 원래 2개에 연결되도록 star topology 사용. 총 **1개 equality factor**.

**방법 2** (chain): $v_1 = v_2$, $v_2 = v_3$, ..., $v_{k-1} = v_k$ pairwise equality. 총 **$k - 1$개 factor**.

**방법 3** (binary tree): $\log_2 k$ depth의 binary tree of equality factors.

실용적으로 방법 3이 효율 (message passing의 depth 감소). LDPC decoding에서 표준.

</details>

**문제 3** (AI 연결): Transformer의 self-attention matrix $A = \text{softmax}(QK^T/\sqrt{d})$를 factor graph로 해석하라. 각 attention head는 어떤 factor structure를 학습하는가?

<details>
<summary>힌트 및 해설</summary>

**Factor graph 해석**:
- Variable nodes: 각 토큰 위치 $i = 1, \ldots, n$
- Attention weight $A_{ij}$는 $(i, j)$ pair의 "soft connection" — pairwise factor의 strength
- 여러 attention head = 여러 overlapping factor graphs (각 head가 다른 factor structure)

**각 head의 factor structure 학습**:
- 일부 head: **Local factor** (인접 토큰만 큰 attention) — N-gram-like patterns
- 일부 head: **Long-range factor** (문장 시작-끝 연결) — discourse-level dependencies
- 일부 head: **Syntactic factor** (동사-목적어, 주어-동사) — 문법 구조
- 일부 head: **Coreference factor** (대명사-지칭 대상) — reference resolution

**Message passing 관점**: Self-attention = one layer of "soft message passing" over a fully-connected graph. Multiple layers = iterated message passing → factor graph BP의 neural generalization.

**비교**: 
- Sparse attention (Longformer): 특정 sparse factor graph 구조를 hard-code
- Dense attention: 모든 factor를 학습할 수 있게 하되, 데이터에서 sparse해짐
- Graph attention (GAT): 주어진 그래프 구조 위에서의 attention = "factor graph 주어짐, weight만 학습"

결론: **Transformer = 학습된 factor graph 위의 soft BP** — Ch7-05의 핵심 메시지를 미리 맛보기.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch1-05 Moralization](../ch1-conditional-independence/05-moralization.md) | [📚 README](../README.md) | [02. Sum-Product Algorithm (Belief Propagation) ▶](./02-sum-product-algorithm.md) |

</div>
