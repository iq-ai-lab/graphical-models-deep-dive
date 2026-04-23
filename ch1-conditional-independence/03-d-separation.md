# 03. d-separation — 세 경로 패턴의 대수

## 🎯 핵심 질문

- DAG의 세 경로 패턴 — Chain, Fork, Collider — 은 왜 조건부 독립에 서로 다르게 반응하는가?
- d-separation은 정확히 무엇이고, 왜 그래프 구조만으로 CI를 판정할 수 있는가?
- **Soundness**(d-sep ⟹ CI)와 **Completeness**(CI ⟹ d-sep)는 어떻게 증명되는가?
- **Explaining away**는 왜 collider의 대수적 필연성인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**d-separation**은 BN에서 CI를 **그래프 알고리즘으로 판정**할 수 있게 해주는 도구다. Pearl의 **do-calculus**(인과 추론의 세 규칙)는 모두 d-separation에 기반한다. **Front-door criterion**, **back-door criterion** 같은 인과 식별 조건도 d-sep의 응용이다. Variational Inference에서 **어떤 latent variable이 서로 CI인지** 알아야 variational family를 올바르게 선택할 수 있다. **Causal Discovery**(PC algorithm, FCI)는 d-sep을 역으로 사용해 관측 데이터로부터 DAG를 복원한다. d-sep을 모르면 **Simpson paradox**, **collider bias**, **selection bias** 같은 인과 추론의 함정을 구분할 수 없다.

---

## 📐 수학적 선행 조건

- [Ch1-01 조건부 독립의 정의와 성질](./01-conditional-independence-definition.md): CI의 정의와 semi-graphoid 공리
- [Ch1-02 Bayesian Network — DAG 기반 인수분해](./02-bayesian-network-factorization.md): BN factorization, local Markov property
- Graph theory: path, cycle, ancestor/descendant

---

## 📖 직관적 이해

### 세 가지 경로 패턴

DAG에서 임의의 **undirected path**(화살표 방향 무시)를 따라가면, 각 내부 꼭짓점 $W$는 다음 중 하나:

**1. Chain (연쇄)**: $\cdots \to W \to \cdots$ 또는 $\cdots \leftarrow W \leftarrow \cdots$

$$X \to W \to Y: \quad X \perp\!\!\!\perp Y? \text{ No}; \quad X \perp\!\!\!\perp Y \mid W? \text{ Yes} \quad (W \text{ blocks})$$

**2. Fork (분기)**: $\cdots \leftarrow W \to \cdots$

$$X \leftarrow W \to Y: \quad X \perp\!\!\!\perp Y? \text{ No}; \quad X \perp\!\!\!\perp Y \mid W? \text{ Yes} \quad (W \text{ blocks})$$

**3. Collider (v-structure)**: $\cdots \to W \leftarrow \cdots$

$$X \to W \leftarrow Y: \quad X \perp\!\!\!\perp Y? \text{ Yes}; \quad X \perp\!\!\!\perp Y \mid W? \text{ No} \quad (W \text{ opens!})$$

요약:

| 패턴 | 그림 | $W \notin Z$ (observed) | $W \in Z$ (또는 후손) |
|------|------|------------------------|----------------------|
| Chain | $X \to W \to Y$ | **open** (경로 열림) | **blocked** |
| Fork | $X \leftarrow W \to Y$ | **open** | **blocked** |
| Collider | $X \to W \leftarrow Y$ | **blocked** | **open!** |

Collider가 반대인 것이 핵심 반전. 이것이 **explaining away**의 출처.

### Blocked Path의 정의

조건 집합 $Z$가 주어졌을 때, path는 **blocked**이다 (by $Z$):

- Chain/Fork의 중간 $W$가 $Z$에 있거나
- Collider의 $W$ 또는 그 후손 중 **아무도** $Z$에 없다

path의 어딘가가 막히면 그 path는 정보를 전달 못함. **모든 path가 막히면** $X$와 $Y$가 $Z$ 주어졌을 때 **d-separated**.

### Explaining Away (Collider)

"천재이거나 부모 인맥이 있으면 명문대 합격" 예시:

```
   Intelligence (X) ─► Admitted (W) ◄─ Connections (Y)
```

- **주변**: $P(X, Y) = P(X) P(Y)$ — 지능과 인맥은 무관
- **조건부 (W 관측)**: "합격한 학생"을 본다고 할 때 지능 낮으면 인맥 높을 가능성↑

즉 관측된 collider가 **양의 상관을 음의 상관으로 만듦** — 이것이 **selection bias**의 수학적 기원.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Trail (Undirected Path)

DAG $\mathcal{G}$에서 **trail** $\tau$는 꼭짓점의 수열 $v_0, v_1, \ldots, v_k$로, 각 $(v_i, v_{i+1})$이 **방향에 무관하게** 간선.

### 정의 3.2 — Blocked Trail

Trail $\tau = v_0 \cdots v_k$가 조건 집합 $Z \subseteq V$에 의해 **blocked**이다 iff 다음 중 하나의 내부 꼭짓점 $v_i$가 존재:

(a) $v_i$가 chain 또는 fork이고 $v_i \in Z$  
  $(v_{i-1} \to v_i \to v_{i+1}, v_{i-1} \leftarrow v_i \leftarrow v_{i+1}, \text{또는 } v_{i-1} \leftarrow v_i \to v_{i+1})$

(b) $v_i$가 collider이고 $v_i \notin Z$이며 $v_i$의 **모든 후손 $\text{de}(v_i) \cup \{v_i\}$이 $Z$와 분리**
  $(v_{i-1} \to v_i \leftarrow v_{i+1})$

blocked이 아니면 **active**(또는 unblocked).

### 정의 3.3 — d-separation

집합 $X, Y, Z \subseteq V$ (서로 소)에 대해, **$X$와 $Y$가 $Z$에 의해 d-separated**이다 (표기: $X \perp_{d,\mathcal{G}} Y \mid Z$) iff

모든 trail $\tau$ (한 끝점 $X$, 다른 끝점 $Y$)가 $Z$에 의해 blocked.

---

## 🔬 정리와 증명

### 정리 3.1 — Soundness (Verma–Pearl 1988)

**명제**: 분포 $P$가 DAG $\mathcal{G}$의 factorization $p(x) = \prod p(x_v \mid x_{\text{pa}(v)})$를 만족하면,

$$X \perp_{d,\mathcal{G}} Y \mid Z \implies X \perp\!\!\!\perp Y \mid Z \text{ under } P$$

**증명 스케치**:

Global Markov property를 먼저 증명하는 방법이 표준. Global Markov: "$X$와 $Y$가 $Z$에 대해 $\mathcal{G}$에서 d-sep되면 CI". 이는 local Markov와 동치임을 보이고(정리 2.2), 다음을 귀납적으로 확장:

**Step 1** (Moralization으로 undirected 근사): **Ancestral set** $\text{An}(X \cup Y \cup Z) \cup X \cup Y \cup Z$의 subgraph을 취한 후, 이 DAG에서의 moralization(Ch1-05)을 만들어 MRF로 변환. 이 moral graph에서의 **separation**(graph separation)이 원래 DAG의 d-separation과 동치.

**Step 2** (Hammersley–Clifford에 의한 MRF 성질): Moral graph에서 graph-separated ⟹ CI (MRF의 global Markov, Ch1-04에서 증명).

**Step 3** (합성): 위 두 단계로 d-sep ⟹ CI. $\square$

전체 상세 증명은 Lauritzen(1996) "Graphical Models" Theorem 3.27 참조.

### 정리 3.2 — Completeness (Meek 1995)

**명제**: DAG $\mathcal{G}$에 대해, **거의 모든** 분포 $P$ (factorization을 만족하는)에서

$$X \perp\!\!\!\perp Y \mid Z \text{ under } P \implies X \perp_{d,\mathcal{G}} Y \mid Z$$

엄밀히: d-sep되지 않은 $(X, Y, Z)$에 대해 "CI임을 만족하는 $P$"의 집합은 Lebesgue measure 0.

**증명 스케치**: 

파라미터 공간 (각 CPT의 값들) 위에서, CI는 다항식 방정식 $P(X, Y | Z) - P(X|Z)P(Y|Z) = 0$을 만족해야 함. 이 방정식이 자명히 성립하지 않으면 다항식의 zero set이 되어 measure zero.

즉, "**faithfulness**" 가정(분포가 DAG가 함의하는 CI 외 다른 CI를 갖지 않음)은 generic에서 성립. $\square$

### 정리 3.3 — 세 패턴의 대수적 증명

Chain, Fork, Collider 각각에서 CI 구조를 직접 계산.

**Chain $X \to W \to Y$**: $p(x, w, y) = p(x) p(w|x) p(y|w)$
- $X \perp\!\!\!\perp Y$? $p(x, y) = \sum_w p(x) p(w|x) p(y|w) = p(x) \sum_w p(w|x) p(y|w) = p(x) p(y|x)$. 일반적으로 $\neq p(x) p(y)$.
- $X \perp\!\!\!\perp Y | W$? $p(x, y | w) = \frac{p(x) p(w|x) p(y|w)}{p(w)} = \frac{p(x) p(w|x)}{p(w)} \cdot p(y|w) = p(x|w) p(y|w)$. **Yes**. $\square$

**Fork $X \leftarrow W \to Y$**: $p(x, w, y) = p(w) p(x|w) p(y|w)$
- $X \perp\!\!\!\perp Y$? 일반적으로 아님.
- $X \perp\!\!\!\perp Y | W$? $p(x, y | w) = \frac{p(w) p(x|w) p(y|w)}{p(w)} = p(x|w) p(y|w)$. **Yes**. $\square$

**Collider $X \to W \leftarrow Y$**: $p(x, w, y) = p(x) p(y) p(w|x, y)$
- $X \perp\!\!\!\perp Y$? $p(x, y) = \sum_w p(x) p(y) p(w|x,y) = p(x) p(y) \sum_w p(w|x,y) = p(x) p(y)$. **Yes** (주변). $\square$
- $X \perp\!\!\!\perp Y | W$? $p(x, y | w) = \frac{p(x) p(y) p(w|x, y)}{p(w)}$. 일반적으로 $p(x|w) p(y|w)$와 같지 않음. 구체적 반례: $X, Y \in \{0, 1\}$ iid Bernoulli(0.5), $W = X \oplus Y$. 그러면 $p(W=0) = 1/2$. $p(X=0 | W=0) = 1/2$, $p(Y=0 | W=0) = 1/2$, $p(X=Y=0 | W=0) = 1/2 \neq 1/4$. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 간단한 d-separation 판정기 구현
def find_active_trails(G, source, target, Z):
    """BFS로 active trail 존재 여부 판정 (Koller & Friedman 알고리즘 3.1)."""
    # 각 상태: (node, direction_from_prev) — "up"은 자식→부모, "down"은 부모→자식
    # 모든 ancestor of Z를 먼저 계산 (collider activation에 필요)
    anc_Z = set(Z)
    stack = list(Z)
    while stack:
        n = stack.pop()
        for p in G.predecessors(n):
            if p not in anc_Z:
                anc_Z.add(p)
                stack.append(p)
    
    # BFS
    visited = set()
    queue = [(source, 'up')]  # source에서 시작, 방향은 "어디서 왔는지"의 반대
    visited.add((source, 'up'))
    visited.add((source, 'down'))
    
    while queue:
        new_queue = []
        for node, direction in queue:
            if node == target:
                return True  # active trail 존재
            
            if direction == 'up':
                # 자식 → 부모로 들어옴. 부모(up)로 갈 수 있고, 자식(down)으로도 갈 수 있음 (fork)
                if node not in Z:
                    for p in G.predecessors(node):
                        if (p, 'up') not in visited:
                            visited.add((p, 'up'))
                            new_queue.append((p, 'up'))
                    for c in G.successors(node):
                        if (c, 'down') not in visited:
                            visited.add((c, 'down'))
                            new_queue.append((c, 'down'))
            else:  # direction == 'down'
                # 부모 → 자식으로 들어옴. chain은 node ∉ Z일 때만 진행
                if node not in Z:
                    for c in G.successors(node):
                        if (c, 'down') not in visited:
                            visited.add((c, 'down'))
                            new_queue.append((c, 'down'))
                # collider: node ∈ an(Z) ∪ Z 이면 부모로 올라갈 수 있음
                if node in anc_Z:
                    for p in G.predecessors(node):
                        if (p, 'up') not in visited:
                            visited.add((p, 'up'))
                            new_queue.append((p, 'up'))
        queue = new_queue
    
    return False

def d_separated(G, X, Y, Z):
    """X ⊥_d Y | Z 판정."""
    for x in X:
        for y in Y:
            if find_active_trails(G, x, y, set(Z)):
                return False
    return True

# 테스트 DAG: Student network
G = nx.DiGraph()
G.add_edges_from([('D', 'G'), ('I', 'G'), ('I', 'S'), ('G', 'L')])

queries = [
    ({'D'}, {'S'}, set()),           # True (disjoint components)
    ({'D'}, {'S'}, {'G'}),           # False? D → G ← I → S, G is collider, observed → opens
    ({'D'}, {'I'}, set()),           # True (no trail)
    ({'D'}, {'I'}, {'G'}),           # False (collider G observed)
    ({'L'}, {'S'}, set()),           # False (L ← G ← I → S)
    ({'L'}, {'S'}, {'G'}),           # True (G blocks chain)
    ({'L'}, {'S'}, {'I'}),           # False (I blocks fork but G ← I → S path is blocked; 
                                     #  but L ← G ← I → S: I blocks → YES d-sep)
]

for X, Y, Z in queries:
    sep = d_separated(G, X, Y, Z)
    print(f"{X} ⊥_d {Y} | {Z} → {sep}")

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
pos = {'D': (0, 1), 'I': (2, 1), 'G': (1, 0), 'S': (3, 0), 'L': (1, -1)}
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue',
        arrows=True, arrowsize=25, ax=ax, font_size=14, font_weight='bold')
ax.set_title('Student BN — d-separation 판정 대상')
plt.tight_layout()
plt.savefig('d_sep_student.png', dpi=120, bbox_inches='tight')
plt.show()

# 수치 시뮬레이션으로 collider의 explaining away 확인
np.random.seed(0)
N = 100_000
D = np.random.randint(0, 2, N)
I = np.random.randint(0, 2, N)
G_val = (D + I) % 2  # Collider: G = D XOR I

# 주변: D ⊥ I? YES
print(f"\nD ⊥ I 검증: P(D=0,I=0)={np.mean((D==0)&(I==0)):.3f}, "
      f"P(D=0)P(I=0)={np.mean(D==0) * np.mean(I==0):.3f}")

# 조건부: D ⊥ I | G? NO (explaining away)
for g in [0, 1]:
    mask = G_val == g
    p_D0 = np.mean(D[mask] == 0)
    p_I0 = np.mean(I[mask] == 0)
    p_D0_I0 = np.mean((D[mask] == 0) & (I[mask] == 0))
    print(f"  G={g}: P(D=0,I=0|G)={p_D0_I0:.3f}, P(D=0|G)P(I=0|G)={p_D0 * p_I0:.3f} ← 차이가 큼!")
```

**출력 예시**:
```
{'D'} ⊥_d {'S'} | set() → True
{'D'} ⊥_d {'S'} | {'G'} → False
{'D'} ⊥_d {'I'} | set() → True
{'D'} ⊥_d {'I'} | {'G'} → False
{'L'} ⊥_d {'S'} | set() → False
{'L'} ⊥_d {'S'} | {'G'} → True
{'L'} ⊥_d {'S'} | {'I'} → True

D ⊥ I 검증: P(D=0,I=0)=0.250, P(D=0)P(I=0)=0.250
  G=0: P(D=0,I=0|G)=0.499, P(D=0|G)P(I=0|G)=0.250 ← 차이가 큼!
  G=1: P(D=0,I=0|G)=0.000, P(D=0|G)P(I=0|G)=0.250 ← 차이가 큼!
```

수치적으로 collider를 관측하면 원래 독립인 두 변수가 강한 상관을 보이게 됨이 확인된다.

---

## 🔗 AI/ML 연결

### Causal Discovery와 PC Algorithm

PC algorithm (Spirtes–Glymour–Scheines 2000)은 d-separation을 **역으로** 사용:
1. 관측 데이터에서 모든 쌍의 조건부 독립을 검정
2. CI 집합과 일치하는 DAG를 찾음
3. Markov equivalence class까지 복원 (개별 DAG는 일반적으로 identifiable하지 않음)

핵심 가정: **Faithfulness** — 관측된 CI가 모두 d-sep로 설명 가능.

### Do-Calculus와 인과 추론

Pearl의 **do-calculus 세 규칙**은 모두 d-sep에 기반:

**Rule 1** (Insertion/Deletion of observations): $P(y \mid \text{do}(x), z, w) = P(y \mid \text{do}(x), w)$ if $Y \perp_{d, \mathcal{G}_{\overline{X}}} Z \mid X, W$

**Rule 2** (Action/Observation exchange): $P(y \mid \text{do}(x), \text{do}(z), w) = P(y \mid \text{do}(x), z, w)$ if $Y \perp_{d, \mathcal{G}_{\overline{X}, \underline{Z}}} Z \mid X, W$

**Rule 3** (Insertion/Deletion of actions): $P(y \mid \text{do}(x), \text{do}(z), w) = P(y \mid \text{do}(x), w)$ if $Y \perp_{d, \mathcal{G}_{\overline{X}, \overline{Z(W)}}} Z \mid X, W$

여기서 $\mathcal{G}_{\overline{X}}$는 $X$로 들어오는 edge 제거, $\underline{Z}$는 $Z$에서 나가는 edge 제거한 modified DAG. 이 규칙들의 복잡한 조건들이 모두 **d-sep의 변형**.

### Collider Bias와 Selection Bias

Berkson paradox: 병원에 입원한 환자만 표본으로 쓰면, 입원 = collider의 실현으로 작용하여 원래 독립인 두 질병이 음의 상관을 보이게 됨. 이는 ML의 **biased sampling**, **survivor bias**, **causal effects from observational data**의 문제점과 직결.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| DAG 주어짐 | 실제로는 데이터에서 학습(Ch7-03), 일반적으로 Markov equivalence class까지만 |
| Faithfulness | 실제 분포가 정확히 DAG의 CI에 따르지 않을 수 있음 (deterministic relations) |
| Causal sufficiency | 모든 common cause가 관측됨 — 숨은 변수 있으면 FCI/MAG 필요 |
| Completeness는 generic | 측도 0인 예외에서는 parameter-specific CI 가능 |

**주의**: d-separation은 **그래프 구조만으로** 판정 — 실제 CI는 분포의 파라미터에 의존할 수 있으나, **generic하게** 그래프가 결정한다는 것이 정리 3.2의 의미.

---

## 📌 핵심 정리

$$\boxed{\text{Chain/Fork: } W \text{ blocks iff } W \in Z; \quad \text{Collider: } W \text{ opens iff } W \text{ 또는 후손} \in Z}$$

| 패턴 | $W \notin Z$ | $W \in Z$ |
|------|------------|----------|
| Chain $X \to W \to Y$ | open | **blocked** |
| Fork $X \leftarrow W \to Y$ | open | **blocked** |
| Collider $X \to W \leftarrow Y$ | **blocked** | open (explaining away) |

**d-separation**: 모든 undirected path가 $Z$에 의해 blocked ⟺ $X \perp_{d, \mathcal{G}} Y \mid Z$.

**Soundness & Completeness**: $X \perp_{d, \mathcal{G}} Y \mid Z \iff X \perp\!\!\!\perp Y \mid Z$ (generic 분포에서).

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 DAG에서 d-separation을 판정하라.

```
A ──► B ──► D
       ▲
       C
       │
A ─────┘
```

즉 edges: $A \to B, A \to C, C \to B, B \to D$.

(a) $A \perp_d D \mid \emptyset$?
(b) $A \perp_d D \mid B$?
(c) $A \perp_d D \mid C$?

<details>
<summary>힌트 및 해설</summary>

Trail $A \to B \to D$와 $A \to C \to B \to D$가 있음.

(a) $Z = \emptyset$: $A \to B \to D$ — chain $B$가 open → active trail. 따라서 **NOT d-sep**.

(b) $Z = \{B\}$: 
- Trail 1 ($A \to B \to D$): $B$가 chain의 middle, $B \in Z$ → blocked.
- Trail 2 ($A \to C \to B \to D$): $C$가 chain의 middle, $C \notin Z$ → 통과. $B$는 (에서의 $C \to B \leftarrow A$?) 아니, 이 trail의 구조는 $A \to C \to B \to D$. $B$는 **chain**의 middle ($C \to B \to D$), $B \in Z$ → blocked.
- 모든 trail blocked → **d-sep**.

(c) $Z = \{C\}$:
- Trail 1 ($A \to B \to D$): 중간 $B$만 있음. $B \notin Z$ → open.
- 첫 trail이 열려 있으므로 **NOT d-sep**.

</details>

**문제 2** (심화): Collider $X \to W \leftarrow Y$에서 $W$의 **후손**을 관측해도 왜 $X$와 $Y$가 dependent해지는지 직관과 대수를 모두로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**직관**: $W$의 자식 $D$를 관측하면, $D$는 $W$의 정보를 일부 담고 있다 ($W$가 유일 부모라면). 따라서 $D$ 관측 = $W$에 대한 noisy 관측. Noisy 관측이라도 $W$에 대한 정보를 주므로, explaining away 효과가 부분적으로 작동.

**대수**: $W \to D$이면 $p(X, Y, W, D) = p(X) p(Y) p(W | X, Y) p(D | W)$. $D$를 조건으로:
$$p(X, Y | D) = \sum_W p(X, Y, W | D) = \sum_W \frac{p(X) p(Y) p(W | X, Y) p(D | W)}{p(D)}$$
$$= \frac{p(X) p(Y)}{p(D)} \sum_W p(W | X, Y) p(D | W)$$

일반적으로 $\sum_W p(W | X, Y) p(D | W) \neq p(D | X) p(D | Y) / \text{const}$로 factorize 불가 → **dependent**.

구체적 예: $X, Y \sim \text{Bernoulli}(0.5)$, $W = X + Y$ (0, 1, 2 중 하나), $D = \mathbb{1}[W \geq 1]$. 그러면 $D = 1$ 관측 시 $P(X = 0, Y = 0 | D = 1) = 0$이지만 $P(X = 0 | D = 1) P(Y = 0 | D = 1) > 0$ — dependent.

</details>

**문제 3** (AI 연결): VAE의 generative model이 $z \to x$ 구조이고 posterior $q(z | x)$를 학습한다. 두 개의 데이터 포인트 $x_1, x_2$에 대해 $z_1, z_2$가 $x_1, x_2$ 주어졌을 때 CI인가? 이것이 mini-batch 학습과 어떤 관계인가?

<details>
<summary>힌트 및 해설</summary>

VAE의 plate notation:
```
     z_i ──► x_i      i = 1, ..., N
```

즉 각 데이터 포인트마다 독립적 $z_i \to x_i$ 구조, 그리고 $i$ 간 **공유 파라미터** $\theta, \phi$가 있음.

**질문: $z_1 \perp_d z_2 \mid x_1, x_2$?** 

파라미터를 고정시키면 (즉 $\theta, \phi$가 observed의 경우), 각 $(z_i, x_i)$ pair는 서로 다른 plate 안. 그래프는:
```
z_1 → x_1
z_2 → x_2
```
이 둘 사이에 path가 **없음** (파라미터를 제외하면). → **d-separated**.

따라서 posterior는 factorize: $q(z_1, z_2 | x_1, x_2) = q(z_1 | x_1) q(z_2 | x_2)$.

**Mini-batch 학습과의 관계**: 이 CI가 성립하므로 각 데이터 포인트를 **독립적으로** encode/decode할 수 있다. ELBO도 $\sum_i \text{ELBO}(x_i)$로 분해되어 mini-batch 평균으로 추정 가능. 만약 $z_1, z_2$ 간 CI가 깨지면 (hierarchical VAE 같은) 전체 데이터셋을 동시에 처리해야 해서 mini-batch 학습 불가.

이것이 BN 구조의 **계산적 이점** 중 하나 — CI가 모델의 병렬·분산 학습을 가능케 함.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Bayesian Network — DAG 기반 인수분해](./02-bayesian-network-factorization.md) | [📚 README](../README.md) | [04. Markov Random Field와 Hammersley–Clifford ▶](./04-markov-random-field.md) |

</div>
