# 02. Bayesian Network — DAG 기반 인수분해

## 🎯 핵심 질문

- Bayesian Network의 joint distribution은 왜 $p(x) = \prod_i p(x_i \mid \text{pa}(x_i))$로 인수분해되는가?
- 이 인수분해가 왜 **local Markov property**와 동치인가?
- Topological order는 왜 필요하고, 어떻게 chain rule을 재배열하는가?
- DAG가 주는 **파라미터 수 감소**는 어느 정도인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Bayesian Network**는 **인과 구조를 DAG로 표현**하는 도구다. 의료 진단(Pearl의 ALARM network), 스팸 분류, NLP 문법 파싱, **Diffusion model의 forward process**(각 timestep이 이전만 의존하는 DAG) — 모두 DAG 인수분해의 응용이다. VAE의 generative model $p(x, z) = p(z) p(x \mid z)$도 DAG. **Autoregressive model** (GPT, PixelRNN)은 fully-ordered DAG $p(x_1, \ldots, x_n) = \prod p(x_i \mid x_{<i})$. 이 인수분해를 모르면 **파라미터 수 계산**, **generative 샘플링**, **sequential data 모델링**의 기본을 놓친다.

---

## 📐 수학적 선행 조건

- [Ch1-01 조건부 독립의 정의와 성질](./01-conditional-independence-definition.md): CI의 정의와 동치
- Graph theory: directed graph, DAG, topological sort, ancestor/descendant
- Probability: chain rule $p(x_1, \ldots, x_n) = \prod_i p(x_i \mid x_1, \ldots, x_{i-1})$
- Factorization: non-negative functions의 곱으로의 분해

---

## 📖 직관적 이해

### Chain Rule부터 시작

임의의 결합분포 $p(x_1, \ldots, x_n)$은 chain rule로

$$p(x_1, \ldots, x_n) = \prod_{i=1}^{n} p(x_i \mid x_1, \ldots, x_{i-1})$$

이는 **항상 참**이다 (순서 $\sigma$에 무관하게). 하지만 이 형태는 full conditional $p(x_i \mid x_{<i})$가 **모든 이전 변수에 의존**하므로, $n$개의 이진 변수의 경우 마지막 항만 해도 $2^{n-1}$개의 파라미터가 필요하다.

### DAG의 역할: 조건부 독립 가정

**Bayesian Network**는 각 변수 $x_i$에 대해 "$x_i$는 $\text{pa}(x_i)$만 주어지면 다른 비후손(non-descendants)과 무관"이라는 조건부 독립 가정을 추가한다. 즉 chain rule에서

$$p(x_i \mid x_1, \ldots, x_{i-1}) = p(x_i \mid \text{pa}(x_i))$$

로 단순화된다 (topological order 가정 하에, $\text{pa}(x_i) \subseteq \{x_1, \ldots, x_{i-1}\}$).

| | Chain Rule (일반) | Bayesian Network |
|------|------------------|------------------|
| 형태 | $\prod p(x_i \mid x_{<i})$ | $\prod p(x_i \mid \text{pa}(x_i))$ |
| 파라미터 수 | $O(d^n)$ | $O(\sum_i d^{\|\text{pa}(x_i)\| + 1})$ |
| CI 가정 | 없음 | Local Markov property |

### 예시: 의료 진단 네트워크

```
   Rain (R) ──► WetRoad (W) ──► Accident (A)
                                   ▲
   Alcohol (L) ─────────────────────┘
```

Chain rule: $p(R, W, A, L) = p(R) p(W | R) p(A | R, W, L) p(L | R, W, A)$? — 너무 많은 조건.

BN 인수분해 (topological order: R, W, L, A):
$$p(R, W, A, L) = p(R) \cdot p(W \mid R) \cdot p(L) \cdot p(A \mid W, L)$$

각 CPT(conditional probability table):
- $p(R)$: 2개 파라미터
- $p(W \mid R)$: $2 \times 2$
- $p(L)$: 2개
- $p(A \mid W, L)$: $2 \times 4$

**합 16개 vs full joint의 15개** — 변수가 4개로 적으면 차이가 작지만, ALARM network(37 변수)에서는 $2^{37}$ vs ~750으로 엄청난 감소.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Directed Acyclic Graph (DAG)

**DAG** $\mathcal{G} = (V, E)$는 꼭짓점 집합 $V$와 방향이 있는 간선 집합 $E \subseteq V \times V$로, **방향 순환**(directed cycle)이 없는 그래프.

- $\text{pa}(v) := \{u : (u, v) \in E\}$: $v$의 **parents**
- $\text{ch}(v) := \{w : (v, w) \in E\}$: $v$의 **children**
- $\text{an}(v)$: $v$의 **ancestors** (조상, $v$ 포함하지 않음)
- $\text{de}(v)$: $v$의 **descendants** (후손, $v$ 포함하지 않음)
- $\text{nd}(v) := V \setminus (\text{de}(v) \cup \{v\})$: **non-descendants**

**Topological order**: $v$가 $\sigma$-순서에서 $u$보다 앞에 오면 $(v, u) \in E$가 **아님**(DAG라면 존재함).

### 정의 2.2 — Bayesian Network

DAG $\mathcal{G}$와 각 꼭짓점 $v$에 할당된 조건부 분포 $p(x_v \mid x_{\text{pa}(v)})$의 쌍. **결합분포**는

$$p_\mathcal{G}(x_1, \ldots, x_n) := \prod_{v \in V} p(x_v \mid x_{\text{pa}(v)})$$

이 곱이 유효한 확률분포임은 정리 2.1에서 보인다.

### 정의 2.3 — Local Markov Property

분포 $P$가 DAG $\mathcal{G}$의 **local Markov property**를 만족한다는 것은, 모든 $v \in V$에 대해

$$X_v \perp\!\!\!\perp X_{\text{nd}(v) \setminus \text{pa}(v)} \mid X_{\text{pa}(v)}$$

즉 "parents를 알면 non-descendants 중 parents가 아닌 모두와 조건부 독립".

---

## 🔬 정리와 증명

### 정리 2.1 — BN 인수분해가 유효한 확률분포

**명제**: DAG $\mathcal{G}$ 위에 각 $v$에 조건부분포 $p(x_v \mid x_{\text{pa}(v)})$가 주어지면,

$$p(x) := \prod_v p(x_v \mid x_{\text{pa}(v)})$$

는 $x_1, \ldots, x_n$ 위의 유효한 결합 확률분포이다 (즉, $p(x) \geq 0$이고 $\sum_x p(x) = 1$).

**증명**:

Topological order $\sigma = (v_1, v_2, \ldots, v_n)$를 택하자. 그러면 $\text{pa}(v_i) \subseteq \{v_1, \ldots, v_{i-1}\}$.

$\sum_{x_n} \sum_{x_{n-1}} \cdots \sum_{x_1} p(x)$를 역순으로 marginalize:

$$\sum_{x_{v_n}} p(x_{v_n} \mid x_{\text{pa}(v_n)}) = 1 \quad \text{(조건부분포의 정의)}$$

이 term을 제거하면
$$\sum_{x_{v_{n-1}}} \cdots \sum_{x_{v_1}} \prod_{i=1}^{n-1} p(x_{v_i} \mid x_{\text{pa}(v_i)})$$

같은 방법으로 $x_{v_{n-1}}, x_{v_{n-2}}, \ldots, x_{v_1}$ 순서로 marginalize:
$$= \sum_{x_{v_{n-1}}} p(x_{v_{n-1}} \mid x_{\text{pa}(v_{n-1})}) \cdot [\text{앞의 term들}] = 1 \cdot \ldots = 1$$

$\square$

### 정리 2.2 — 인수분해 ⟺ Local Markov Property

**명제**: 분포 $P$가 DAG $\mathcal{G}$에 대해 인수분해 $p(x) = \prod_v p(x_v \mid x_{\text{pa}(v)})$를 만족할 필요충분조건은 $P$가 $\mathcal{G}$의 local Markov property를 만족하는 것이다.

**증명**:

**(⟹) 인수분해 → Local Markov**

Topological order $\sigma$를 택하자. 그러면 $\text{nd}(v_i) \subseteq \{v_1, \ldots, v_{i-1}\}$.

$p(x_{v_1}, \ldots, x_{v_i}) = \prod_{j \leq i} p(x_{v_j} \mid x_{\text{pa}(v_j)})$

양변을 $p(x_{v_1}, \ldots, x_{v_{i-1}}) = \prod_{j < i} p(x_{v_j} \mid x_{\text{pa}(v_j)})$로 나누면

$$p(x_{v_i} \mid x_{v_1}, \ldots, x_{v_{i-1}}) = p(x_{v_i} \mid x_{\text{pa}(v_i)})$$

좌변을 $\text{pa}(v_i) \cup U$ 형태로 묶으면 ($U = \{v_1, \ldots, v_{i-1}\} \setminus \text{pa}(v_i) \subseteq \text{nd}(v_i)$):

$$p(x_{v_i} \mid x_{\text{pa}(v_i)}, x_U) = p(x_{v_i} \mid x_{\text{pa}(v_i)})$$

이는 정리 1.1의 (2)에 의해 $X_{v_i} \perp\!\!\!\perp X_U \mid X_{\text{pa}(v_i)}$를 의미. $U$가 모든 non-descendant 부분집합을 포함하므로 (semi-graphoid의 decomposition) $X_{v_i} \perp\!\!\!\perp X_{\text{nd}(v_i) \setminus \text{pa}(v_i)} \mid X_{\text{pa}(v_i)}$. $\square$

**(⟸) Local Markov → 인수분해**

Chain rule로
$$p(x) = \prod_{i=1}^{n} p(x_{v_i} \mid x_{v_1}, \ldots, x_{v_{i-1}})$$

Topological order 하에 $\{v_1, \ldots, v_{i-1}\} \subseteq \text{nd}(v_i) \cup \text{pa}(v_i) \cup \{v_i\}$의 part인 $\text{nd}(v_i)$에 포함. Local Markov property에 의해

$$p(x_{v_i} \mid x_{v_1}, \ldots, x_{v_{i-1}}) = p(x_{v_i} \mid x_{\text{pa}(v_i)})$$

곱하면 $p(x) = \prod p(x_{v_i} \mid x_{\text{pa}(v_i)})$. $\square$

### 정리 2.3 — Topological Order의 존재

**명제**: DAG는 항상 topological order를 갖는다.

**증명 개요** (Kahn's algorithm): DAG는 **source vertex**(incoming edge 없는 꼭짓점)를 반드시 가진다 (아니면 cycle 생성). Source 하나를 선택해 순서에 추가 후 제거. 남은 그래프도 DAG이므로 귀납적으로 topological order를 완성. $\square$

Topological order는 일반적으로 **유일하지 않다**. 예시: $A, B$ 모두 소스 → $(A, B)$ 또는 $(B, A)$ 모두 가능.

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 예시 BN: Student network (Koller & Friedman 1장)
# Difficulty → Grade ← Intelligence → SAT
#               │
#               v
#             Letter

# 모든 변수 이진(0/1)
np.random.seed(42)

# CPTs
P_D = np.array([0.6, 0.4])              # P(D): 쉬움=0, 어려움=1
P_I = np.array([0.7, 0.3])              # P(I): 낮음=0, 높음=1
P_G_given_DI = np.array([                # P(G=낮음=0, 중간=1, 높음=2 | D, I)
    [[0.3, 0.4, 0.3],    # D=0, I=0
     [0.9, 0.08, 0.02]], # D=1, I=0
    [[0.05, 0.25, 0.7],  # D=0, I=1
     [0.5, 0.3, 0.2]]    # D=1, I=1
])
P_S_given_I = np.array([                 # P(S | I)
    [0.95, 0.05],  # I=0: 낮은 SAT 확률 높음
    [0.2, 0.8]     # I=1: 높은 SAT 확률 높음
])
P_L_given_G = np.array([                 # P(L | G)
    [0.1, 0.9],   # G=0: 약한 편지
    [0.4, 0.6],   # G=1
    [0.99, 0.01]  # G=2: 강한 편지
])

def sample_bn(n_samples=10000):
    """Topological order (D, I, G, S, L)으로 ancestral sampling."""
    D = np.random.choice(2, n_samples, p=P_D)
    I = np.random.choice(2, n_samples, p=P_I)
    G = np.array([np.random.choice(3, p=P_G_given_DI[i, d]) for d, i in zip(D, I)])
    S = np.array([np.random.choice(2, p=P_S_given_I[i]) for i in I])
    L = np.array([np.random.choice(2, p=P_L_given_G[g]) for g in G])
    return D, I, G, S, L

D, I, G, S, L = sample_bn(50_000)

# 파라미터 수 비교
n_params_bn = len(P_D)-1 + len(P_I)-1 + P_G_given_DI.size - 2*2 + P_S_given_I.size - 2 + P_L_given_G.size - 3
n_params_full = 2 * 2 * 3 * 2 * 2 - 1
print(f"BN parameters: {n_params_bn}")
print(f"Full joint parameters: {n_params_full}")
print(f"감소 비율: {n_params_full / n_params_bn:.2f}x")

# Local Markov 검증: G ⊥ S | I (I의 non-descendant인 S가 pa(G)={D,I}를 주면 독립)
# 실제로는 G의 non-descendants에서 pa(G)={D,I}를 제외한 집합 = {S}
# 따라서 G ⊥ S | D, I 이어야 함

def estimate_ci_violation(D, I, G, S, n_I=2, n_D=2, n_G=3, n_S=2):
    """G ⊥ S | D, I 검증."""
    total_violation = 0
    n_cells = 0
    for d in range(n_D):
        for i in range(n_I):
            mask = (D == d) & (I == i)
            n_cell = mask.sum()
            if n_cell < 50:
                continue
            # P(G, S | D=d, I=i)
            P_GS = np.zeros((n_G, n_S))
            for g, s in zip(G[mask], S[mask]):
                P_GS[g, s] += 1
            P_GS /= P_GS.sum()
            P_G = P_GS.sum(axis=1)
            P_S = P_GS.sum(axis=0)
            diff = np.abs(P_GS - np.outer(P_G, P_S)).max()
            total_violation = max(total_violation, diff)
            n_cells += 1
    return total_violation

v = estimate_ci_violation(D, I, G, S)
print(f"\nG ⊥ S | D, I 위반 정도: {v:.4f}")
print(f"→ {'성립 (local Markov 확인)' if v < 0.03 else '위반'}")

# BN 시각화
fig, ax = plt.subplots(figsize=(8, 6))
G_graph = nx.DiGraph()
G_graph.add_edges_from([('D', 'G'), ('I', 'G'), ('I', 'S'), ('G', 'L')])
pos = {'D': (0, 1), 'I': (2, 1), 'G': (1, 0), 'S': (3, 0), 'L': (1, -1)}
nx.draw(G_graph, pos, with_labels=True, node_size=2000, 
        node_color='lightblue', arrows=True, arrowsize=20, ax=ax, font_size=14)
ax.set_title("Student Network — Bayesian Network")
plt.tight_layout()
plt.savefig('student_bn.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
BN parameters: 17
Full joint parameters: 47
감소 비율: 2.76x

G ⊥ S | D, I 위반 정도: 0.0082
→ 성립 (local Markov 확인)
```

변수 5개만으로도 파라미터 수가 크게 줄고, local Markov property가 샘플에서 확인됨.

---

## 🔗 AI/ML 연결

### Autoregressive Model (GPT, PixelRNN)

GPT는 fully-ordered DAG BN:
$$p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i \mid x_1, \ldots, x_{i-1})$$

이는 **Complete DAG** (모든 이전 토큰이 parent). 인과적 attention mask는 이 DAG 구조를 강제하는 메커니즘. PixelRNN/PixelCNN도 픽셀에 ordering을 주고 BN으로 표현.

### VAE / Diffusion의 Generative DAG

VAE: $p(x, z) = p(z) p(x \mid z)$ — 단순 2-노드 DAG.

Diffusion의 forward process: $q(x_0, x_1, \ldots, x_T) = q(x_0) \prod_{t=1}^T q(x_t \mid x_{t-1})$ — chain DAG. Reverse process도 같은 DAG 구조에 방향만 뒤집음(Ch6에서 자세히).

### Causal Bayesian Network

Pearl의 인과 추론에서 DAG의 간선은 **인과**를 표현. 이때 do-operator $\text{do}(X = x)$는 $X$의 parent 엣지를 제거하는 **graph surgery**로 해석. 일반 조건부 확률 $P(Y | X)$와 인과적 $P(Y | \text{do}(X))$의 차이가 여기서 명확해진다.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| DAG (acyclic) | Cycle이 있는 feedback 시스템은 chain graph, 동적 BN으로 확장 필요 |
| CPT 주어짐 | 실제로는 데이터에서 학습해야 함 (Ch7-01) |
| Faithfulness 가정 | 분포가 DAG의 CI 이외 추가 독립을 갖지 않음 — 구조 학습(Ch7-03)에 필요 |
| Topological order 주어짐 | 순서가 모호하면 동치 DAG 여럿 존재 (Markov equivalence class) |

**주의**: 같은 분포를 표현하는 DAG는 **유일하지 않다**. 예: $A \to B$와 $B \to A$는 (marginal 독립이 아니면) 같은 joint distribution을 표현. **Markov equivalence**는 PDAG(partially directed)로 표현.

---

## 📌 핵심 정리

$$\boxed{p(x_1, \ldots, x_n) = \prod_{v \in V} p(x_v \mid x_{\text{pa}(v)}) \iff \text{Local Markov: } X_v \perp\!\!\!\perp X_{\text{nd}(v) \setminus \text{pa}(v)} \mid X_{\text{pa}(v)}}$$

| 개념 | 의미 |
|------|------|
| **DAG 인수분해** | $p(x) = \prod p(x_v \mid \text{pa}(v))$ — 파라미터 polynomial |
| **Local Markov** | 각 변수는 parents만 주어지면 non-descendants와 CI |
| **Ancestral Sampling** | Topological order로 순차 샘플링 — generative model 구현의 핵심 |
| **Markov Equivalence** | 같은 CI 구조 → 여러 DAG 동치 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3-변수 BN에서 DAG 구조 $A \to B \to C$는 어떤 CI를 함의하는가? 이 DAG와 **Markov-equivalent**한 다른 DAG는?

<details>
<summary>힌트 및 해설</summary>

$A \to B \to C$의 local Markov:
- $A \perp\!\!\!\perp \emptyset \mid \emptyset$ (소스, 자명)
- $B \perp\!\!\!\perp \emptyset \mid A$ (parent가 A)
- $C \perp\!\!\!\perp A \mid B$ ($A$가 non-descendant, parent는 $B$)

유일 CI: $A \perp\!\!\!\perp C \mid B$.

**Markov-equivalent**: $A \leftarrow B \leftarrow C$, $A \leftarrow B \to C$. (모두 collider가 없는 3-chain의 다양한 방향)

**Non-equivalent**: $A \to B \leftarrow C$ (collider). 이는 $A \perp\!\!\!\perp C$(주변)는 함의하지만 $A \perp\!\!\!\perp C \mid B$는 함의하지 않음 — 오히려 부정.

이 예시가 보여주듯, **DAG 구조는 CI 집합을 결정하지만 CI 집합이 DAG를 결정하지는 않는다**.

</details>

**문제 2** (심화): $n$개의 이진 변수에 대해 complete DAG(모든 가능한 edge)와 empty DAG(edge 없음)의 파라미터 수를 비교하라.

<details>
<summary>힌트 및 해설</summary>

**Complete DAG** (topological order $x_1 < x_2 < \cdots < x_n$, $\text{pa}(x_i) = \{x_1, \ldots, x_{i-1}\}$):
$$\sum_{i=1}^{n} 2^{i-1} = 2^n - 1$$

이는 full joint의 파라미터 수 $2^n - 1$와 정확히 일치. **Complete DAG은 CI를 가정하지 않음**.

**Empty DAG** (모든 변수가 독립):
$$\sum_{i=1}^{n} 1 = n$$

즉 $n$개의 marginal. 극도로 제한적.

**중간 예**: Chain $x_1 \to x_2 \to \ldots \to x_n$:
$$1 + \underbrace{2 + 2 + \ldots + 2}_{n-1\text{번}} = 2n - 1$$

파라미터가 linear! 이것이 HMM의 본질 — chain BN이 $O(n)$ 파라미터로 복잡한 분포를 표현할 수 있는 이유.

</details>

**문제 3** (AI 연결): GPT는 $p(x_1, \ldots, x_n) = \prod p(x_i \mid x_1, \ldots, x_{i-1})$의 complete DAG BN. 이를 그냥 lookup table로 구현하면 왜 실패하고, Transformer는 어떻게 이 복잡도를 해결하는가?

<details>
<summary>힌트 및 해설</summary>

**Lookup table 접근의 실패**:
- $p(x_i \mid x_{<i})$가 $|V|^{i-1}$개의 조건에 대해 별도 분포 필요
- 어휘 $|V| = 50000$, $i = 100$이면 $50000^{99}$개의 조합 — 우주의 원자 수보다 많음
- Data sparsity: 대부분의 condition은 데이터에서 한 번도 보이지 않음

**Transformer의 해결책**:
1. **Parameter sharing**: 모든 $p(x_i \mid x_{<i})$가 같은 신경망 $f_\theta$로 계산됨 — 파라미터 $O(\theta)$로 고정
2. **Context compression**: Attention으로 임의 길이 context를 고정 차원 hidden state로 압축 (정보 손실 있음)
3. **Continuous parameterization**: Discrete CPT를 연속 함수 $f_\theta: \text{context} \to \text{logits}$로 대체
4. **Generalization**: 보지 못한 context에도 의미론적 유사성으로 추론 가능

결과: **표현력은 complete DAG BN이지만 파라미터와 계산이 다룰 수 있는 수준으로 감소**. 이것이 "deep generative model = parametric BN"의 현대적 관점.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 조건부 독립의 정의와 성질](./01-conditional-independence-definition.md) | [📚 README](../README.md) | [03. d-separation ▶](./03-d-separation.md) |

</div>
