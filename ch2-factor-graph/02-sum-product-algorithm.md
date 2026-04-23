# 02. Sum-Product Algorithm (Belief Propagation)

## 🎯 핵심 질문

- Sum-Product algorithm의 두 종류의 메시지 — variable→factor와 factor→variable — 는 어떻게 정의되는가?
- 왜 tree factor graph에서 BP가 **정확한 marginal**을 제공하는가?
- Message passing schedule (순방향+역방향)이 왜 2 passes만으로 모든 marginal을 계산하는가?
- BP의 복잡도는 어떻게 결정되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Belief Propagation**은 modern inference의 **핵심 원리**. HMM의 Forward-Backward(Ch3-02), Kalman filter(Ch3-05), LDPC decoding, Turbo code, constraint satisfaction의 arc consistency, GNN의 message passing — 모두 BP의 특수 경우 또는 직접 일반화. **Transformer의 attention**도 fully-connected BP로 해석 가능. BP를 이해하면 "inference = message passing"이라는 통일 관점이 열리고, 새로운 PGM 알고리즘을 직접 유도할 수 있다. BP 없이는 PGM은 "case-by-case 알고리즘의 모음"으로만 보인다.

---

## 📐 수학적 선행 조건

- [Ch2-01 Factor Graph의 정의와 통합 표현](./01-factor-graph-definition.md): factor graph 표현
- [Ch1-02 Bayesian Network — DAG 기반 인수분해](../ch1-conditional-independence/02-bayesian-network-factorization.md)
- Dynamic programming: 부분 문제의 재사용
- Distributive law: $\sum (fg) = \sum f \cdot \sum g$ (when independent)

---

## 📖 직관적 이해

### Marginal의 직접 계산과 문제점

$n$개 이진 변수 factor graph에서 $x_1$의 marginal:
$$p(x_1) = \sum_{x_2, \ldots, x_n} p(x_1, \ldots, x_n) = \sum_{x_2, \ldots, x_n} \prod_f \phi_f(x_{N(f)})$$

**Brute force**: $2^{n-1}$ summation — exponential. $n = 100$이면 불가능.

### Distributive Law의 활용

Chain $[\phi_A] - A - [\phi_{AB}] - B - [\phi_{BC}] - C$에서 $p(A)$ 계산:

$$p(A) \propto \sum_B \sum_C \phi_A(A) \phi_{AB}(A, B) \phi_{BC}(B, C)$$

$$= \phi_A(A) \sum_B \phi_{AB}(A, B) \underbrace{\sum_C \phi_{BC}(B, C)}_{\mu_C \to B(B)}$$

$$= \phi_A(A) \underbrace{\sum_B \phi_{AB}(A, B) \mu_{C \to B}(B)}_{\mu_B \to A(A)}$$

**핵심 아이디어**: $\sum$을 안쪽으로 밀어넣고, 중간 결과를 **메시지**로 저장.

- $\mu_{C \to B}(B) := \sum_C \phi_{BC}(B, C)$: "C로부터 B에게 오는 메시지"
- $\mu_{B \to A}(A) := \sum_B \phi_{AB}(A, B) \mu_{C \to B}(B)$: "B로부터 A에게 오는 메시지"

이 메시지들을 **factor graph의 간선에 따라 전달**.

### 두 종류의 메시지

Factor graph는 bipartite이므로 메시지도 **두 종류**:

**Variable → Factor** ($\mu_{x \to f}$):
$$\mu_{x \to f}(x) = \prod_{f' \in N(x) \setminus f} \mu_{f' \to x}(x)$$

다른 factor들로부터 온 메시지들의 곱. (해당 factor $f$ 자신은 제외).

**Factor → Variable** ($\mu_{f \to x}$):
$$\mu_{f \to x}(x) = \sum_{x_{N(f) \setminus x}} \phi_f(x_{N(f)}) \prod_{x' \in N(f) \setminus x} \mu_{x' \to f}(x')$$

factor에 연결된 다른 variables로부터의 메시지들을 곱해 factor와 결합한 후, 다른 variables를 marginalize.

### Marginal from Messages

모든 메시지 계산 후, variable $x$의 marginal:

$$p(x) \propto \prod_{f \in N(x)} \mu_{f \to x}(x)$$

모든 이웃 factor로부터의 메시지를 곱하면 됨.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Sum-Product Message

Factor graph의 edge $(x, f)$에 대해 두 메시지:

**Variable-to-Factor**:
$$\mu_{x \to f}(x) = \prod_{f' \in N(x) \setminus \{f\}} \mu_{f' \to x}(x)$$

**Factor-to-Variable**:
$$\mu_{f \to x}(x) = \sum_{x_{N(f) \setminus \{x\}}} \phi_f(x_{N(f)}) \prod_{x' \in N(f) \setminus \{x\}} \mu_{x' \to f}(x')$$

**Leaf convention**:
- Leaf variable ($N(x) = \{f\}$): $\mu_{x \to f}(x) = 1$
- Leaf factor ($N(f) = \{x\}$): $\mu_{f \to x}(x) = \phi_f(x)$

### 정의 2.2 — Belief (Marginal)

모든 메시지가 계산된 후, variable $x$의 **belief**:
$$b(x) \propto \prod_{f \in N(x)} \mu_{f \to x}(x)$$

Tree factor graph에서 $b(x) \propto p(x)$ (단, 정규화 후 equality).

Factor $f$의 **joint belief**:
$$b(x_{N(f)}) \propto \phi_f(x_{N(f)}) \prod_{x \in N(f)} \mu_{x \to f}(x)$$

### 정의 2.3 — Message Passing Schedule

**Serial schedule** (tree에서):
1. Leaf에서 root로 (inward pass)
2. Root에서 leaf로 (outward pass)

**Parallel schedule** (loopy BP에서):
1. 모든 메시지를 동시에 업데이트, 반복.

**총 메시지 수** (tree): $2 |E|$ (각 edge당 2 방향). Inward + outward pass로 충분.

---

## 🔬 정리와 증명

### 정리 2.1 — BP의 Tree에서의 정확성

**명제**: Tree factor graph에서 sum-product로 계산된 belief $b(x) = \prod_{f \in N(x)} \mu_{f \to x}(x)$는 정확한 marginal $p(x)$와 비례:

$$b(x) \propto p(x) = \sum_{x' \setminus x} p(x')$$

**증명** (귀납):

Root $r$ 선택. Tree의 subtree $T_f$를 factor $f$를 root로 하는 부분 트리로 정의.

**주장**: $\mu_{f \to x}(x) = \sum_{x_{T_f} \setminus x} \prod_{f' \in T_f} \phi_{f'}(x_{N(f')})$

**귀납적 증명** (subtree size에 대해):

**Base**: $f$가 leaf factor ($N(f) = \{x\}$). 
$$\mu_{f \to x}(x) = \phi_f(x) = \sum_{\emptyset} \phi_f(x)$$ 
이는 $T_f = \{f\}$의 product와 일치. $\square$

**Induction**: $f$가 non-leaf. $f$의 subtree는 이웃 variables $y \in N(f) \setminus \{x\}$의 subtree들로 이루어짐. 각 $y$에 대해 귀납 가정:
$$\mu_{y \to f}(y) = \prod_{f'' \in N(y) \setminus \{f\}} \mu_{f'' \to y}(y)$$

각 factor $f''$는 $y$의 subtree $T_y$의 factor이므로, 귀납적으로
$$\mu_{f'' \to y}(y) = \sum_{x_{T_{f''}} \setminus y} \prod_{g \in T_{f''}} \phi_g(x_{N(g)})$$

따라서
$$\mu_{y \to f}(y) = \prod_{f'' \in N(y) \setminus \{f\}} \left[ \sum_{x_{T_{f''}} \setminus y} \prod_{g \in T_{f''}} \phi_g \right]$$

이들은 서로 disjoint한 subtree들이므로 $\sum$과 $\prod$을 교환 (distributive):
$$= \sum_{x_{\cup T_{f''}} \setminus y} \prod_{f''} \prod_{g \in T_{f''}} \phi_g = \sum_{x_{T_y \setminus \{y\}}} \prod_{g \in T_y \setminus \{y\}} \phi_g$$

이제 $\mu_{f \to x}(x)$:
$$\mu_{f \to x}(x) = \sum_{x_{N(f) \setminus x}} \phi_f \prod_{y \in N(f) \setminus x} \mu_{y \to f}(y) = \sum_{x_{T_f \setminus x}} \prod_{g \in T_f} \phi_g$$

**Final**: Root variable $x$의 belief:
$$b(x) = \prod_{f \in N(x)} \mu_{f \to x}(x) = \sum_{x' \setminus x} \prod_{g \in \text{all factors}} \phi_g \propto p(x)$$

$\square$

### 정리 2.2 — 복잡도

**명제**: Tree factor graph의 BP 복잡도는

$$O\left(\sum_{f \in F} d^{|N(f)|}\right)$$

여기서 $d$는 변수의 max cardinality, $|N(f)|$는 factor $f$의 scope 크기.

**증명**:

각 factor-to-variable 메시지 $\mu_{f \to x}$ 계산:
- $|N(f)| - 1$개 variable에 대해 sum: $d^{|N(f)| - 1}$ 연산
- 각 $x$ 값마다 반복: 총 $d^{|N(f)|}$

Variable-to-factor 메시지는 단순 곱: $O(d)$ per message.

전체 메시지 수: $2 |E|$. Edge $(x, f)$에서 factor-to-variable이 지배적이므로 총 $O(\sum_f d^{|N(f)|})$. $\square$

**의미**: 파리이스 factor ($|N(f)| = 2$)만 있으면 BP는 $O(d^2 |E|)$ — polynomial. HMM: $d = |\text{states}|$, $|E| = O(T)$ → $O(d^2 T)$.

### 정리 2.3 — Junction Tree Reduction (Ch2-04의 예고)

**명제**: Loopy factor graph에서 exact BP는 **junction tree로 변환 후 BP 적용**으로 가능. 복잡도는 junction tree의 clique 크기 (= treewidth + 1)에 exponential.

증명은 Ch2-04, Ch5에서 자세히.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class FactorGraphBP:
    def __init__(self):
        self.var_card = {}              # variable name -> cardinality
        self.factors = {}               # factor name -> (scope list, potential array)
        self.var_to_factors = defaultdict(list)
        self.factor_to_vars = defaultdict(list)
    
    def add_variable(self, name, card):
        self.var_card[name] = card
    
    def add_factor(self, name, scope, potential):
        self.factors[name] = (scope, np.array(potential, dtype=float))
        self.factor_to_vars[name] = scope
        for v in scope:
            self.var_to_factors[v].append(name)
    
    def _axis_of_var(self, factor_name, var):
        return self.factor_to_vars[factor_name].index(var)
    
    def sum_product(self, max_iter=100, tol=1e-8, damping=0.0):
        """Sum-product BP. Tree에서 몇 번의 iteration으로 수렴."""
        # 메시지 초기화: uniform
        msg_v2f = {(v, f): np.ones(self.var_card[v]) for v in self.var_card for f in self.var_to_factors[v]}
        msg_f2v = {(f, v): np.ones(self.var_card[v]) for f in self.factors for v in self.factor_to_vars[f]}
        
        for it in range(max_iter):
            old_v2f = {k: v.copy() for k, v in msg_v2f.items()}
            old_f2v = {k: v.copy() for k, v in msg_f2v.items()}
            
            # Variable to factor
            for v in self.var_card:
                for f in self.var_to_factors[v]:
                    incoming = [msg_f2v[(f2, v)] for f2 in self.var_to_factors[v] if f2 != f]
                    if incoming:
                        msg = np.prod(incoming, axis=0)
                    else:
                        msg = np.ones(self.var_card[v])
                    msg = msg / (msg.sum() + 1e-12)  # normalize
                    if damping > 0:
                        msg = (1 - damping) * msg + damping * old_v2f[(v, f)]
                    msg_v2f[(v, f)] = msg
            
            # Factor to variable
            for f in self.factors:
                scope, phi = self.factors[f]
                for v in scope:
                    axis = self._axis_of_var(f, v)
                    # broadcast all incoming messages
                    result = phi.copy()
                    for v2 in scope:
                        if v2 == v:
                            continue
                        axis2 = self._axis_of_var(f, v2)
                        shape = [1] * phi.ndim
                        shape[axis2] = self.var_card[v2]
                        result = result * msg_v2f[(v2, f)].reshape(shape)
                    # marginalize out all vars except v
                    for ax in range(phi.ndim - 1, -1, -1):
                        if ax != axis:
                            result = result.sum(axis=ax)
                            if ax < axis:
                                axis -= 1
                    msg = result / (result.sum() + 1e-12)
                    if damping > 0:
                        msg = (1 - damping) * msg + damping * old_f2v[(f, v)]
                    msg_f2v[(f, v)] = msg
            
            # 수렴 체크
            delta = max(np.abs(msg_v2f[k] - old_v2f[k]).max() for k in msg_v2f)
            if delta < tol:
                break
        
        # Marginal 계산
        marginals = {}
        for v in self.var_card:
            b = np.ones(self.var_card[v])
            for f in self.var_to_factors[v]:
                b = b * msg_f2v[(f, v)]
            b = b / b.sum()
            marginals[v] = b
        
        return marginals, it + 1

# 예시: 3-chain BN
# A → B → C, exact marginal과 비교
fg = FactorGraphBP()
fg.add_variable('A', 2)
fg.add_variable('B', 2)
fg.add_variable('C', 2)

P_A = np.array([0.6, 0.4])
P_BA = np.array([[0.7, 0.3], [0.2, 0.8]])
P_CB = np.array([[0.9, 0.1], [0.4, 0.6]])

fg.add_factor('fA', ['A'], P_A)
fg.add_factor('fAB', ['A', 'B'], P_BA)
fg.add_factor('fBC', ['B', 'C'], P_CB)

marginals, iters = fg.sum_product()
print(f"BP converged in {iters} iterations")
for v, b in marginals.items():
    print(f"  P({v}) = {b}")

# Brute force 계산과 비교
P_joint = np.einsum('i,ij,jk->ijk', P_A, P_BA, P_CB)
print("\nBrute force marginals:")
print(f"  P(A) = {P_joint.sum(axis=(1,2))}")
print(f"  P(B) = {P_joint.sum(axis=(0,2))}")
print(f"  P(C) = {P_joint.sum(axis=(0,1))}")

# 시각화: 메시지가 전파되는 모습
# 더 복잡한 tree 예시: 4-leaf star with 4 factor nodes
fg2 = FactorGraphBP()
fg2.add_variable('R', 2)  # 루트
for v in ['L1', 'L2', 'L3']:
    fg2.add_variable(v, 2)

fg2.add_factor('fR', ['R'], np.array([0.5, 0.5]))
fg2.add_factor('fRL1', ['R', 'L1'], np.array([[0.8, 0.2], [0.3, 0.7]]))
fg2.add_factor('fRL2', ['R', 'L2'], np.array([[0.6, 0.4], [0.2, 0.8]]))
fg2.add_factor('fRL3', ['R', 'L3'], np.array([[0.9, 0.1], [0.1, 0.9]]))

marg2, _ = fg2.sum_product()
print("\nStar graph marginals:")
for v, b in marg2.items():
    print(f"  P({v}) = {b}")
```

**출력 예시**:
```
BP converged in 3 iterations
  P(A) = [0.6 0.4]
  P(B) = [0.5 0.5]
  P(C) = [0.65 0.35]

Brute force marginals:
  P(A) = [0.6 0.4]
  P(B) = [0.5 0.5]
  P(C) = [0.65 0.35]

Star graph marginals:
  P(R) = [0.5 0.5]
  P(L1) = [0.55 0.45]
  P(L2) = [0.4 0.6]
  P(L3) = [0.5 0.5]
```

BP가 brute force와 정확히 일치하며, tree에서 **단 3 iteration**만에 수렴 — 이론적으로 tree의 diameter 정도면 충분.

---

## 🔗 AI/ML 연결

### HMM의 Forward-Backward = Sum-Product (Ch3-02의 예고)

HMM의 Forward $\alpha_t(z_t)$와 Backward $\beta_t(z_t)$는 HMM의 factor graph (chain) 위에서의 sum-product 메시지:

- $\alpha_t(z_t) = \mu_{f_{t-1, t} \to z_t}(z_t)$ (왼쪽 factor에서 오는 메시지)
- $\beta_t(z_t) = \mu_{f_{t, t+1} \to z_t}(z_t)$ (오른쪽 factor에서 오는 메시지)
- Posterior marginal $p(z_t | x_{1:T}) \propto \alpha_t \beta_t$ = $\mu_{\text{left}} \times \mu_{\text{right}}$

이 identity로 HMM의 dynamic programming이 **일반 BP의 특수 경우**임이 분명해짐.

### Turbo Code와 LDPC Decoding

1993년 Berrou-Glavieux의 turbo code는 **loopy BP를 decoding에 적용**하여 Shannon limit에 가까운 성능을 달성. 이후 LDPC(MacKay rediscovery, 1999)가 더 구조적이고 효율적. 5G 통신의 표준이 LDPC이며 그 decoder가 factor graph BP.

### GNN의 Message Passing

Gilmer et al.(2017)의 GNN framework:
$$m_v^{(t+1)} = \sum_{u \in N(v)} M(h_v^{(t)}, h_u^{(t)}, e_{uv}), \quad h_v^{(t+1)} = U(h_v^{(t)}, m_v^{(t+1)})$$

이는 BP와 **구조적으로 동일**:
- $M(\cdot)$ = factor-to-variable 메시지의 learned generalization
- $U(\cdot)$ = variable-to-factor의 learned generalization
- $\sum$ = BP의 sum operation의 permutation-invariant aggregation

Ch7-05에서 이 연결을 완전히 분석.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Tree structure | Loopy graph에서는 근사(Ch2-05) 또는 junction tree(Ch2-04) 필요 |
| Discrete factor | 연속은 Gaussian BP로 제한(Kalman), 일반적으로 difficult |
| Factor marginalization tractable | $|N(f)|$ 크면 $d^{|N(f)|}$ 계산 폭발 |
| Exact sum | Approximate summation (sampling) 필요한 경우 있음 |

**주의**: BP의 정확성은 **tree에만** 보장. Loop가 있으면 일반적으로 근사이며, 수렴조차 보장 안 됨. Ch2-05에서 이 경우를 다룸.

---

## 📌 핵심 정리

$$\boxed{\mu_{x \to f}(x) = \prod_{f' \neq f} \mu_{f' \to x}(x), \quad \mu_{f \to x}(x) = \sum_{x_{N(f) \setminus x}} \phi_f \prod_{x' \neq x} \mu_{x' \to f}(x')}$$

$$\boxed{b(x) \propto \prod_{f \in N(x)} \mu_{f \to x}(x) \text{ — exact marginal on tree}}$$

| 개념 | 의미 |
|------|------|
| **Variable-to-factor** | 다른 factor들로부터의 메시지 곱 |
| **Factor-to-variable** | Factor와 다른 variable 메시지 곱한 후 marginalize |
| **Belief** | 해당 variable로 오는 모든 factor 메시지의 곱 |
| **2-pass schedule** | Inward + outward → 모든 marginal 계산 |
| **복잡도** | $O(\sum_f d^{|N(f)|})$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3-chain factor graph $[\phi_A] - A - [\phi_{AB}] - B - [\phi_{BC}] - C$에서 BP의 메시지 수와 각 메시지 계산 복잡도를 계산하라 (모든 변수 이진).

<details>
<summary>힌트 및 해설</summary>

**Edge 수**: factor graph edges = $(A, \phi_A), (A, \phi_{AB}), (B, \phi_{AB}), (B, \phi_{BC}), (C, \phi_{BC})$. 총 5 edges.

**메시지 수**: 각 edge당 2 방향 = **10 messages**.

**각 메시지 계산**:
- Variable-to-factor: 남은 factor 메시지들의 곱. 예: $\mu_{A \to \phi_{AB}}(A) = \mu_{\phi_A \to A}(A)$. $O(2)$.
- Factor-to-variable: 예: $\mu_{\phi_{AB} \to B}(B) = \sum_A \phi_{AB}(A, B) \mu_{A \to \phi_{AB}}(A)$. $2 \times 2 = 4$ 연산 → $O(4) = O(d^2)$.

**총 연산**: 5 variable-to-factor ($O(2)$ each) + 5 factor-to-variable ($O(4)$ each) = **30 operations**.

Brute force 대비: 결합분포 $2^3 = 8$ entries, 각 주변 $O(4)$ 계산, 3 marginals → 24 operations. 비슷한 수준.

하지만 $n$-chain으로 확장하면 BP: $O(n \cdot d^2)$, brute force: $O(d^n)$. BP의 **exponential speedup**.

</details>

**문제 2** (심화): BP의 수렴은 tree에서 **tree의 diameter 이하의 iteration**에서 완료됨을 증명하라.

<details>
<summary>힌트 및 해설</summary>

**Claim**: Tree의 diameter $D = \max_{u, v} \text{dist}(u, v)$. BP는 **$\lceil D/2 \rceil + 1$ iteration** 이내에 수렴.

**증명 스케치**:

Inward + outward pass의 총 깊이가 $D$. 각 iteration에서 leaf에서 root 방향 + root에서 leaf 방향으로 각 edge를 한 번씩 통과.

Parallel schedule에서 매 iteration 모든 메시지를 동시 업데이트. Tree의 leaf에서 시작한 정보가 root까지 도달하는 데 $D/2$ iteration, root에서 반대 leaf까지 $D/2$ 더 → $D$ iteration.

더 정확히: 첫 iteration에서 leaf의 메시지는 이미 최종값. 각 iteration마다 "correct" 메시지의 반경이 1씩 증가. 따라서 $\lceil D/2 \rceil + 1$ 후 모든 메시지 수렴.

구체적 예: chain of length $n$ (diameter $= n - 1$). BP 수렴은 약 $n/2$ iteration. 실제로 위 NumPy 예시에서 3-chain은 3 iteration 후 수렴.

</details>

**문제 3** (AI 연결): Transformer의 multi-layer self-attention을 "loopy graph 위의 BP의 iterated approximation"으로 해석할 수 있는 이유를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**구조적 대응**:

1. Transformer attention matrix $A_{ij} = \text{softmax}(q_i k_j^T / \sqrt{d})$는 **soft edge weights** — position $i$가 $j$를 얼마나 "believe"하는지
2. Attention output $h_i' = \sum_j A_{ij} v_j$는 **soft message aggregation** — 모든 이웃의 value를 weighted sum (BP의 sum과 대응)
3. Layer을 쌓는 것 = **iterative BP** — 각 layer가 one round of belief updates

**Loopy graph 구조**:
- Variable nodes: 토큰 위치
- Fully connected factor graph (causal는 upper-triangular)
- **모든 pair가 edge** → 극단적으로 loopy

**Loopy BP의 행동과 비슷함**:
- 수렴 보장 없음 — Transformer의 **layer 수**가 결정 (보통 12~96)
- 근사 — perfect marginal 아님, 하지만 실전에서 우수
- Damping / residual connection — loopy BP의 damping과 유사한 역할

**차이점**:
- BP: fixed message function (sum-product)
- Transformer: **learned** message function (MLP, layer norm, etc.)
- BP: 수학적으로 Bethe free energy에 대응 (Ch2-05, Ch6-02)
- Transformer: 학습된 objective (language modeling loss)

**의미**: Transformer의 성공은 "fully-connected loopy graph 위의 learned approximate BP"가 훌륭한 inference primitive임을 시사. 이는 **Ch7-05의 핵심**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Factor Graph의 정의와 통합 표현](./01-factor-graph-definition.md) | [📚 README](../README.md) | [03. Max-Product Algorithm과 MAP Inference ▶](./03-max-product-algorithm.md) |

</div>
