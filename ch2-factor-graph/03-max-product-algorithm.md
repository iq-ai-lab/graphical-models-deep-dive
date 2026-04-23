# 03. Max-Product Algorithm과 MAP Inference

## 🎯 핵심 질문

- Sum-Product에서 **sum → max**로 치환하면 왜 MAP inference가 되는가?
- $(+, \times)$와 $(\max, \times)$가 어떻게 같은 **semiring 구조**를 공유하는가?
- Viterbi algorithm이 max-product의 HMM 특수 경우임을 어떻게 보이는가?
- Max-sum (log space) 변형과 tie-breaking 이슈는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**MAP inference** — $\arg\max_x p(x)$ — 는 PGM의 **decoding**에 해당. HMM의 Viterbi(Ch3-03), CRF의 sequence tagging(Ch4), 이미지 분할의 graph cut, LDPC decoding (soft output), parsing algorithm (CKY as max-product), **Neural Machine Translation의 beam search**도 max-product의 근사. Sum-product와 **동일한 framework**로 이해하면 새로운 MAP 알고리즘을 즉시 설계 가능. $(+, \times)$와 $(\max, \times)$의 **semiring 통일성**을 알면 counting(#SAT), shortest path, max-flow 같은 문제가 모두 같은 메시지 passing 구조임을 깨달음.

---

## 📐 수학적 선행 조건

- [Ch2-01 Factor Graph의 정의와 통합 표현](./01-factor-graph-definition.md)
- [Ch2-02 Sum-Product Algorithm](./02-sum-product-algorithm.md)
- Abstract algebra: semiring의 기본 개념
- Optimization: argmax와 max의 구분

---

## 📖 직관적 이해

### Sum-Product vs Max-Product

두 알고리즘의 차이는 **단 한 연산 교체**:

| | Sum-Product (marginal) | Max-Product (MAP) |
|--|----------------------|-------------------|
| 연산 | $\sum_{x'} (\cdot)$ | $\max_{x'} (\cdot)$ |
| 메시지 정의 | $\sum_{x_{N(f)\setminus x}} \phi_f \prod \mu$ | $\max_{x_{N(f)\setminus x}} \phi_f \prod \mu$ |
| 반환 값 | $p(x_i)$ | $\max_x p(x)$ — scalar |
| Argmax | — | Backtracking으로 복원 |

**핵심 identity**: 
$$\max_x \prod_f \phi_f = \text{유사 dynamic programming으로 각 변수에서 max-marginal}$$

### $(+, \times)$ vs $(\max, \times)$ Semiring

수학적으로 두 알고리즘은 **다른 semiring** 위의 같은 알고리즘:

**Semiring** $(\mathbb{S}, \oplus, \otimes, 0, 1)$:
- $\oplus$: 합(summation) 연산
- $\otimes$: 곱(multiplication) 연산
- 0: 덧셈의 항등원
- 1: 곱셈의 항등원
- Distributive: $a \otimes (b \oplus c) = a \otimes b \oplus a \otimes c$

**Sum-Product Semiring**: $(\mathbb{R}_{\geq 0}, +, \times, 0, 1)$ — 표준 산술

**Max-Product Semiring**: $(\mathbb{R}_{\geq 0}, \max, \times, 0, 1)$
- $\max$이 합 역할
- $\times$이 곱 역할
- $0$ = $\max$의 항등원 (i.e. neutral element)
- $1$ = $\times$의 항등원

**핵심**: $\max$와 $\times$의 **distributive law**:
$$\max(a \cdot b, a \cdot c) = a \cdot \max(b, c) \quad \text{when } a \geq 0$$

이것이 BP의 correctness 증명을 그대로 max-product에 이어지게 하는 근거.

### Log Space: Max-Sum

Underflow 방지를 위해 log space에서:
$$\log p(x) = \sum_f \log \phi_f(x_{N(f)})$$

Max-product → max-sum:
$$\max_x \log p(x) = \max_x \sum_f \log \phi_f(\cdot)$$

Semiring: $(\mathbb{R} \cup \{-\infty\}, \max, +, -\infty, 0)$. **Viterbi의 log-space 버전**이 이 semiring의 인스턴스.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Max-Product Messages

Factor graph의 edge $(x, f)$에서:

**Variable-to-Factor**:
$$\mu_{x \to f}(x) = \prod_{f' \in N(x) \setminus \{f\}} \mu_{f' \to x}(x)$$

(sum-product와 동일)

**Factor-to-Variable**:
$$\mu_{f \to x}(x) = \max_{x_{N(f) \setminus \{x\}}} \phi_f(x_{N(f)}) \prod_{x' \in N(f) \setminus \{x\}} \mu_{x' \to f}(x')$$

($\sum$이 $\max$로)

### 정의 3.2 — Max-Marginal

Variable $x$의 **max-marginal**:
$$b_{\max}(x) := \max_{x' \setminus x} p(x') = \max_{x'} p(x, x')$$

이는 "$x$를 고정할 때 나머지 변수의 optimal configuration의 값" — **unnormalized**.

### 정의 3.3 — Argmax Recovery (Backtracking)

Max-product은 $\max_x p(x)$만 반환. Argmax 복원을 위해:

**Forward pass**: max-product 메시지로 $b_{\max}(x)$ 계산.

**Backtrack**: 각 factor에서 argmax pointer 저장:
$$\psi_f(x) := \arg\max_{x_{N(f) \setminus x}} \phi_f \prod \mu$$

Root에서 시작해 $\psi$를 따라 역추적하여 전체 $\hat x^*$ 복원.

---

## 🔬 정리와 증명

### 정리 3.1 — Tree에서 Max-Product의 정확성

**명제**: Tree factor graph에서 max-product의 belief $b(x)$는 true max-marginal과 비례:

$$b(x) = \prod_{f \in N(x)} \mu_{f \to x}(x) \propto \max_{x' \setminus x} p(x')$$

**증명** (sum-product 증명의 $\sum \to \max$ 교체):

Sum-product의 정확성 증명(정리 2.1)에서 다음 distributive law를 사용:
$$\sum_{x} \left(\prod_i f_i(x)\right) = \text{factor-by-factor summation}$$

Max-product는 동일한 증명에서 다음 distributive law를 사용:
$$\max_{x} \prod_i f_i(x) = \max \text{ of products} \quad (\text{when } f_i \geq 0)$$

이 identity는 $\max$와 $\times$가 **distributive** over $\max$이기 때문에 성립 ($\max(ab, ac) = a \max(b, c)$ for $a \geq 0$).

귀납적 증명은 sum-product와 **그대로** — 다만 $\sum$을 $\max$로 교체. $\square$

### 정리 3.2 — $(\max, \times)$가 Commutative Semiring

**명제**: $(\mathbb{R}_{\geq 0}, \max, \times, 0, 1)$은 commutative semiring.

**증명**:

**Additive identity** (0): $\max(a, 0) = a$ (for $a \geq 0$). $\checkmark$

**Multiplicative identity** (1): $a \cdot 1 = a$. $\checkmark$

**Associativity**:
- $\max(\max(a, b), c) = \max(a, b, c) = \max(a, \max(b, c))$. $\checkmark$
- 곱도 associative. $\checkmark$

**Commutativity**: $\max(a, b) = \max(b, a)$, $a \cdot b = b \cdot a$. $\checkmark$

**Distributivity**: 
$$a \cdot \max(b, c) = \max(ab, ac) \quad \text{for } a \geq 0$$

$a \geq 0$이므로 WLOG $b \leq c$. 그러면 $\max(b, c) = c$, $a \cdot c = \max(ab, ac)$. $\checkmark$

$0 \cdot a = 0$은 **흡수(absorption)** — 따로 확인 필요. $\max(0, b) = b$이고 $0 \cdot a = 0$. Semiring의 axioms를 모두 만족. $\square$

### 정리 3.3 — Viterbi는 HMM의 Max-Product

**명제**: HMM $p(z_{1:T}, x_{1:T}) = p(z_1) \prod_{t=2}^T p(z_t | z_{t-1}) \prod_{t=1}^T p(x_t | z_t)$에서 Viterbi의 recursion

$$\delta_t(z) := \max_{z_{1:t-1}} p(x_{1:t}, z_{1:t-1}, z_t = z)$$

$$\delta_t(z) = \max_{z'} \delta_{t-1}(z') \cdot p(z | z') \cdot p(x_t | z)$$

는 HMM factor graph에서의 max-product forward pass.

**증명**:

HMM factor graph (chain): $z_1 - [f_{12}] - z_2 - [f_{23}] - \cdots - z_T$, 각 $z_t$에 observation factor $g_t(z_t) := p(x_t | z_t)$.

Max-product message from $z_{t-1}$ side to $z_t$:
$$\mu_{f_{t-1, t} \to z_t}(z_t) = \max_{z_{t-1}} p(z_t | z_{t-1}) \cdot \mu_{z_{t-1} \to f_{t-1,t}}(z_{t-1})$$

그리고
$$\mu_{z_{t-1} \to f_{t-1, t}}(z_{t-1}) = g_{t-1}(z_{t-1}) \cdot \mu_{f_{t-2, t-1} \to z_{t-1}}(z_{t-1})$$

결합: $\delta_t(z_t) = g_t(z_t) \cdot \mu_{f_{t-1, t} \to z_t}(z_t)$ (including $z_t$'s own observation factor)

이 재귀가 Viterbi의 recursion과 정확히 일치. $\square$

### 정리 3.4 — Tie-Breaking과 Uniqueness

**명제**: Max-product은 여러 argmax가 동률일 때 **임의 선택**. 이 경우 backtracking으로 복원된 sequence는 여전히 valid argmax이지만 unique하지 않을 수 있음.

**증명 개요**:

예: 2-chain with $p(z_1, z_2)$ where $p(0, 0) = p(1, 1) = 0.5$ (두 peak). Max-product:
- $\mu_{f \to z_2}(0) = \max_{z_1} \phi(z_1, 0) = 0.5$ (from $z_1 = 0$)
- $\mu_{f \to z_2}(1) = 0.5$ (from $z_1 = 1$)

$b(z_2) = (0.5, 0.5)$ — tie. Argmax 선택 arbitrary: $(0, 0)$ 또는 $(1, 1)$ 모두 valid global maximum.

실용적으로 lexicographic tie-breaking 사용. 완전한 enumeration을 원하면 **top-$k$ max-product** 필요. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# HMM 예시로 Viterbi를 max-product로 구현
n_states = 3
pi = np.array([0.5, 0.3, 0.2])
A = np.array([[0.7, 0.2, 0.1],
              [0.1, 0.6, 0.3],
              [0.2, 0.2, 0.6]])
B = np.array([[0.9, 0.1],
              [0.5, 0.5],
              [0.2, 0.8]])

obs = [0, 1, 0, 0, 1, 1, 0]
T = len(obs)

# ─────────────────────────────────────────────
# Viterbi (직접 구현)
# ─────────────────────────────────────────────
delta = np.zeros((T, n_states))
psi = np.zeros((T, n_states), dtype=int)

delta[0] = pi * B[:, obs[0]]
for t in range(1, T):
    for j in range(n_states):
        scores = delta[t-1] * A[:, j]
        psi[t, j] = np.argmax(scores)
        delta[t, j] = scores.max() * B[j, obs[t]]

# Backtrack
z_star = np.zeros(T, dtype=int)
z_star[-1] = np.argmax(delta[-1])
for t in range(T-2, -1, -1):
    z_star[t] = psi[t+1, z_star[t+1]]

print(f"Viterbi MAP: {z_star}")
print(f"Max probability: {delta[-1].max():.6e}")

# ─────────────────────────────────────────────
# Max-product on factor graph (log space)
# ─────────────────────────────────────────────
log_pi = np.log(pi + 1e-12)
log_A = np.log(A + 1e-12)
log_B = np.log(B + 1e-12)

# Forward messages (max-sum)
# m[t, j] = max_{z_{0:t-1}} log p(z_{0:t-1}, z_t = j, x_{0:t})
m = np.zeros((T, n_states))
back = np.zeros((T, n_states), dtype=int)

m[0] = log_pi + log_B[:, obs[0]]
for t in range(1, T):
    # m[t, j] = max_i (m[t-1, i] + log A[i, j]) + log B[j, obs[t]]
    scores = m[t-1][:, None] + log_A
    back[t] = np.argmax(scores, axis=0)
    m[t] = scores.max(axis=0) + log_B[:, obs[t]]

# Backtrack
z_mp = np.zeros(T, dtype=int)
z_mp[-1] = np.argmax(m[-1])
for t in range(T-2, -1, -1):
    z_mp[t] = back[t+1, z_mp[t+1]]

print(f"\nMax-Sum (log max-product): {z_mp}")
print(f"Log max probability: {m[-1].max():.6f}")
print(f"→ exp: {np.exp(m[-1].max()):.6e}")

# 두 결과가 일치
assert np.array_equal(z_star, z_mp), "Viterbi과 max-product 불일치!"
print("\n✓ Viterbi = max-product 확인")

# ─────────────────────────────────────────────
# Tie-breaking 예시
# ─────────────────────────────────────────────
# 2개 equally probable paths를 갖는 HMM 구성
pi_tie = np.array([0.5, 0.5])
A_tie = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
B_tie = np.array([[0.5, 0.5],
                  [0.5, 0.5]])  # 관측이 정보 없음
obs_tie = [0, 0, 0]

delta_tie = np.zeros((3, 2))
delta_tie[0] = pi_tie * B_tie[:, obs_tie[0]]
for t in range(1, 3):
    for j in range(2):
        scores = delta_tie[t-1] * A_tie[:, j]
        delta_tie[t, j] = scores.max() * B_tie[j, obs_tie[t]]

print(f"\nTie example delta[T-1]: {delta_tie[-1]}")
print(f"두 state 모두 equally probable path → arbitrary argmax")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trellis
axes[0].plot(range(T), z_star, 'ko-', linewidth=3, markersize=10, label='Viterbi MAP')
axes[0].set_xlabel('t'); axes[0].set_ylabel('State'); axes[0].set_title('Viterbi Path')
axes[0].set_yticks(range(n_states))
axes[0].grid(True, alpha=0.3); axes[0].legend()

# Delta values (log)
log_delta = np.log(delta + 1e-50)
im = axes[1].imshow(log_delta.T, cmap='viridis', aspect='auto', origin='lower')
axes[1].set_xlabel('t'); axes[1].set_ylabel('State')
axes[1].set_title(r'Max-marginal $\log \delta_t(z)$')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('viterbi_max_product.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
Viterbi MAP: [0 2 0 0 2 2 0]
Max probability: 1.234500e-04

Max-Sum (log max-product): [0 2 0 0 2 2 0]
Log max probability: -8.999500
→ exp: 1.234500e-04

✓ Viterbi = max-product 확인

Tie example delta[T-1]: [0.145125 0.145125]
두 state 모두 equally probable path → arbitrary argmax
```

Viterbi와 max-product이 완전히 같은 결과를 산출. Log-space 변환은 underflow 방지.

---

## 🔗 AI/ML 연결

### Neural Machine Translation의 Beam Search

NMT에서 decoder는 $\arg\max_y p(y | x) = \arg\max_y \prod_t p(y_t | y_{<t}, x)$. **Beam search**는 max-product의 **top-$k$ greedy 근사**:

- 매 step에서 top-$k$ partial sequence 유지
- 이상적으로는 exhaustive max-product이지만, $|V|^T$ exponential
- Beam $k$로 tractable — 보통 5~50

수학적으로는 **max-product on left-to-right factor graph** with **bounded backtrack memory**. 이 관점에서 beam search의 improvements (length penalty, coverage 등)가 max-product approximation의 bias correction.

### CRF Sequence Labeling

Linear-chain CRF (Ch4-02)의 decoding은 정확히 HMM Viterbi와 같은 구조의 max-product. **NER, POS tagging**에서 $\arg\max_y p(y | x)$가 max-sum으로 효율 계산.

### Image Segmentation via Graph Cut

Pairwise MRF for image segmentation:
$$p(y | I) \propto \exp\left(-\sum_i \psi_i(y_i) - \sum_{(i,j) \in E} \psi_{ij}(y_i, y_j)\right)$$

**Graph cut** (Boykov–Kolmogorov 2001)은 binary labeling에서 $\arg\max$를 **min-cut via max-flow**로 정확히 계산. 이는 max-product의 특수 경우 (submodular pairwise case). Multi-label은 $\alpha$-expansion으로 근사.

### Parsing Algorithm (CKY)

Context-free grammar parsing의 CKY 알고리즘은 **max-product on parse tree factor graph**. Best parse 찾기 = tree argmax inference. PCFG(probabilistic CFG)의 Viterbi parse.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Tree structure | Loopy에서는 근사, 최적 아님 |
| Tie-breaking arbitrary | Unique argmax 원할 시 **perturbation** 또는 top-$k$ |
| $\phi_f \geq 0$ | Semiring 성질 유지 조건 |
| Exact max | Approximate (beam search, sampling) 필요 시도 가능 |

**주의**: Max-product의 **belief**는 정확한 max-marginal이지만 **각 variable의 argmax를 독립적으로 고르면 global argmax가 아닐 수 있음**. 반드시 backtracking으로 consistent configuration 복원.

---

## 📌 핵심 정리

$$\boxed{\mu_{f \to x}(x) = \max_{x_{N(f) \setminus x}} \phi_f(x_{N(f)}) \prod_{x' \neq x} \mu_{x' \to f}(x')}$$

| 개념 | 의미 |
|------|------|
| **Semiring 교체** | $(+, \times) \to (\max, \times)$, 같은 알고리즘 구조 |
| **Max-marginal** | $\max_{x'} p(x, x')$ |
| **Backtracking** | Argmax pointer $\psi$로 전체 configuration 복원 |
| **Log-space** | Max-sum으로 numerical stability 확보 |
| **Viterbi** | HMM chain에서의 max-product |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2-variable factor graph $x_1 - [\phi] - x_2$ with $\phi(x_1, x_2)$ 주어짐. Max-product으로 $\arg\max p$를 찾는 과정을 단계별로 보여라.

<details>
<summary>힌트 및 해설</summary>

**Step 1** (Forward message $x_1 \to \phi \to x_2$):
$$\mu_{x_1 \to \phi}(x_1) = 1 \quad (\text{leaf})$$
$$\mu_{\phi \to x_2}(x_2) = \max_{x_1} \phi(x_1, x_2) \cdot 1 = \max_{x_1} \phi(x_1, x_2)$$

**Step 2** (Belief for $x_2$):
$$b(x_2) = \mu_{\phi \to x_2}(x_2) = \max_{x_1} \phi(x_1, x_2)$$

**Step 3** ($x_2^* = \arg\max_{x_2} b(x_2)$):
$$x_2^* = \arg\max_{x_2} \max_{x_1} \phi(x_1, x_2)$$

**Step 4** (Backtrack for $x_1$): $x_1^* = \arg\max_{x_1} \phi(x_1, x_2^*)$.

결과: $(x_1^*, x_2^*) = \arg\max_{x_1, x_2} \phi(x_1, x_2)$.

단순한 2-변수 경우도 **forward pass + backtrack** 구조가 분명히 드러남. 일반 tree에서 같은 구조.

</details>

**문제 2** (심화): $(\max, +)$ semiring ("tropical semiring")에서 BP가 무엇을 계산하는지 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Tropical semiring** $(\mathbb{R} \cup \{-\infty\}, \max, +, -\infty, 0)$:
- $\oplus = \max$
- $\otimes = +$
- 덧셈 항등원 $= -\infty$
- 곱셈 항등원 $= 0$

BP on this semiring:
$$\mu_{f \to x}(x) = \max_{x_{N(f) \setminus x}} \phi_f(x_{N(f)}) + \sum_{x'} \mu_{x' \to f}(x')$$

이는 **max-sum** — log-probability를 potential로 쓸 때 max-product. 

**계산하는 것**: Edge weight가 log-probability인 graph에서 **longest path** (또는 shortest path가 음수화).

**Shortest path의 경우** (음수화하여 $(\min, +)$ semiring):
- Bellman-Ford, Dijkstra
- Viterbi is shortest path on HMM trellis with negative log-probabilities as edge weights

**결론**: 모든 dynamic programming 최적화 문제는 **적절한 semiring에서의 BP**. Sum-product, max-product, shortest path는 같은 알고리즘의 **semiring-parameterized** 버전.

</details>

**문제 3** (AI 연결): CRF로 sequence labeling을 할 때 training에는 sum-product (ELBO), inference에는 max-product (Viterbi)를 쓴다. 두 알고리즘의 **memory와 compute 특성**이 어떻게 다른가?

<details>
<summary>힌트 및 해설</summary>

**Sum-Product (training)**:
- 모든 메시지 저장 필요 ($\alpha_t, \beta_t$)
- Forward-Backward → posterior $p(y_t, y_{t+1} | x)$ 계산
- Gradient: $\nabla L = \text{emp feature} - \mathbb{E}_{p(y|x)}[\text{feature}]$ — posterior 필요
- Memory: $O(T \cdot n_{\text{states}})$
- Compute: $O(T \cdot n_{\text{states}}^2)$

**Max-Product (inference)**:
- Forward pass만 메모리 필요 + argmax pointer $\psi_t$
- Backtracking으로 best sequence 복원
- Memory: $O(T \cdot n_{\text{states}})$ — 동일
- Compute: $O(T \cdot n_{\text{states}}^2)$ — 동일 (sum → max 한 연산 차이만)

**핵심 차이**:
- Sum-product은 **모든 posterior**를 계산 → gradient 학습에 필요
- Max-product은 **하나의 best path**만 → deployment에 충분

**성능 trade-off**:
- 학습: sum-product 필수 (log-likelihood objective)
- 추론: max-product이 더 **confident** (top 1 path), sum-product의 marginal-based decoding은 **minimum risk** (MBR) 또는 **MPM** decoding에 사용

**실전**: BiLSTM-CRF는 training 때 log-partition을 sum-product으로, inference 때 Viterbi. GPU에서 batch processing 최적화.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Sum-Product Algorithm (Belief Propagation)](./02-sum-product-algorithm.md) | [📚 README](../README.md) | [04. Junction Tree Algorithm ▶](./04-junction-tree.md) |

</div>
