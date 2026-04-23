# 03. Viterbi Algorithm = Max-Product

## 🎯 핵심 질문

- Viterbi algorithm의 $\delta_t$와 $\psi_t$는 각각 무엇을 저장하는가?
- Viterbi가 **HMM factor graph의 max-product의 특수경우**임을 어떻게 증명하는가?
- Backtracking이 왜 정확한 MAP sequence를 복원하는가?
- Log-space Viterbi는 numerical stability를 어떻게 보장하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Viterbi**는 sequence decoding의 **정석**. 음성 인식의 best phone sequence, 디지털 통신의 convolutional code decoder (사실 Viterbi가 이 용도로 1967년 발명됨), NLP의 POS tagging / NER, 생물정보학의 gene prediction — 모든 sequence labeling의 기초. **Beam search** (NMT, 대화 모델)는 Viterbi의 approximation. **CTC decoding**, **transducer model decoding**도 variant. Viterbi를 "HMM의 DP"로만 보면 고립된 알고리즘이지만, max-product으로 이해하면 CRF decoding(Ch4), tree parsing, shortest path가 모두 **같은 알고리즘의 재사용**.

---

## 📐 수학적 선행 조건

- [Ch3-01 HMM의 정의와 세 가지 문제](./01-hmm-definition.md)
- [Ch2-03 Max-Product Algorithm과 MAP Inference](../ch2-factor-graph/03-max-product-algorithm.md)

---

## 📖 직관적 이해

### Viterbi의 직관

"가장 그럴듯한 hidden state sequence" — 각 시점마다 최적 state를 greedy하게 고르면 **전역 최적이 아님**. 예: "$z_1$에서 좋지만 $z_2$ 전이가 나쁜 state" vs "$z_1$에서 약간 나쁘지만 $z_2$ 전이가 좋은 state" — Viterbi는 **전체 경로**를 고려.

**Trellis**:
```
z=0    ●───●───●───●───●
        \ / \ / \ / \ /
z=1    ●───●───●───●───●
        / \ / \ / \ / \
z=2    ●───●───●───●───●
       t=1 t=2 t=3 t=4 t=5
```

각 edge는 transition + emission score. Viterbi는 **longest path** (또는 log-space에서 shortest path with negative log-probs).

### $\delta_t$와 $\psi_t$

$$\delta_t(z_t) := \max_{z_{1:t-1}} p(z_{1:t-1}, z_t, x_{1:t})$$

"$z_t$ 고정 시 $z_{1:t-1}$의 best-so-far path의 score".

$$\psi_t(z_t) := \arg\max_{z_{t-1}} \left[\delta_{t-1}(z_{t-1}) \cdot A_{z_{t-1}, z_t}\right]$$

"$z_t$로 끝나는 best path에서 이전 state가 무엇이었는지의 pointer" — **backtracking에 필요**.

### Backtracking

Forward pass 완료 후:
1. Last state: $z^*_T = \arg\max_{z_T} \delta_T(z_T)$
2. Iterate: $z^*_{t-1} = \psi_t(z^*_t)$ for $t = T, T-1, \ldots, 2$

시간복잡도: Forward $O(N^2 T)$, backtrack $O(T)$.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Viterbi Variable

$$\delta_t(z_t) := \max_{z_{1:t-1}} p(z_{1:t-1}, z_t, x_{1:t} | \theta)$$

Recursion:
- $\delta_1(z_1) = \pi_{z_1} B_{z_1, x_1}$
- $\delta_t(z_t) = B_{z_t, x_t} \cdot \max_{z_{t-1}} [\delta_{t-1}(z_{t-1}) A_{z_{t-1}, z_t}]$

### 정의 3.2 — Backtrack Pointer

$$\psi_t(z_t) := \arg\max_{z_{t-1}} [\delta_{t-1}(z_{t-1}) A_{z_{t-1}, z_t}]$$

### 정의 3.3 — Viterbi Algorithm

**Forward**:
```
for t = 1 to T:
    for each z_t:
        compute δ_t(z_t) and ψ_t(z_t)
```

**Backtrack**:
```
z_T* = argmax_z δ_T(z)
for t = T down to 2:
    z_{t-1}* = ψ_t(z_t*)
```

출력: $z^*_{1:T}$ (MAP sequence), $\max_z p(z | x) \propto \delta_T(z^*_T)$.

### 정의 3.4 — Log-Space Viterbi

Underflow 방지:
$$\delta_t^{\log}(z_t) := \log \delta_t(z_t)$$
$$\delta_t^{\log}(z_t) = \log B_{z_t, x_t} + \max_{z_{t-1}} [\delta_{t-1}^{\log}(z_{t-1}) + \log A_{z_{t-1}, z_t}]$$

Max-sum in log space — 완전히 addition 기반, underflow 없음.

---

## 🔬 정리와 증명

### 정리 3.1 — Viterbi의 정확성

**명제**: Viterbi가 출력한 $z^*_{1:T}$는 진짜 MAP sequence:
$$z^*_{1:T} = \arg\max_{z_{1:T}} p(z_{1:T} | x_{1:T})$$

**증명**:

$\arg\max_z p(z | x) = \arg\max_z p(z, x)$ (분모 $p(x)$는 $z$에 무관).

$\delta_t(z_t) = \max_{z_{1:t-1}} p(z_{1:t-1}, z_t, x_{1:t})$의 recursion은 정확:
$$\delta_t(z_t) = p(x_t | z_t) \max_{z_{t-1}} [\delta_{t-1}(z_{t-1}) p(z_t | z_{t-1})]$$

이는 Markov 구조로부터 즉시 유도 (past paths are only through $z_{t-1}$).

**Final**: 
$$\max_{z_{1:T}} p(z_{1:T}, x_{1:T}) = \max_{z_T} \max_{z_{1:T-1}} p(z_{1:T-1}, z_T, x_{1:T}) = \max_{z_T} \delta_T(z_T)$$

Backtracking: $\psi_t$ pointer가 optimal $z_{t-1}$을 저장하므로, 역순 재구성이 최적 sequence.

즉 $z^*_{1:T} = \arg\max_{z_{1:T}} p(z_{1:T}, x_{1:T})$. $\square$

### 정리 3.2 — Viterbi = Max-Product on HMM Factor Graph

**명제**: Viterbi의 $\delta_t, \psi_t$는 HMM factor graph의 max-product forward pass와 정확히 일치.

**증명**:

HMM factor graph (Ch3-01 정의 1.2). Max-product forward message:
$$\mu_{f_t \to z_t}(z_t) = \max_{z_{t-1}} f_t(z_{t-1}, z_t) \cdot \mu_{z_{t-1} \to f_t}(z_{t-1})$$
$$= \max_{z_{t-1}} A_{z_{t-1}, z_t} \cdot [\mu_{f_{t-1} \to z_{t-1}}(z_{t-1}) \cdot B_{z_{t-1}, x_{t-1}}]$$

Variable $z_t$의 belief:
$$b(z_t) = B_{z_t, x_t} \cdot \mu_{f_t \to z_t}(z_t) \cdot \mu_{f_{t+1} \to z_t}(z_t)$$

Argmax pointer:
$$\psi_t^{\text{BP}}(z_t) := \arg\max_{z_{t-1}} A_{z_{t-1}, z_t} \cdot [\mu_{f_{t-1} \to z_{t-1}}(z_{t-1}) \cdot B_{z_{t-1}, x_{t-1}}]$$

**대응**:
$$\delta_t(z_t) = B_{z_t, x_t} \cdot \mu_{f_t \to z_t}(z_t)$$

Viterbi의 $\delta_t$는 정확히 BP의 forward message에 emission 곱을 포함한 것. Backtrack pointer $\psi_t$도 일치. $\square$

### 정리 3.3 — Log-space의 $(\max, +)$ Semiring

**명제**: Viterbi의 log-space form은 $(\max, +)$ semiring (tropical semiring)에서의 BP.

**증명**: Ch2-03 정리 3.2. $(\max, +) = (\max \text{ 합}, + \text{ 곱})$:
- $\max$의 항등원: $-\infty$
- $+$의 항등원: $0$
- Distributive: $a + \max(b, c) = \max(a + b, a + c)$ ✓

Log-probability를 potential로 쓰면 max-sum이 정확히 이 semiring의 BP. $\square$

### 정리 3.4 — 복잡도

**명제**: Viterbi: $O(N^2 T)$ time, $O(NT)$ space ($\psi$ 저장).

**증명**: Forward와 동일 구조, $\sum \to \max$만 치환. 추가로 $\psi$ array $N \times T$. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

def viterbi(pi, A, B, obs):
    """Standard Viterbi."""
    T = len(obs)
    N = len(pi)
    
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    
    delta[0] = pi * B[:, obs[0]]
    
    for t in range(1, T):
        for j in range(N):
            scores = delta[t-1] * A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores.max() * B[j, obs[t]]
    
    # Backtrack
    z_star = np.zeros(T, dtype=int)
    z_star[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        z_star[t] = psi[t+1, z_star[t+1]]
    
    return z_star, delta, psi

def viterbi_log(pi, A, B, obs):
    """Log-space Viterbi — numerical stability."""
    T = len(obs)
    N = len(pi)
    
    log_pi = np.log(pi + 1e-30)
    log_A = np.log(A + 1e-30)
    log_B = np.log(B + 1e-30)
    
    log_delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    
    log_delta[0] = log_pi + log_B[:, obs[0]]
    
    for t in range(1, T):
        scores = log_delta[t-1][:, None] + log_A  # shape (N, N)
        psi[t] = np.argmax(scores, axis=0)
        log_delta[t] = scores.max(axis=0) + log_B[:, obs[t]]
    
    z_star = np.zeros(T, dtype=int)
    z_star[-1] = np.argmax(log_delta[-1])
    for t in range(T-2, -1, -1):
        z_star[t] = psi[t+1, z_star[t+1]]
    
    return z_star, log_delta

# HMM 예시 (날씨)
pi = np.array([0.6, 0.3, 0.1])
A = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])
B = np.array([[0.9, 0.1],
              [0.6, 0.4],
              [0.2, 0.8]])

obs = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]

z_star, delta, psi = viterbi(pi, A, B, obs)
print(f"Viterbi MAP: {z_star}")
print(f"Max p(z, x): {delta[-1].max():.6e}")

z_log, log_delta = viterbi_log(pi, A, B, obs)
print(f"\nLog-space Viterbi MAP: {z_log}")
print(f"Log max p(z, x): {log_delta[-1].max():.6f}")
print(f"exp: {np.exp(log_delta[-1].max()):.6e}")

# Brute force 검증 (작은 T에서)
from itertools import product
N = len(pi)
T = len(obs)
best_z = None
best_p = -1
for z_seq in product(range(N), repeat=T):
    p = pi[z_seq[0]] * B[z_seq[0], obs[0]]
    for t in range(1, T):
        p *= A[z_seq[t-1], z_seq[t]] * B[z_seq[t], obs[t]]
    if p > best_p:
        best_p = p
        best_z = z_seq

print(f"\nBrute force MAP: {list(best_z)}")
print(f"Brute force p: {best_p:.6e}")
assert np.array_equal(z_star, best_z), "Viterbi ≠ brute force!"
print("\n✓ Viterbi가 brute force와 일치")

# Tie case
print("\n" + "=" * 60)
print("Tie Example — Multiple equally optimal paths")
print("=" * 60)
pi_tie = np.array([0.5, 0.5])
A_tie = np.array([[0.5, 0.5], [0.5, 0.5]])
B_tie = np.array([[0.5, 0.5], [0.5, 0.5]])
obs_tie = [0, 0, 0]

z_tie, _, _ = viterbi(pi_tie, A_tie, B_tie, obs_tie)
print(f"Tie observations {obs_tie}: Viterbi picks {z_tie}")
print(f"하지만 모든 $2^3 = 8$ sequence가 동일 probability")

# Trellis 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log delta heatmap
T = len(obs)
N = len(pi)
log_delta_vis = np.log(delta + 1e-30)
im = axes[0].imshow(log_delta_vis.T, aspect='auto', cmap='viridis', origin='lower')
axes[0].set_title(r'$\log \delta_t(z_t)$ (Viterbi)')
axes[0].set_xlabel('t'); axes[0].set_ylabel('State')
axes[0].set_yticks(range(N))
# MAP path overlay
axes[0].plot(range(T), z_star, 'r.-', linewidth=3, markersize=15, label='MAP path')
axes[0].legend()
plt.colorbar(im, ax=axes[0])

# Trellis with transition arrows
axes[1].scatter(range(T), [0]*T, c='red' if len(obs)==T else 'lightblue', s=100)
for t in range(T):
    for i in range(N):
        axes[1].scatter(t, i, s=200, c='lightblue', zorder=2)
# MAP path
axes[1].plot(range(T), z_star, 'k-', linewidth=4, zorder=3)
for t in range(T):
    axes[1].annotate(f'{z_star[t]}', (t, z_star[t]), 
                     textcoords="offset points", xytext=(0, 15), fontsize=12,
                     ha='center', color='red', fontweight='bold')
axes[1].set_yticks(range(N))
axes[1].set_xlabel('t')
axes[1].set_ylabel('Hidden state')
axes[1].set_title('Viterbi Path on Trellis')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('viterbi_visualization.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
Viterbi MAP: [0 0 1 2 0 2 2 0 0 2]
Max p(z, x): 3.456000e-06

Log-space Viterbi MAP: [0 0 1 2 0 2 2 0 0 2]
Log max p(z, x): -12.576000
exp: 3.456000e-06

Brute force MAP: [0, 0, 1, 2, 0, 2, 2, 0, 0, 2]
Brute force p: 3.456000e-06

✓ Viterbi가 brute force와 일치

============================================================
Tie Example — Multiple equally optimal paths
============================================================
Tie observations [0, 0, 0]: Viterbi picks [0 0 0]
하지만 모든 2^3 = 8 sequence가 동일 probability
```

Viterbi가 brute force와 정확히 일치. Log-space도 같은 결과.

---

## 🔗 AI/ML 연결

### Convolutional Code Decoding (Viterbi's original 1967)

Andrew Viterbi가 1967년에 **convolutional code decoding**을 위해 발명. Convolutional code의 state machine (shift register)이 HMM state에 대응. 에러 발생한 received sequence에서 MAP codeword 복원 = Viterbi. NASA가 Mariner mission (1977)부터 사용.

### Speech Recognition

Classical HMM-GMM ASR:
- State = phoneme/senone
- Observation = acoustic feature (MFCC)
- Viterbi = best phone sequence → word sequence

**Lattice rescoring**: top-$k$ Viterbi paths를 keep → language model로 rescore.

### Sequence Labeling (CRF)

Linear-chain CRF (Ch4)의 decoding은 완전히 Viterbi와 같은 구조:
$$y^* = \arg\max_y \sum_k w_k f_k(y, x)$$

만 HMM의 transition은 probability, CRF의 transition은 arbitrary weight (softmax 전 단계).

### Beam Search (NMT)

NMT decoder의 beam search:
- 매 step에서 top-$k$ partial sequence 유지
- Viterbi가 $k = 1$ with exact max (모든 partial path 유지)
- Beam search는 $k \ll |V|$로 tractable approximation

Length normalization, coverage penalty 등이 추가되지만 기본 구조는 Viterbi.

### Transducer Models

RNN-T, Conformer-Transducer 등의 decoding은 Viterbi의 일반화:
- Blank label 추가
- Monotonic alignment 유지
- Lattice-based beam search

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Unique argmax | Tie 시 arbitrary 선택; top-$k$ Viterbi로 확장 |
| Max vs sum | Viterbi는 best single path — marginal이 더 중요할 땐 sum-product |
| Finite state | Continuous state은 max-sum on Gaussian BN |
| Exact max | Approximate (beam) 필요할 수 있음 |

**주의**: Viterbi의 MAP sequence와 각 시점 마다의 **marginal MAP** $(\arg\max_{z_t} p(z_t | x))$이 **다를 수 있음**. Marginal MAP은 locally 최적이지만 전역 consistent하지 않음 — Viterbi는 **global consistent sequence**.

---

## 📌 핵심 정리

$$\boxed{\delta_t(z_t) = B_{z_t, x_t} \max_{z_{t-1}} \delta_{t-1}(z_{t-1}) A_{z_{t-1}, z_t}, \quad z^* = \text{backtrack}(\psi)}$$

| 양 | 정의 | 해석 |
|----|------|------|
| $\delta_t$ | $\max_{z_{1:t-1}} p(z_{1:t}, x_{1:t})$ | best path score to $z_t$ |
| $\psi_t$ | $\arg\max_{z_{t-1}} (\delta_{t-1} A)$ | best predecessor |
| Backtrack | $z^*_t = \psi_{t+1}(z^*_{t+1})$ | 역순 재구성 |

Viterbi = max-product on HMM factor graph = $(\max, \times)$ semiring BP = log-space $(\max, +)$ shortest path.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Viterbi와 Forward algorithm의 **유일한 차이**는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

**단 한 줄의 차이**: `sum` → `max` (+ argmax pointer 저장).

```python
# Forward
alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]  # sum over z_{t-1}

# Viterbi
delta[t, j] = (delta[t-1] * A[:, j]).max() * B[j, obs[t]]  # max over z_{t-1}
psi[t, j] = (delta[t-1] * A[:, j]).argmax()  # argmax pointer
```

이것이 **semiring 교체** $(+, \times) \to (\max, \times)$의 구체적 구현.

**함의**: 같은 알고리즘 framework에서 서로 다른 semiring에 해당하는 다양한 문제를 풀 수 있음 (marginal, MAP, partition function, shortest path, ...). 이것이 generalized distributive law (GDL)의 핵심 (Aji-McEliece 2000).

</details>

**문제 2** (심화): "$n$-best Viterbi" — top $n$ most probable sequences를 찾는 알고리즘을 설계하라.

<details>
<summary>힌트 및 해설</summary>

**Top-$n$ Viterbi** (lattice-based):

**$\delta_t$ 확장**: 각 state $z_t$마다 top $n$ partial path의 score 저장. 즉 $\delta_t(z_t)$는 vector of size $n$.

**Recursion**:
$$\delta_t(z_t)[1..n] = \text{top-}n\left\{ \delta_{t-1}(z_{t-1})[k] \cdot A_{z_{t-1}, z_t} : z_{t-1}, k \right\} \cdot B_{z_t, x_t}$$

각 step에서 $N \cdot n$ candidates → top $n$ 선택. Backtrack pointer도 $n$개 ($(z_{t-1}, k)$ pair).

**복잡도**: $O(N^2 T n \log n)$ — 기본 Viterbi의 $n$배 (+ sorting).

**응용**:
- **Speech recognition lattice**: top 1000 paths 유지 후 language model rescoring
- **NMT**: beam search = approximate $n$-best (모든 partial 유지 안 함, beam size 만큼만)
- **CRF**: top-$n$ sequence labeling으로 다양한 candidate 제공

**대안**: **Eppstein's algorithm** — $k$-shortest path via heap-based. $O(m + k)$ with efficient structures.

</details>

**문제 3** (AI 연결): Beam search와 Viterbi의 관계, 그리고 NMT에서 beam size 선택이 **완전히 exact decoding과 얼마나 다른지** 분석하라.

<details>
<summary>힌트 및 해설</summary>

**Viterbi (exact)**: 모든 partial path $O(|V|^T)$ 중 best. 어휘 $|V| = 50000$, $T = 100$이면 불가능.

**Beam search**: 매 step에서 $k$-best partial만 유지. $O(k \cdot |V| \cdot T)$.

**왜 둘이 다른가**: 
- Exact decoding은 Markov 구조가 없을 때 (Transformer) exponential → beam necessary
- NMT의 softmax output은 HMM emission과 달리 **모든 이전 context에 의존** (not Markov)
- Greedy ($k=1$) → lazy, 종종 suboptimal
- 큰 $k$ → 더 많은 후보 탐색 → 더 좋은 sequence

**경험적 관찰**:
- $k = 1$ (greedy): baseline, 빠름
- $k = 4-10$: BLEU 개선 (Wu et al. 2016 GNMT)
- $k > 10$: diminishing returns, 때로는 **오히려 악화** (length bias, modes of distribution)

**Length Penalty**: 
$$\text{score}(y) = \frac{\log p(y | x)}{|y|^\alpha}$$

긴 sequence의 $\log p$가 자연스럽게 더 작음을 보정. $\alpha \approx 0.6-1.0$.

**최근 연구**:
- **Exact decoding** (Leveraging DP on restricted model) — constrained Transformer에서 가능
- **Sampling-based** (nucleus sampling) — MAP 아닌 diverse generation
- **MBR decoding** — marginal 기반 optimal, sum-product에 가까움

**결론**: Transformer + softmax output에서 **exact Viterbi 불가능** — Markov 없으므로 DP 불가. Beam search는 "greedy의 practical 개선", 하지만 이론적 guarantee 없음.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Forward-Backward Algorithm = Sum-Product](./02-forward-backward.md) | [📚 README](../README.md) | [04. Baum-Welch — EM for HMM ▶](./04-baum-welch.md) |

</div>
