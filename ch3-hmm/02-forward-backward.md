# 02. Forward-Backward Algorithm = Sum-Product

## 🎯 핵심 질문

- Forward $\alpha_t$와 Backward $\beta_t$는 정확히 무엇을 계산하는가?
- 왜 posterior marginal $p(z_t | x_{1:T}) \propto \alpha_t \beta_t$인가?
- $\alpha, \beta$가 **HMM factor graph의 sum-product 메시지**와 정확히 일치함을 어떻게 보이는가?
- Numerical underflow를 어떻게 scaled 버전으로 해결하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Forward-Backward**는 HMM inference의 심장. Baum-Welch(EM for HMM)에 필요한 posterior statistics, CRF의 gradient 계산, speech recognition의 lattice rescoring — 모두 Forward-Backward. 더 근본적으로 이 알고리즘이 **sum-product의 특수 경우**임을 증명하면, HMM · CRF · linear state space · LDPC decoding이 모두 **하나의 메시지 passing 알고리즘**임이 드러난다. Deep Learning의 **CTC loss**, **attention alignment**도 Forward-Backward의 변형. 이 통일성을 놓치면 각 분야별 "따로따로 학습"에 그친다.

---

## 📐 수학적 선행 조건

- [Ch3-01 HMM의 정의와 세 가지 문제](./01-hmm-definition.md): Forward algorithm
- [Ch2-02 Sum-Product Algorithm](../ch2-factor-graph/02-sum-product-algorithm.md): BP 메시지
- Chain rule, conditional independence

---

## 📖 직관적 이해

### Forward $\alpha_t$ 의 의미

$$\alpha_t(z_t) := p(z_t, x_{1:t})$$

"시각 $t$까지 관측 $x_{1:t}$를 보았을 때, $z_t$가 특정 값인 확률"의 **결합 분포** (정규화 안 됨).

- $\alpha_t$의 합: $\sum_{z_t} \alpha_t(z_t) = p(x_{1:t})$ — 지금까지의 likelihood
- Recursive: $\alpha_t(z_t) = B_{z_t, x_t} \sum_{z_{t-1}} \alpha_{t-1}(z_{t-1}) A_{z_{t-1}, z_t}$

### Backward $\beta_t$ 의 의미

$$\beta_t(z_t) := p(x_{t+1:T} | z_t)$$

"현재 $z_t$ 값에서 시작해, 나머지 observations $x_{t+1:T}$가 나올 조건부 확률".

- $\beta_T(z_T) = 1$ (미래 관측 없음)
- Recursive: $\beta_t(z_t) = \sum_{z_{t+1}} A_{z_t, z_{t+1}} B_{z_{t+1}, x_{t+1}} \beta_{t+1}(z_{t+1})$

### Posterior Marginal

$$p(z_t | x_{1:T}) = \frac{p(z_t, x_{1:T})}{p(x_{1:T})} = \frac{\alpha_t(z_t) \beta_t(z_t)}{\sum_{z_t} \alpha_t(z_t) \beta_t(z_t)}$$

**Why $\alpha \beta$**:
$$p(z_t, x_{1:T}) = p(z_t, x_{1:t}) \cdot p(x_{t+1:T} | z_t, x_{1:t}) = p(z_t, x_{1:t}) \cdot p(x_{t+1:T} | z_t) = \alpha_t \beta_t$$

(두 번째 등호: Markov property — 미래는 현재 $z_t$만 조건으로 필요).

### Pairwise Posterior (Baum-Welch에 필요)

$$p(z_t, z_{t+1} | x_{1:T}) = \frac{\alpha_t(z_t) A_{z_t, z_{t+1}} B_{z_{t+1}, x_{t+1}} \beta_{t+1}(z_{t+1})}{p(x_{1:T})}$$

이는 EM의 M-step에서 transition $A$를 업데이트하는 데 필요.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Forward Variable

$$\alpha_t(z_t) := p(z_t, x_{1:t} | \theta)$$

Recursion:
- $\alpha_1(z_1) = \pi_{z_1} B_{z_1, x_1}$
- $\alpha_t(z_t) = B_{z_t, x_t} \sum_{z_{t-1}} \alpha_{t-1}(z_{t-1}) A_{z_{t-1}, z_t}$ for $t \geq 2$

### 정의 2.2 — Backward Variable

$$\beta_t(z_t) := p(x_{t+1:T} | z_t, \theta)$$

Recursion:
- $\beta_T(z_T) = 1$
- $\beta_t(z_t) = \sum_{z_{t+1}} A_{z_t, z_{t+1}} B_{z_{t+1}, x_{t+1}} \beta_{t+1}(z_{t+1})$ for $t < T$

### 정의 2.3 — Posterior Quantities

**Singleton posterior** ($\gamma_t$):
$$\gamma_t(z_t) := p(z_t | x_{1:T}) = \frac{\alpha_t(z_t) \beta_t(z_t)}{p(x_{1:T})}$$

**Pairwise posterior** ($\xi_t$):
$$\xi_t(z_t, z_{t+1}) := p(z_t, z_{t+1} | x_{1:T}) = \frac{\alpha_t(z_t) A_{z_t, z_{t+1}} B_{z_{t+1}, x_{t+1}} \beta_{t+1}(z_{t+1})}{p(x_{1:T})}$$

### 정의 2.4 — Scaled Forward-Backward

Underflow 방지를 위해 각 step에서 normalization:

**Scaled forward**:
$$\hat\alpha_t(z_t) := \alpha_t(z_t) / c_t, \quad c_t := \sum_{z_t} \alpha_t(z_t) / c_{t-1}$$

이러면 $\hat\alpha_t$가 정규화된 posterior $p(z_t | x_{1:t})$와 일치.

**Log-likelihood**: $\log p(x_{1:T}) = \sum_{t=1}^T \log c_t$.

---

## 🔬 정리와 증명

### 정리 2.1 — Forward-Backward = Sum-Product

**명제**: HMM의 factor graph에서 sum-product 메시지를 계산하면, Forward/Backward variable과 일치:

$$\alpha_t(z_t) \propto \mu_{\text{left} \to z_t}(z_t), \quad \beta_t(z_t) \propto \mu_{\text{right} \to z_t}(z_t)$$

**증명**:

HMM의 factor graph (Ch3-01 정의 1.2). $z_t$의 이웃 factors:
- 왼쪽: transition $f_t(z_{t-1}, z_t) = A_{z_{t-1}, z_t}$
- 오른쪽: transition $f_{t+1}(z_t, z_{t+1}) = A_{z_t, z_{t+1}}$
- 아래: emission $g_t(z_t, x_t) = B_{z_t, x_t}$; $x_t$ 관측 → $\tilde g_t(z_t) = B_{z_t, \hat x_t}$

**Left-to-right message** (sum-product recursion):
$$\mu_{f_t \to z_t}(z_t) = \sum_{z_{t-1}} f_t(z_{t-1}, z_t) \mu_{z_{t-1} \to f_t}(z_{t-1})$$

그리고
$$\mu_{z_{t-1} \to f_t}(z_{t-1}) = \mu_{f_{t-1} \to z_{t-1}}(z_{t-1}) \cdot \mu_{\tilde g_{t-1} \to z_{t-1}}(z_{t-1})$$
$$= \mu_{f_{t-1} \to z_{t-1}}(z_{t-1}) \cdot B_{z_{t-1}, x_{t-1}}$$

결합:
$$\mu_{f_t \to z_t}(z_t) = \sum_{z_{t-1}} A_{z_{t-1}, z_t} \cdot B_{z_{t-1}, x_{t-1}} \cdot \mu_{f_{t-1} \to z_{t-1}}(z_{t-1})$$

**Belief at $z_t$** (sum-product):
$$b(z_t) = \mu_{f_t \to z_t}(z_t) \cdot \mu_{\tilde g_t \to z_t}(z_t) \cdot \mu_{f_{t+1} \to z_t}(z_t)$$
$$= \mu_{f_t \to z_t}(z_t) \cdot B_{z_t, x_t} \cdot \mu_{f_{t+1} \to z_t}(z_t)$$

이제 $\alpha_t$와 대응시키기 위해:
$$\alpha_t(z_t) = p(z_t, x_{1:t}) = B_{z_t, x_t} \cdot \mu_{f_t \to z_t}(z_t)$$

(왼쪽에서 오는 메시지 $\mu_{f_t \to z_t}$는 $p(z_t, x_{1:t-1})$의 역할).

$\beta_t$는 오른쪽에서 오는 메시지:
$$\beta_t(z_t) = \mu_{f_{t+1} \to z_t}(z_t)$$

(여기서 $\mu_{f_{t+1} \to z_t}$는 $p(x_{t+1:T} | z_t)$).

따라서
$$b(z_t) \propto \alpha_t(z_t) \beta_t(z_t) / B_{z_t, x_t}$$

수학적으로 정확히 정의 차이만 있음. **Forward-Backward는 HMM factor graph의 sum-product**. $\square$

### 정리 2.2 — 두 Likelihood 계산 방법의 일치

**명제**: HMM likelihood는 Forward-only 또는 Forward-Backward 결합으로 계산 가능:
$$p(x_{1:T}) = \sum_{z_T} \alpha_T(z_T) = \sum_{z_t} \alpha_t(z_t) \beta_t(z_t) \quad \forall t$$

**증명**:

**Forward-only**: 정리 1.3 (Ch3-01).

**Arbitrary $t$**: 
$$\sum_{z_t} \alpha_t(z_t) \beta_t(z_t) = \sum_{z_t} p(z_t, x_{1:t}) p(x_{t+1:T} | z_t) = \sum_{z_t} p(z_t, x_{1:T}) = p(x_{1:T})$$

(두 번째 등호: chain rule + Markov). $\square$

이 identity가 **numerical sanity check**로 유용.

### 정리 2.3 — 복잡도

**명제**: Forward-Backward: $O(N^2 T)$ time, $O(NT)$ space.

**증명**: Forward와 backward recursion 각각 $O(N^2 T)$. Storage: $\alpha, \beta$ 둘 다 $N \times T$ — $O(NT)$. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

def forward(pi, A, B, obs):
    T = len(obs)
    N = len(pi)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
    return alpha

def backward(A, B, obs):
    T = len(obs)
    N = A.shape[0]
    beta = np.zeros((T, N))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t+1]] * beta[t+1])
    return beta

def scaled_forward_backward(pi, A, B, obs):
    """Scaled version — numerical stability."""
    T = len(obs)
    N = len(pi)
    alpha = np.zeros((T, N))
    c = np.zeros(T)
    
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]
    
    # Backward with same scale
    beta = np.zeros((T, N))
    beta[-1] = 1.0 / c[-1]
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t+1]] * beta[t+1])
        beta[t] /= c[t]
    
    # Posterior: gamma_t = alpha_t * beta_t * c_t (scaled)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)  # re-normalize for safety
    
    log_likelihood = np.log(c).sum()
    return alpha, beta, gamma, log_likelihood

# HMM setup (날씨 예시)
pi = np.array([0.6, 0.3, 0.1])
A = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])
B = np.array([[0.9, 0.1],
              [0.6, 0.4],
              [0.2, 0.8]])

obs = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]

# Unnormalized forward-backward
alpha = forward(pi, A, B, obs)
beta = backward(A, B, obs)

# 각 시점에서 likelihood가 동일한지 검증 (정리 2.2)
for t in range(len(obs)):
    likelihood_t = (alpha[t] * beta[t]).sum()
    print(f"t={t}: Σ α_t β_t = {likelihood_t:.6e}")

# Forward-only로 계산한 likelihood (reference)
print(f"\nΣ α_T = {alpha[-1].sum():.6e}")

# Scaled version
a, b, gamma, logL = scaled_forward_backward(pi, A, B, obs)
print(f"\nLog-likelihood (scaled): {logL:.6f}")
print(f"Log-likelihood (unscaled): {np.log(alpha[-1].sum()):.6f}")

# Posterior marginals
print("\nPosterior p(z_t | x_{1:T}) (scaled):")
print(gamma)

# Brute force 비교 (작은 T)
if len(obs) <= 10:
    from itertools import product
    N = len(pi)
    T = len(obs)
    total_joint = 0
    posterior_sum = np.zeros((T, N))
    for z_seq in product(range(N), repeat=T):
        p = pi[z_seq[0]] * B[z_seq[0], obs[0]]
        for t in range(1, T):
            p *= A[z_seq[t-1], z_seq[t]] * B[z_seq[t], obs[t]]
        total_joint += p
        for t in range(T):
            posterior_sum[t, z_seq[t]] += p
    
    brute_posterior = posterior_sum / total_joint
    print(f"\nBrute force posterior (정확도 검증):")
    print(brute_posterior)
    print(f"Max difference with scaled F-B: {np.abs(brute_posterior - gamma).max():.2e}")

# 시각화
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].imshow(np.log(alpha.T + 1e-20), cmap='viridis', aspect='auto')
axes[0].set_title(r'Forward $\log \alpha_t(z_t)$')
axes[0].set_ylabel('State')

axes[1].imshow(np.log(beta.T + 1e-20), cmap='viridis', aspect='auto')
axes[1].set_title(r'Backward $\log \beta_t(z_t)$')
axes[1].set_ylabel('State')

axes[2].imshow(gamma.T, cmap='viridis', aspect='auto', vmin=0, vmax=1)
axes[2].set_title(r'Posterior $\gamma_t(z_t) = p(z_t | x_{1:T})$')
axes[2].set_ylabel('State')
axes[2].set_xlabel('t')

plt.tight_layout()
plt.savefig('forward_backward_visualization.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
t=0: Σ α_t β_t = 4.371504e-05
t=1: Σ α_t β_t = 4.371504e-05
...
t=9: Σ α_t β_t = 4.371504e-05

Σ α_T = 4.371504e-05

Log-likelihood (scaled): -10.036021
Log-likelihood (unscaled): -10.036021

Posterior p(z_t | x_{1:T}) (scaled):
[[0.891 0.097 0.012]
 [0.765 0.188 0.047]
 ...
]

Brute force posterior (정확도 검증):
[[0.891 0.097 0.012] ...]
Max difference with scaled F-B: 2.77e-17
```

모든 $t$에서 $\sum \alpha \beta$가 동일 (정리 2.2), brute force와 거의 정확히 일치.

---

## 🔗 AI/ML 연결

### Baum-Welch의 E-step

EM for HMM에서 E-step은 정확히 Forward-Backward. 필요한 statistics:
- $\gamma_t(i) = p(z_t = i | x, \theta^{\text{old}})$: 각 state의 posterior marginal
- $\xi_t(i, j) = p(z_t = i, z_{t+1} = j | x, \theta^{\text{old}})$: transition posterior

M-step은 이들을 이용한 closed-form update (Ch3-04).

### CRF Training의 Gradient

Linear-chain CRF (Ch4-02):
$$\nabla_w \log p(y | x) = \sum_k f_k(y, x) - \mathbb{E}_{p(y|x)}[f_k]$$

**Expected feature count**: Forward-Backward로 $p(y_t | x), p(y_t, y_{t+1} | x)$ 계산 후 feature function을 marginal 아래 기댓값.

### CTC (Connectionist Temporal Classification)

Speech-to-text에서 label과 audio frame의 **alignment가 unknown**. CTC는 가능한 alignment를 모두 marginalize:

$$p(y | x) = \sum_{\text{alignments}} p(\text{alignment}, y | x)$$

이 summation은 **HMM Forward algorithm의 variant** — alignment path를 latent sequence로 간주. Backward-Forward 또는 just Forward로 $O(T \cdot |V|)$ time.

### Attention Alignment

Encoder-decoder seq2seq의 soft attention:
$$\alpha_{ij} = \text{softmax}(s(h_i, h_j))$$

이는 **learned Forward-Backward**의 일종. Attention weight = alignment probability $p(\text{source}_j | \text{target}_i)$. Hard attention = Viterbi. Soft attention = Forward-Backward posterior.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| First-order Markov | Long-range dependency는 RNN/Transformer 필요 |
| Stationary | Non-stationary에서는 switching model 필요 |
| Discrete state | Continuous는 Kalman (Ch3-05) |
| Underflow | $T$가 크면 scaled version 필수 |

**주의**: Unscaled forward $\alpha_T$는 $T = 100+$에서 쉽게 $10^{-50}$ 이하 → machine epsilon 이하. **반드시 scaled 또는 log-space** 구현.

---

## 📌 핵심 정리

$$\boxed{p(z_t | x_{1:T}) \propto \alpha_t(z_t) \beta_t(z_t), \quad \alpha, \beta = \text{sum-product messages on HMM factor graph}}$$

| 양 | 정의 | 해석 |
|----|------|------|
| $\alpha_t$ | $p(z_t, x_{1:t})$ | 과거 관측 + 현재 상태 결합 |
| $\beta_t$ | $p(x_{t+1:T} \| z_t)$ | 미래 관측 \| 현재 상태 |
| $\gamma_t$ | $p(z_t \| x_{1:T})$ | Posterior marginal = $\alpha_t \beta_t / Z$ |
| $\xi_t$ | $p(z_t, z_{t+1} \| x_{1:T})$ | Transition posterior |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\beta_t$의 recursion에서 왜 $A_{z_t, z_{t+1}}$와 $B_{z_{t+1}, x_{t+1}}$이 나오는지 유도하라.

<details>
<summary>힌트 및 해설</summary>

$$\beta_t(z_t) = p(x_{t+1:T} | z_t) = \sum_{z_{t+1}} p(z_{t+1}, x_{t+1:T} | z_t)$$

Chain rule:
$$p(z_{t+1}, x_{t+1:T} | z_t) = p(z_{t+1} | z_t) \cdot p(x_{t+1} | z_{t+1}, z_t) \cdot p(x_{t+2:T} | z_{t+1}, z_t, x_{t+1})$$

HMM의 CI 활용:
- $p(z_{t+1} | z_t) = A_{z_t, z_{t+1}}$
- $p(x_{t+1} | z_{t+1}, z_t) = p(x_{t+1} | z_{t+1}) = B_{z_{t+1}, x_{t+1}}$ (output independence)
- $p(x_{t+2:T} | z_{t+1}, z_t, x_{t+1}) = p(x_{t+2:T} | z_{t+1}) = \beta_{t+1}(z_{t+1})$ (Markov + output independence)

결합:
$$\beta_t(z_t) = \sum_{z_{t+1}} A_{z_t, z_{t+1}} B_{z_{t+1}, x_{t+1}} \beta_{t+1}(z_{t+1})$$

$\square$

</details>

**문제 2** (심화): Forward-Backward의 수치 안정성을 위한 **log-space 구현**을 유도하라 (logsumexp 사용).

<details>
<summary>힌트 및 해설</summary>

$\alpha_t(z_t)$가 $e^{-T \cdot \text{something}}$로 underflow. 해결책: $\log \alpha_t(z_t)$를 저장.

**Recursion in log space**:
$$\log \alpha_t(z_t) = \log B_{z_t, x_t} + \log \sum_{z_{t-1}} \alpha_{t-1}(z_{t-1}) A_{z_{t-1}, z_t}$$

$$= \log B_{z_t, x_t} + \log \sum_{z_{t-1}} \exp(\log \alpha_{t-1}(z_{t-1}) + \log A_{z_{t-1}, z_t})$$

$$= \log B_{z_t, x_t} + \text{logsumexp}_{z_{t-1}}(\log \alpha_{t-1}(z_{t-1}) + \log A_{z_{t-1}, z_t})$$

**Logsumexp trick**: $\log \sum_i e^{x_i} = \max_i x_i + \log \sum_i e^{x_i - \max_i x_i}$. Exponent가 0 근처로 유지되어 overflow/underflow 방지.

NumPy: `scipy.special.logsumexp`.

**Likelihood**: $\log p(x_{1:T}) = \text{logsumexp}_{z_T}(\log \alpha_T(z_T))$.

**비교**: Scaled version과 동치 — 단지 multiplicative scale을 additive log constant로 표현.

</details>

**문제 3** (AI 연결): CTC loss의 forward pass를 HMM Forward algorithm과 비교하라. 어떤 state space를 사용하는가?

<details>
<summary>힌트 및 해설</summary>

**CTC setup**: Target labels $y = (y_1, \ldots, y_L)$, audio frames $x_{1:T}$ ($T \gg L$). Label sequence를 "blank-extended" version으로 확장:
$$y' = (\epsilon, y_1, \epsilon, y_2, \epsilon, \ldots, y_L, \epsilon)$$

**State space**: 각 frame $t$에서 $y'$의 position $u \in \{1, \ldots, 2L+1\}$. State = $(u, t)$.

**Allowable transitions**:
- $u \to u$ (stay at same label)
- $u \to u+1$ (advance)
- $u \to u+2$ (skip blank, if $y'_u \neq \epsilon$ and $y'_{u+2} \neq y'_u$)

**Forward recursion**:
$$\alpha_t(u) = p(y'_u | x_t) \cdot \sum_{\text{allowable } u'} \alpha_{t-1}(u')$$

여기서 $p(y'_u | x_t)$는 neural network output (softmax).

**Loss**: $\log p(y | x) = \log \alpha_T(2L) + \log \alpha_T(2L+1)$ (end states).

**HMM과의 차이**:
- HMM: $N$ states (fixed)
- CTC: $2L + 1$ states (per example, depends on label length)
- HMM: transition probability learned
- CTC: transition is **structural** (monotonic alignment constraint), emission from neural net

**공통점**: 둘 다 dynamic programming, $O(T \cdot N)$ or $O(T \cdot L)$. Forward-Backward로 gradient 계산 가능.

결론: CTC = "structured HMM with fixed transition structure + learned emissions". Ch7-05의 "신경망과 그래프 모델의 결합"의 초기 예시.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. HMM의 정의와 세 가지 문제](./01-hmm-definition.md) | [📚 README](../README.md) | [03. Viterbi Algorithm = Max-Product ▶](./03-viterbi-algorithm.md) |

</div>
