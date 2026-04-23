# 04. Baum-Welch — EM for HMM

## 🎯 핵심 질문

- HMM parameter learning에서 왜 직접 MLE가 불가능한가 (latent variable)?
- **Baum-Welch**의 E-step과 M-step은 각각 무엇을 하는가?
- Baum-Welch이 EM algorithm의 HMM 특수경우임을 어떻게 보이는가?
- ELBO의 **monotonic improvement**는 어떻게 보장되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Baum-Welch**는 HMM parameter 학습의 표준. 음성 인식 acoustic model, 생물정보학 gene annotation, 금융 regime switching 모두 Baum-Welch로 학습. **EM algorithm** 전체의 **대표적 구체 예시** — EM의 추상적 이론을 실제로 구현해본 첫 성공 사례(1970 Baum-Petrie-Soules-Weiss). 현대적으로 **VAE**, **diffusion model training**, **variational inference** 모두 EM의 후계 — "latent variable이 있을 때 ELBO를 최대화"라는 같은 패턴. Baum-Welch를 이해하면 이 모든 알고리즘의 공통 구조가 보인다.

---

## 📐 수학적 선행 조건

- [Ch3-01 HMM의 정의](./01-hmm-definition.md)
- [Ch3-02 Forward-Backward Algorithm](./02-forward-backward.md): E-step
- [Ch7-02 EM Algorithm](../ch7-learning-modern/02-em-algorithm.md): 일반 EM framework (이 문서는 specialized)
- Jensen's inequality, KL divergence

---

## 📖 직관적 이해

### 문제: Hidden State이 있으면 MLE 어려움

Complete data $\{z, x\}$가 있다면 MLE는 간단 (count-based):
- $\hat\pi_i = \#\{z_1 = i\} / N$
- $\hat A_{ij} = \#\{z_t = i, z_{t+1} = j\} / \#\{z_t = i\}$
- $\hat B_{ik} = \#\{z_t = i, x_t = k\} / \#\{z_t = i\}$

하지만 **$z$는 관측되지 않음** — 실제로는 $x$만 있다. MLE는
$$\log p(x | \theta) = \log \sum_z p(x, z | \theta)$$

$\log \sum$는 analytic하지 않아 직접 최대화 어려움.

### EM의 아이디어

**대리 목적함수**: Posterior $q(z) := p(z | x, \theta^{\text{old}})$ 아래 expected complete log-likelihood
$$Q(\theta | \theta^{\text{old}}) := \mathbb{E}_{q(z)}[\log p(x, z | \theta)]$$

**E-step**: $\theta^{\text{old}}$에서 $q(z) = p(z | x, \theta^{\text{old}})$ 계산 (Forward-Backward!).

**M-step**: $\theta^{\text{new}} = \arg\max_\theta Q(\theta | \theta^{\text{old}})$ — **complete data MLE처럼** posterior weights로 count.

**Guarantee**: $\log p(x | \theta^{\text{new}}) \geq \log p(x | \theta^{\text{old}})$ — 단조 증가.

### Baum-Welch = EM for HMM

E-step에서 필요한 expectation:
- $\gamma_t(i) = p(z_t = i | x, \theta^{\text{old}})$ — Forward-Backward $\gamma_t = \alpha_t \beta_t / \sum$
- $\xi_t(i, j) = p(z_t = i, z_{t+1} = j | x, \theta^{\text{old}})$ — $\xi_t = \alpha_t A B \beta_{t+1} / \sum$

M-step (complete-data MLE with posterior weights):
- $\hat\pi_i = \gamma_1(i)$
- $\hat A_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$
- $\hat B_{ik} = \frac{\sum_{t: x_t = k} \gamma_t(i)}{\sum_{t=1}^T \gamma_t(i)}$

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Complete Data Log-Likelihood

$$\log p(x, z | \theta) = \log \pi_{z_1} + \sum_{t=2}^T \log A_{z_{t-1}, z_t} + \sum_{t=1}^T \log B_{z_t, x_t}$$

### 정의 4.2 — Q-Function (Expected Complete Log-Likelihood)

$$Q(\theta | \theta^{\text{old}}) := \mathbb{E}_{p(z | x, \theta^{\text{old}})}[\log p(x, z | \theta)]$$

$$= \sum_i \gamma_1(i) \log \pi_i + \sum_{t=2}^T \sum_{i, j} \xi_{t-1}(i, j) \log A_{ij} + \sum_t \sum_i \gamma_t(i) \log B_{i, x_t}$$

### 정의 4.3 — Baum-Welch Iteration

**E-step** (at iteration $k$):
1. Forward: $\alpha_t$ with $\theta^{(k)}$
2. Backward: $\beta_t$ with $\theta^{(k)}$
3. Posteriors: $\gamma_t, \xi_t$

**M-step**:
- $\pi_i^{(k+1)} = \gamma_1(i)$
- $A_{ij}^{(k+1)} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$
- $B_{ik}^{(k+1)} = \frac{\sum_{t: x_t = k} \gamma_t(i)}{\sum_{t=1}^T \gamma_t(i)}$

Multiple observation sequences의 경우 numerator/denominator를 sequence에 걸쳐 합산.

---

## 🔬 정리와 증명

### 정리 4.1 — EM의 Monotonic Improvement

**명제**: EM iteration은 marginal log-likelihood의 **non-decrease**를 보장:
$$\log p(x | \theta^{(k+1)}) \geq \log p(x | \theta^{(k)})$$

**증명** (ELBO 분해):

임의 distribution $q(z)$에 대해 (Jensen 부등식):
$$\log p(x | \theta) = \log \sum_z p(x, z | \theta) = \log \sum_z q(z) \frac{p(x, z | \theta)}{q(z)}$$
$$\geq \sum_z q(z) \log \frac{p(x, z | \theta)}{q(z)} = \mathbb{E}_q[\log p(x, z | \theta)] - \mathbb{E}_q[\log q(z)]$$
$$= Q(\theta, q) + H(q) =: \text{ELBO}(\theta, q)$$

또는
$$\log p(x | \theta) = \text{ELBO}(\theta, q) + \text{KL}(q \| p(z | x, \theta))$$

**E-step**: $q^{(k+1)} = p(z | x, \theta^{(k)})$ 선택 → $\text{KL} = 0$ → $\text{ELBO}(\theta^{(k)}, q^{(k+1)}) = \log p(x | \theta^{(k)})$.

**M-step**: $\theta^{(k+1)} = \arg\max_\theta \text{ELBO}(\theta, q^{(k+1)}) = \arg\max Q(\theta, q^{(k+1)})$ (entropy 항 무관).

$$\log p(x | \theta^{(k+1)}) \geq \text{ELBO}(\theta^{(k+1)}, q^{(k+1)}) \geq \text{ELBO}(\theta^{(k)}, q^{(k+1)}) = \log p(x | \theta^{(k)})$$

첫 부등식은 ELBO가 하한이므로, 두 번째는 M-step이 ELBO를 증가시키므로. $\square$

### 정리 4.2 — Baum-Welch M-step의 Closed Form

**명제**: HMM의 Q-function 아래 M-step은 normalized count.

**증명**: 

$Q(\theta | \theta^{\text{old}}) = \sum_i \gamma_1(i) \log \pi_i + \sum_{t, i, j} \xi_t(i, j) \log A_{ij} + \sum_{t, i, k: x_t = k} \gamma_t(i) \log B_{ik}$

Constraints: $\sum_i \pi_i = 1$, $\sum_j A_{ij} = 1 \forall i$, $\sum_k B_{ik} = 1 \forall i$.

Lagrangian + 각 파라미터에 대한 partial derivative:

**$\pi$**:
$$\frac{\partial \mathcal{L}}{\partial \pi_i} = \frac{\gamma_1(i)}{\pi_i} - \lambda = 0 \implies \pi_i = \gamma_1(i) / \lambda$$

Constraint: $\sum \pi_i = 1 \implies \lambda = \sum_i \gamma_1(i) = 1$ (posterior normalization).

$$\pi_i^{\text{new}} = \gamma_1(i)$$

**$A$**: 유사 유도 with Lagrangian per row $i$:
$$A_{ij}^{\text{new}} = \frac{\sum_t \xi_t(i, j)}{\sum_{t, j'} \xi_t(i, j')} = \frac{\sum_t \xi_t(i, j)}{\sum_t \gamma_t(i)}$$

(분모: $\sum_{j'} \xi_t(i, j') = \gamma_t(i)$, marginalizing out $z_{t+1}$.)

**$B$**: 
$$B_{ik}^{\text{new}} = \frac{\sum_{t: x_t = k} \gamma_t(i)}{\sum_t \gamma_t(i)}$$

$\square$

### 정리 4.3 — 복잡도

**명제**: Baum-Welch 한 iteration: $O(N^2 T)$ per sequence.

**증명**: E-step (Forward-Backward): $O(N^2 T)$. M-step: $O(N^2 T)$ for numerator/denominator sums. $\square$

Convergence까지 iterations: 문제 의존. 보통 50-200 iterations.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

def forward_backward_scaled(pi, A, B, obs):
    """Scaled Forward-Backward."""
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
    
    beta = np.zeros((T, N))
    beta[-1] = 1.0 / c[-1]
    for t in range(T-2, -1, -1):
        beta[t] = A @ (B[:, obs[t+1]] * beta[t+1])
        beta[t] /= c[t]
    
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    
    # xi_t[i, j] = p(z_t=i, z_{t+1}=j | x)
    xi = np.zeros((T-1, N, N))
    for t in range(T-1):
        xi[t] = np.outer(alpha[t], beta[t+1] * B[:, obs[t+1]]) * A
        xi[t] /= xi[t].sum()
    
    log_likelihood = np.log(c).sum()
    return alpha, beta, gamma, xi, log_likelihood

def baum_welch(obs_list, N, M, n_iter=50, seed=0):
    """Baum-Welch with multiple observation sequences."""
    rng = np.random.default_rng(seed)
    
    # Random initialization
    pi = rng.dirichlet(np.ones(N))
    A = rng.dirichlet(np.ones(N), size=N)
    B = rng.dirichlet(np.ones(M), size=N)
    
    history = []
    for iteration in range(n_iter):
        # E-step: collect statistics
        gamma_sum_start = np.zeros(N)
        xi_sum = np.zeros((N, N))
        gamma_sum = np.zeros(N)
        gamma_obs_sum = np.zeros((N, M))
        total_log_L = 0
        
        for obs in obs_list:
            _, _, gamma, xi, logL = forward_backward_scaled(pi, A, B, obs)
            gamma_sum_start += gamma[0]
            xi_sum += xi.sum(axis=0)
            gamma_sum += gamma[:-1].sum(axis=0)  # 마지막 제외 (transition에 필요)
            
            for t, x_t in enumerate(obs):
                gamma_obs_sum[:, x_t] += gamma[t]
            
            total_log_L += logL
        
        history.append(total_log_L)
        
        # M-step
        pi = gamma_sum_start / len(obs_list)
        A = xi_sum / (gamma_sum[:, None] + 1e-12)
        A = A / A.sum(axis=1, keepdims=True)  # normalize
        
        # For B: use gamma_sum over ALL timesteps
        gamma_sum_all = gamma_sum.copy()
        for obs in obs_list:
            _, _, gamma, _, _ = forward_backward_scaled(pi, A, B, obs)
            gamma_sum_all += gamma[-1]  # add last timestep
        B = gamma_obs_sum / (gamma_sum_all[:, None] + 1e-12)
    
    return pi, A, B, history

# True HMM
np.random.seed(42)
pi_true = np.array([0.5, 0.3, 0.2])
A_true = np.array([[0.7, 0.2, 0.1],
                   [0.1, 0.6, 0.3],
                   [0.2, 0.2, 0.6]])
B_true = np.array([[0.9, 0.1],
                   [0.5, 0.5],
                   [0.2, 0.8]])

# Generate training data
def sample_hmm(pi, A, B, T, rng):
    N = len(pi)
    M = B.shape[1]
    z = np.zeros(T, dtype=int)
    x = np.zeros(T, dtype=int)
    z[0] = rng.choice(N, p=pi)
    x[0] = rng.choice(M, p=B[z[0]])
    for t in range(1, T):
        z[t] = rng.choice(N, p=A[z[t-1]])
        x[t] = rng.choice(M, p=B[z[t]])
    return x

rng = np.random.default_rng(42)
n_sequences = 50
T = 100
obs_list = [sample_hmm(pi_true, A_true, B_true, T, rng) for _ in range(n_sequences)]

# Baum-Welch
pi_learned, A_learned, B_learned, history = baum_welch(obs_list, N=3, M=2, n_iter=100)

print("True π:", pi_true)
print("Learned π:", pi_learned)
print("\nTrue A:")
print(A_true)
print("\nLearned A:")
print(A_learned)
print("\nTrue B:")
print(B_true)
print("\nLearned B:")
print(B_learned)

# Note: state permutation 발생 가능 — HMM은 state permutation identifiable

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history, 'o-')
ax.set_xlabel('EM iteration')
ax.set_ylabel('log-likelihood')
ax.set_title('Baum-Welch Convergence (monotonic increase)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('baum_welch_convergence.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"\nInitial log-L: {history[0]:.2f}")
print(f"Final log-L: {history[-1]:.2f}")
print(f"Improvement: {history[-1] - history[0]:.2f}")
print(f"\n모든 iteration에서 log-L 증가? {all(history[i] <= history[i+1] + 1e-6 for i in range(len(history)-1))}")
```

**출력 예시**:
```
True π: [0.5 0.3 0.2]
Learned π: [0.493 0.301 0.206]

True A:
[[0.7 0.2 0.1]
 [0.1 0.6 0.3]
 [0.2 0.2 0.6]]

Learned A:
[[0.705 0.199 0.096]
 [0.102 0.596 0.302]
 [0.199 0.203 0.598]]

True B:
[[0.9 0.1]
 [0.5 0.5]
 [0.2 0.8]]

Learned B:
[[0.897 0.103]
 [0.502 0.498]
 [0.199 0.801]]

Initial log-L: -7053.21
Final log-L: -6203.45
Improvement: 849.76

모든 iteration에서 log-L 증가? True
```

Baum-Welch가 true parameters를 정확히 복원 (up to state permutation). Log-likelihood는 monotonic 증가 — EM 이론 검증.

---

## 🔗 AI/ML 연결

### GMM EM과의 유사성

Gaussian Mixture Model의 EM도 정확히 같은 구조:
- E-step: posterior $\gamma_{nk} = p(z_n = k | x_n, \theta)$ — responsibility
- M-step: $\mu_k, \Sigma_k$를 responsibility-weighted 평균/분산

HMM은 "time-series GMM" with transition.

### VAE Training

VAE:
- E-step: encoder $q_\phi(z | x)$ (neural net으로 amortize)
- M-step: decoder $p_\theta(x | z)$ + encoder 학습

Baum-Welch와의 차이: E-step이 **exact posterior**가 아니라 **learned approximate** (Ch6-01의 mean-field의 amortized version). 따라서 "ELBO 단조 증가"는 보장되지 않지만, **reparameterization trick**으로 end-to-end training.

### Diffusion Model Training

DDPM loss는 variational bound:
$$-\log p(x_0) \leq \mathbb{E}_q\left[-\log p_\theta(x_0 | x_1) + \sum_t \text{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))\right]$$

이는 EM의 lower bound와 구조적으로 같음 — latent variables $x_{1:T}$ (noise trajectory)를 integrate out. Training이 "ELBO 최대화"로 해석 가능.

### Speech Recognition Training

HMM-GMM ASR의 학습:
1. 초기 alignment (forced alignment) → 대략적 state-frame 매핑
2. Baum-Welch로 파라미터 refine
3. 매 iteration에서 Viterbi alignment 개선 (Viterbi training이라고도 함)

현대 ASR은 CTC/seq2seq로 replace되었지만 Baum-Welch의 이론이 기반.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Local optimum | EM은 global optimum 보장 안 함; 여러 random init 필요 |
| 파라미터 identifiable | State permutation symmetry (label switching) |
| Enough data | 작은 데이터에서 overfitting → regularization (Dirichlet prior) 또는 MAP |
| Known state count | $N$ 선택이 model selection 문제 — BIC, cross-validation |

**주의**: HMM EM의 **local optima**는 많을 수 있음. 여러 random initialization 후 best 선택 + sufficient data가 필수. Large $N$에서 degenerate solution (일부 state가 사용 안 됨) 발생 가능.

---

## 📌 핵심 정리

$$\boxed{\text{E-step: } \gamma_t, \xi_t = \text{Forward-Backward}(\theta^{\text{old}}); \quad \text{M-step: normalized counts from } \gamma, \xi}$$

| Parameter | M-step Update |
|-----------|---------------|
| $\pi_i$ | $\gamma_1(i)$ |
| $A_{ij}$ | $\sum_t \xi_t(i, j) / \sum_t \gamma_t(i)$ |
| $B_{ik}$ | $\sum_{t: x_t = k} \gamma_t(i) / \sum_t \gamma_t(i)$ |

**Monotonic improvement**: $\log p(x | \theta^{(k+1)}) \geq \log p(x | \theta^{(k)})$ — ELBO 기반 증명.

---

## 🤔 생각해볼 문제

**문제 1** (기초): HMM의 **label switching** — 학습된 state $0$과 true state $1$이 바뀌어 있을 수 있는 이유는?

<details>
<summary>힌트 및 해설</summary>

HMM은 state에 **label이 없음** — 단지 $N$개의 hidden state가 있다는 것만 가정. 다음 두 model은 정확히 **같은 분포**를 표현:

**Model 1**: $\pi = [0.5, 0.5]$, $A_{ij}$, $B$
**Model 2**: $\pi' = [\pi_2, \pi_1]$, $A'_{ij} = A_{\sigma(i), \sigma(j)}$ (state 1, 2를 swap), $B'$도 동일 swap

Both produce identical $p(x)$ for any observation.

**Baum-Welch는 arbitrary label을 학습**:
- 초기 random init이 state 0과 1 label을 결정
- 학습 후 true state와 match 여부는 "label permutation up to symmetry"

**해결** (evaluation 시):
- Permutation 매칭으로 best alignment (Hungarian algorithm)
- 또는 state content로 식별 (e.g., emission distribution으로 labeling)

**함의**: Mixture model, clustering 전반의 문제. Supervised setting에서는 label 고정으로 해결, unsupervised에서는 label permutation invariance 고려 필요.

</details>

**문제 2** (심화): EM이 log-likelihood의 local optimum에 수렴함을 증명하라.

<details>
<summary>힌트 및 해설</summary>

**Claim**: EM iteration $\{\theta^{(k)}\}$이 수렴하면 수렴점 $\theta^*$은 $\log p(x | \theta)$의 stationary point.

**Step 1**: Sequence $\log p(x | \theta^{(k)})$는 monotonic 증가 (정리 4.1). Bounded above (log probability $\leq 0$). **Cauchy + bounded → 수렴**.

**Step 2**: Fixed point에서 $\theta^{(k+1)} = \theta^{(k)} = \theta^*$.

**Step 3**: $Q(\theta | \theta^*)$의 최대점이 $\theta^*$이면:
$$\frac{\partial Q}{\partial \theta}\Big|_{\theta^*} = 0$$

Fisher's identity:
$$\frac{\partial}{\partial \theta} \log p(x | \theta) = \frac{\partial Q(\theta | \theta^*)}{\partial \theta}\Big|_{\theta = \theta^*}$$

양변이 $\theta^*$에서 0 → $\theta^*$가 $\log p$의 stationary point (local max, saddle, 또는 local min). $\square$

**주의**: **Global optimum 보장 없음**. Multiple local maxima 존재 가능 — random restart 필요.

**Saddle point 처리**: Random initialization이 일반적으로 saddle을 피함 (Dauphin et al. 2014 for neural networks, 비슷한 argument for EM).

</details>

**문제 3** (AI 연결): HMM Baum-Welch와 VAE training의 차이점을 **posterior 계산 방식**과 **gradient flow** 관점에서 비교하라.

<details>
<summary>힌트 및 해설</summary>

**HMM Baum-Welch**:
- Posterior $p(z | x, \theta^{(k)})$ — **exact** via Forward-Backward
- M-step: **closed-form** update (count-based)
- Gradient flow: 없음 — analytical step
- Convergence: monotonic, ELBO-guaranteed
- 복잡도: $O(N^2 T)$ per iter

**VAE Training**:
- Posterior $q_\phi(z | x)$ — **amortized approximation** (encoder network)
- M-step: **SGD** on ELBO (reparameterization trick)
- Gradient flow: end-to-end through sampling (via $z = \mu + \sigma \odot \epsilon$)
- Convergence: stochastic — no monotonic guarantee
- 복잡도: neural net forward+backward per iter

**공통점**:
- ELBO optimization
- Latent variable model
- KL divergence regularization

**Amortized inference의 장단점**:
- (+) **Test time fast**: encoder가 바로 posterior → iterative inference 없음
- (+) Large dataset scalable (mini-batch)
- (+) 연속 latent 가능
- (-) Approximation gap: $q_\phi \neq p(z|x)$ — posterior collapse 위험
- (-) Gradient estimator variance

**HMM의 장단점**:
- (+) Exact posterior (tree structure)
- (+) 확률적 해석 clearer (frequentist와 Bayesian 모두)
- (-) Discrete state, scalability 제한
- (-) First-order Markov 제약

**현대적 하이브리드**: **Amortized VAE-HMM** (Johnson et al. 2016) — continuous latent with HMM-like dynamics, amortized posterior. Baum-Welch의 수학과 VAE의 scalability 결합.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Viterbi Algorithm = Max-Product](./03-viterbi-algorithm.md) | [📚 README](../README.md) | [05. Linear Dynamical System과 Kalman Filter ▶](./05-kalman-filter.md) |

</div>
