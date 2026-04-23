# 02. EM Algorithm — 불완전 데이터

## 🎯 핵심 질문

- EM의 **Q-function** $Q(\theta | \theta^{\text{old}}) = \mathbb{E}_{p(z|x, \theta^{\text{old}})}[\log p(x, z|\theta)]$는 어떻게 유도되는가?
- **ELBO의 lower bound** 관점으로 EM의 monotonic improvement를 어떻게 증명하는가?
- GMM, HMM, LDA의 EM이 공통 framework의 특수경우인 이유는?
- **Generalized EM (GEM)**, **Stochastic EM** 같은 variant는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**EM Algorithm** (Dempster-Laird-Rubin 1977)은 **latent variable model 학습의 universal framework**. GMM, HMM (Baum-Welch, Ch3-04), LDA (Ch7-04), PCA(EM version), mixture of experts — 모두 EM의 특수 경우. Modern **VAE**, **diffusion training**도 EM의 연속 generalization (amortized E-step). EM의 **monotonic ELBO improvement** 이론이 variational inference, 이후 stochastic VI, amortized VI 모두의 기반. "Latent 있을 때 likelihood 최대화" 패턴의 수학적 정형화.

---

## 📐 수학적 선행 조건

- [Ch3-04 Baum-Welch — EM for HMM](../ch3-hmm/04-baum-welch.md): 구체적 예시
- [Ch6-01 Mean-Field Variational Inference](../ch6-approximate-inference/01-mean-field-vi.md): ELBO
- Jensen inequality
- KL divergence

---

## 📖 직관적 이해

### Latent Variable Models

$x$: observed, $z$: latent (unobserved).
$$p(x | \theta) = \sum_z p(x, z | \theta)$$

- **GMM**: $z$ = cluster assignment
- **HMM**: $z$ = hidden state sequence
- **Factor Analysis**: $z$ = low-dim latent factor
- **PCA**: $z$ = principal components (probabilistic)
- **LDA**: $z$ = topic assignment
- **VAE**: $z$ = continuous latent code

### EM의 기본 idea

Log-likelihood:
$$\ell(\theta) = \sum_i \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)} | \theta)$$

**Problem**: $\log \sum$ — non-convex, no closed form in general.

**Trick**: Posterior $q(z) = p(z | x, \theta^{\text{old}})$ (using current $\theta^{\text{old}}$)를 "가상 completion":

**E-step**: Compute $q(z) = p(z | x, \theta^{\text{old}})$.

**M-step**: Maximize expected complete-data log-likelihood:
$$\theta^{\text{new}} = \arg\max_\theta \mathbb{E}_q[\log p(x, z | \theta)] = \arg\max_\theta Q(\theta | \theta^{\text{old}})$$

### ELBO Interpretation

$$\log p(x | \theta) \geq \text{ELBO}(\theta, q) = \mathbb{E}_q[\log p(x, z | \theta)] - \mathbb{E}_q[\log q]$$

**EM as coordinate ascent on ELBO**:
- **E-step**: Fix $\theta$, optimize $q$ → $q^* = p(z | x, \theta)$ → ELBO = $\log p(x | \theta)$ (gap closed)
- **M-step**: Fix $q$, optimize $\theta$ → $\theta^{\text{new}}$ (ELBO 증가)

Each step **monotonically increases** $\ell$.

### GMM EM 예시

Gaussian Mixture Model:
$$p(x | \theta) = \sum_k \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

**E-step**: Responsibility
$$\gamma_{nk} = p(z_n = k | x_n, \theta) = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{k'} \pi_{k'} \mathcal{N}(x_n | \mu_{k'}, \Sigma_{k'})}$$

**M-step**: Responsibility-weighted updates
$$\pi_k = \frac{\sum_n \gamma_{nk}}{N}, \quad \mu_k = \frac{\sum_n \gamma_{nk} x_n}{\sum_n \gamma_{nk}}, \quad \Sigma_k = \frac{\sum_n \gamma_{nk} (x_n - \mu_k)(x_n - \mu_k)^T}{\sum_n \gamma_{nk}}$$

"Soft K-means" — responsibility가 hard 대신 soft assignment.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Q-Function

$$Q(\theta | \theta^{\text{old}}) := \mathbb{E}_{p(z | x, \theta^{\text{old}})}[\log p(x, z | \theta)]$$

### 정의 2.2 — EM Algorithm

Initialize $\theta^{(0)}$. For $t = 0, 1, 2, \ldots$:

**E-step**: Compute $q^{(t)}(z) := p(z | x, \theta^{(t)})$.

**M-step**: $\theta^{(t+1)} := \arg\max_\theta Q(\theta | \theta^{(t)})$.

Continue until $\theta$ converges or $\log p(x | \theta)$ increment < tolerance.

### 정의 2.3 — ELBO Decomposition

$$\log p(x | \theta) = \underbrace{\mathbb{E}_q[\log p(x, z | \theta)] - \mathbb{E}_q[\log q]}_{\text{ELBO}(\theta, q)} + \underbrace{\text{KL}(q \| p(z | x, \theta))}_{\geq 0}$$

### 정의 2.4 — Generalized EM (GEM)

M-step을 **완전 최대화** 대신 **increase**만으로도 충분 (수렴 여전히 보장):
$$\theta^{(t+1)}: Q(\theta^{(t+1)} | \theta^{(t)}) \geq Q(\theta^{(t)} | \theta^{(t)})$$

### 정의 2.5 — Stochastic EM

Large data에서 **mini-batch** E-step (또는 single sample). Monte Carlo EM, Stochastic EM 등 variants.

---

## 🔬 정리와 증명

### 정리 2.1 — EM의 Monotonic Improvement

**명제**: EM iteration은 $\log p(x | \theta)$의 non-decreasing:
$$\log p(x | \theta^{(t+1)}) \geq \log p(x | \theta^{(t)})$$

**증명**:

ELBO 분해:
$$\log p(x | \theta) = \text{ELBO}(\theta, q) + \text{KL}(q \| p(z | x, \theta))$$

**E-step** at $\theta = \theta^{(t)}$:
- $q^{(t+1)} := p(z | x, \theta^{(t)})$
- $\text{KL}(q^{(t+1)} \| p(z | x, \theta^{(t)})) = 0$
- $\text{ELBO}(\theta^{(t)}, q^{(t+1)}) = \log p(x | \theta^{(t)})$

**M-step**: $\theta^{(t+1)} = \arg\max_\theta \text{ELBO}(\theta, q^{(t+1)})$:

$$\text{ELBO}(\theta^{(t+1)}, q^{(t+1)}) \geq \text{ELBO}(\theta^{(t)}, q^{(t+1)}) = \log p(x | \theta^{(t)})$$

그런데:
$$\log p(x | \theta^{(t+1)}) = \text{ELBO}(\theta^{(t+1)}, q^{(t+1)}) + \text{KL}(q^{(t+1)} \| p(z | x, \theta^{(t+1)})) \geq \text{ELBO}(\theta^{(t+1)}, q^{(t+1)}) \geq \log p(x | \theta^{(t)})$$

(KL $\geq 0$, M-step's non-decrease).

$\square$

### 정리 2.2 — Q-Function Maximization = ELBO Maximization in $\theta$

**명제**: M-step의 $\arg\max_\theta Q(\theta | \theta^{\text{old}}) = \arg\max_\theta \text{ELBO}(\theta, q^{\text{old}})$.

**증명**:

$$\text{ELBO}(\theta, q) = \mathbb{E}_q[\log p(x, z | \theta)] - \mathbb{E}_q[\log q]$$

$q$ fixed (just $q^{\text{old}}$), so $\mathbb{E}_q[\log q]$ constant in $\theta$.

$$\text{ELBO}(\theta, q^{\text{old}}) = Q(\theta | \theta^{\text{old}}) + \text{const}$$

Therefore $\arg\max_\theta = $ same. $\square$

### 정리 2.3 — Fixed Point = Stationary of Log-Likelihood

**명제**: EM의 fixed point $\theta^*$는 $\log p(x | \theta)$의 stationary point:
$$\frac{\partial \log p(x | \theta)}{\partial \theta}\bigg|_{\theta = \theta^*} = 0$$

**증명** (Fisher's identity):

$$\frac{\partial \log p(x | \theta)}{\partial \theta} = \frac{1}{p(x | \theta)} \frac{\partial p(x | \theta)}{\partial \theta} = \sum_z \frac{p(x, z | \theta)}{p(x | \theta)} \frac{\partial \log p(x, z | \theta)}{\partial \theta}$$
$$= \mathbb{E}_{p(z | x, \theta)}\left[\frac{\partial \log p(x, z | \theta)}{\partial \theta}\right]$$

$$= \frac{\partial}{\partial \theta} Q(\theta | \theta)\bigg|_{\theta = \theta}$$

$\theta^*$ = M-step fixed point: $\partial Q(\theta | \theta^*) / \partial \theta = 0$ at $\theta = \theta^*$.

Fisher's identity: this equals $\partial \log p(x | \theta) / \partial \theta$ at $\theta^*$.

$\square$

### 정리 2.4 — 수렴

**명제**: Under regularity conditions, EM sequence $\theta^{(t)}$ converges to local optimum of $\log p(x | \theta)$.

**증명 개요**: 

$\log p(x | \theta^{(t)})$ monotonic + bounded → converges. 정리 2.3으로 limit이 stationary point.

**Global optimum은 NOT 보장**. Random initialization 여러 번 필요.

**Saddle points**: EM can get stuck at saddle points in principle, but practically rare (random init + numerical noise).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# GMM EM 구현
class GMM_EM:
    def __init__(self, K, n_features):
        self.K = K
        self.n_features = n_features
    
    def fit(self, X, n_iter=100, seed=0):
        N, d = X.shape
        rng = np.random.default_rng(seed)
        
        # Initialize
        idx = rng.choice(N, self.K, replace=False)
        self.mu = X[idx]
        self.Sigma = np.array([np.cov(X.T) + 0.1 * np.eye(d) for _ in range(self.K)])
        self.pi = np.ones(self.K) / self.K
        
        history = []
        for it in range(n_iter):
            # E-step
            log_gamma = np.zeros((N, self.K))
            for k in range(self.K):
                log_gamma[:, k] = np.log(self.pi[k] + 1e-300) + \
                                  multivariate_normal.logpdf(X, self.mu[k], self.Sigma[k])
            log_gamma -= log_gamma.max(axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            gamma /= gamma.sum(axis=1, keepdims=True)
            
            # M-step
            N_k = gamma.sum(axis=0)
            self.pi = N_k / N
            for k in range(self.K):
                self.mu[k] = (gamma[:, k:k+1] * X).sum(axis=0) / N_k[k]
                diff = X - self.mu[k]
                self.Sigma[k] = (gamma[:, k:k+1] * diff).T @ diff / N_k[k]
                self.Sigma[k] += 1e-6 * np.eye(d)  # regularization
            
            # Log-likelihood (for monitoring)
            log_L = 0
            for n in range(N):
                component_probs = [self.pi[k] * multivariate_normal.pdf(X[n], self.mu[k], self.Sigma[k])
                                   for k in range(self.K)]
                log_L += np.log(sum(component_probs) + 1e-300)
            history.append(log_L)
        
        return history
    
    def predict(self, X):
        N = X.shape[0]
        log_gamma = np.zeros((N, self.K))
        for k in range(self.K):
            log_gamma[:, k] = np.log(self.pi[k]) + \
                              multivariate_normal.logpdf(X, self.mu[k], self.Sigma[k])
        return log_gamma.argmax(axis=1)

# 데이터 생성
np.random.seed(42)
N = 500
true_K = 3
true_mu = [np.array([0, 0]), np.array([4, 4]), np.array([-3, 3])]
true_Sigma = [np.eye(2), np.eye(2) * 0.5, np.eye(2) * 2]
true_pi = [0.4, 0.3, 0.3]

X = []
labels = []
for n in range(N):
    k = np.random.choice(true_K, p=true_pi)
    X.append(np.random.multivariate_normal(true_mu[k], true_Sigma[k]))
    labels.append(k)
X = np.array(X)
labels = np.array(labels)

# EM
gmm = GMM_EM(K=3, n_features=2)
history = gmm.fit(X, n_iter=50, seed=1)
pred = gmm.predict(X)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
axes[0].set_title('True clusters')
axes[0].grid(alpha=0.3)

axes[1].scatter(X[:, 0], X[:, 1], c=pred, cmap='viridis', alpha=0.6)
for k in range(3):
    axes[1].scatter(gmm.mu[k, 0], gmm.mu[k, 1], c='red', s=200, marker='x', linewidths=3)
axes[1].set_title('EM-learned clusters (red X = mean)')
axes[1].grid(alpha=0.3)

axes[2].plot(history, 'o-', markersize=3)
axes[2].set_xlabel('EM iteration')
axes[2].set_ylabel('log-likelihood')
axes[2].set_title('Monotonic log-likelihood increase')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gmm_em.png', dpi=120, bbox_inches='tight')
plt.show()

# 검증
print(f"Final log-likelihood: {history[-1]:.2f}")
print(f"Monotonic? {all(history[i] <= history[i+1] + 1e-6 for i in range(len(history)-1))}")

# Learned parameters
print(f"\nLearned π: {gmm.pi}")
print(f"Learned μ:")
for k in range(3):
    print(f"  Cluster {k}: {gmm.mu[k]}")

# Multiple random restarts for robustness
print("\n여러 random init 후 best fit:")
best_logL = -np.inf
for seed in range(5):
    gmm_try = GMM_EM(K=3, n_features=2)
    hist = gmm_try.fit(X, n_iter=50, seed=seed)
    if hist[-1] > best_logL:
        best_logL = hist[-1]
        best_seed = seed
    print(f"  Seed {seed}: final log-L = {hist[-1]:.2f}")
print(f"Best seed: {best_seed}, log-L = {best_logL:.2f}")
```

**출력 예시**:
```
Final log-likelihood: -1847.23
Monotonic? True

Learned π: [0.302 0.396 0.302]
Learned μ:
  Cluster 0: [-3.04  3.01]
  Cluster 1: [ 0.05 -0.02]
  Cluster 2: [ 4.01  4.02]

여러 random init 후 best fit:
  Seed 0: final log-L = -1847.23
  Seed 1: final log-L = -1847.23
  Seed 2: final log-L = -1912.34
  Seed 3: final log-L = -1847.23
  Seed 4: final log-L = -1898.12
Best seed: 0, log-L = -1847.23
```

EM이 true clusters를 정확히 복원, log-likelihood monotonic 증가.

---

## 🔗 AI/ML 연결

### Baum-Welch (HMM EM)

Ch3-04에서 자세히. $z$ = hidden state sequence. E-step: Forward-Backward. M-step: count-based.

### Variational EM (General)

True posterior intractable → **approximate E-step** with variational $q$:

- **Mean-field**: $q = \prod q_i$
- **Structured MF**: partial factorization
- **Sample-based**: Monte Carlo EM

**ELBO**: Lower bound on $\log p(x | \theta)$. Maximize ELBO over $(q, \theta)$.

### LDA Variational EM (Ch7-04의 예고)

LDA variational EM:
- E-step: MF update on $\theta_d, \phi_k, z_n$ (각 document, topic, word)
- M-step: Dirichlet hyperparameters ($\alpha$, $\eta$) updates

Classic, scalable for document collections.

### VAE Training

VAE = "amortized EM":
- E-step: encoder $q_\phi(z | x)$ — learned, not iterative
- M-step: decoder $p_\theta(x | z)$ + encoder weights update

**Difference from EM**:
- Both $q$ and $p$ parameters updated simultaneously (not strict EM)
- Reparameterization trick for backprop
- Posterior approximated, not exact

### Diffusion Model Training

DDPM의 variational lower bound:
$$-\log p(x_0) \leq \mathbb{E}_q\left[-\log p(x_T) + \sum_t \text{KL}(q(x_{t-1} | x_t, x_0) \| p(x_{t-1} | x_t))\right]$$

이는 EM의 lower bound와 구조 동일 — latent path $x_{1:T}$의 posterior (Gaussian, fixed by forward process) + parameters.

Training이 EM의 한 step만 반복적으로.

### Factor Analysis, PCA, pPCA

**Probabilistic PCA**:
$$z \sim \mathcal{N}(0, I), \quad x | z \sim \mathcal{N}(W z + \mu, \sigma^2 I)$$

**EM for pPCA**:
- E-step: $q(z | x) = \mathcal{N}(\cdot)$ (closed form)
- M-step: closed form update on $W, \sigma^2, \mu$

$W$ at convergence = principal components (up to rotation).

### Gaussian Mixture as K-means Soft Version

**K-means**: hard assignments.
**GMM EM**: soft assignments (responsibilities).

Special case: $\Sigma_k = \sigma^2 I$ with $\sigma \to 0$ → GMM reduces to K-means.

EM framework unifies them.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Tractable $p(z | x, \theta)$ | Intractable → variational EM (Ch6) |
| Tractable M-step | Complex $\theta$은 GEM or numerical opt |
| Local optimum | Multiple random inits 필요 |
| Parameter identifiable | Label switching, rotation ambiguity |

**주의**: EM의 **local optimum** 문제는 심각. GMM의 $K$ 증가 시 여러 local minima. 실제로는 **K-means initialization**, **small random perturbation**으로 완화.

---

## 📌 핵심 정리

$$\boxed{\text{E-step: } q^{(t)}(z) = p(z | x, \theta^{(t)}); \quad \text{M-step: } \theta^{(t+1)} = \arg\max_\theta \mathbb{E}_q[\log p(x, z | \theta)]}$$

$$\boxed{\log p(x | \theta) = \text{ELBO}(\theta, q) + \text{KL}(q \| p(z | x, \theta))}$$

| 특수 경우 | E-step | M-step |
|-----------|--------|--------|
| GMM | Responsibility $\gamma_{nk}$ | Mean, cov update |
| HMM (Baum-Welch) | Forward-Backward | Transition, emission |
| LDA (variational) | Dirichlet+Cat update | Prior hyperparameters |
| VAE | Encoder forward | Decoder + encoder gradient |
| Diffusion | Forward noising (fixed) | Denoiser training |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GMM EM에서 $K = 1$ (single Gaussian)일 때 EM이 첫 iteration에서 MLE에 도달함을 보여라.

<details>
<summary>힌트 및 해설</summary>

$K = 1$: 모든 $z_n = 1$ (deterministic). Single Gaussian $\mathcal{N}(\mu, \Sigma)$.

**E-step**: $\gamma_{n, 1} = 1$ for all $n$ (no ambiguity).

**M-step**: 
- $\pi_1 = 1$
- $\mu = \frac{\sum_n \gamma_{n, 1} x_n}{\sum_n \gamma_{n, 1}} = \frac{1}{N} \sum_n x_n$ = sample mean
- $\Sigma = \frac{\sum_n (x_n - \mu)(x_n - \mu)^T}{N}$ = sample covariance

이는 **MLE for single Gaussian** — 1 iteration에 도달.

**$K > 1$**: Multiple iterations 필요. Soft responsibilities가 계속 refine, cluster도 점점 separate.

**의미**: EM의 **converge 속도**는 문제 structure에 따라 다름. Single cluster는 trivial, multi-cluster는 non-convex optimization.

</details>

**문제 2** (심화): EM의 local optimum 문제를 완화하는 방법 3가지를 수학적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**1. Multiple Random Initialization**:
- Run EM from $M$ different random initial points
- Pick best final log-likelihood
- **Theoretical**: Higher probability of finding global optimum
- **Practical**: $M = 10-50$ typically

**2. K-means Initialization**:
- Run K-means first (hard assignments)
- Use K-means centroids as initial $\mu_k$
- Responsibilities initialized by K-means clusters
- **Reason**: K-means의 good solution이 EM starting point로 좋음

**3. Annealing / Deterministic Annealing**:
- Initial high "temperature" $T$: soften posterior $q(z) \propto p(z | x)^{1/T}$
- Gradually decrease $T \to 1$
- High $T$: smooth landscape, easy to find global region
- Low $T$: original EM for refinement

**다른 방법들**:

**4. Mixture Split / Merge**:
- EM converges → diagnose problematic components
- Split degenerate, merge similar
- RJMCMC-like (Ch6-06) local moves within EM

**5. Bayesian EM (MAP-EM)**:
- Add prior regularization: $\theta^{\text{MAP}} = \arg\max [\log p(x | \theta) + \log p(\theta)]$
- Prior prevents degenerate solutions (e.g., $\Sigma_k \to 0$)

**6. Stochastic EM**:
- Mini-batch E-step: noise escapes shallow local optima
- Monte Carlo EM: sample from $q$ instead of exact expectation

**Convergence Guarantees**:
- EM의 local convergence: standard (정리 2.4)
- Global convergence: **not guaranteed** in general
- For specific models (e.g., GMM with well-separated clusters): global convergence under conditions

**현대적 관점**:
- Deep learning의 SGD도 non-convex optimization — saddle point problem이 더 치명적
- EM의 local optima는 saddle에서 벗어나기 쉬움 (coordinate ascent)
- Practical: 여러 random inits의 성공률이 deep learning보다 EM에서 높은 경우 많음

</details>

**문제 3** (AI 연결): VAE의 ELBO loss가 EM의 M-step objective와 어떻게 대응되며, 왜 VAE에서는 "E-step이 iterative하지 않고 amortized"인지 설명하라.

<details>
<summary>힌트 및 해설</summary>

**EM ELBO**:
$$\text{ELBO}(\theta, q) = \mathbb{E}_q[\log p(x, z | \theta)] - \mathbb{E}_q[\log q(z)]$$

EM:
- E-step: $q^* = p(z | x, \theta)$ (exact posterior, closed form for conjugate)
- M-step: $\theta^* = \arg\max \text{ELBO}$

**VAE setup**: 
- $p_\theta(x | z)$: decoder network
- $p(z)$: prior (usually $\mathcal{N}(0, I)$)
- **Posterior $p(z | x, \theta)$ intractable** (decoder is neural net, no closed form)

**Solution — Amortized E-step**:
- Parametric approximation: $q_\phi(z | x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi(x)^2 I)$
- $\phi$ = encoder network parameters
- For each $x$, **single forward pass** gives $q_\phi(z | x)$ — no iterative optimization

**ELBO with amortization**:
$$\text{ELBO}(\theta, \phi) = \mathbb{E}_{q_\phi(z | x)}[\log p_\theta(x | z)] - \text{KL}(q_\phi(z | x) \| p(z))$$

**Joint optimization**:
- Not strict EM (EM alternates, VAE joint)
- $\theta$ and $\phi$ updated **simultaneously** by SGD on $\text{ELBO}$
- Reparameterization $z = \mu + \sigma \odot \epsilon$ for differentiable sampling

**왜 "amortized"**:
- Classic EM: 각 data point마다 $q^*$ compute — iterative, expensive for $N$ points
- VAE: encoder가 **all possible $x$에 대한 posterior approximation** 학습 — $N$-shot $q$ 계산이 single network parameterization으로
- Test time: forward pass만 필요

**Trade-off**:
- Amortization gap: $q_\phi \neq p_\theta(z|x)$ (approximation error)
- Suboptimality per individual $x$
- But scalability, end-to-end training

**이론적 위치**:
- VAE = Mean-field VI + amortization + reparameterization
- Modern Bayesian deep learning의 기반
- Normalizing flow, diffusion model 모두 variational framework 확장

**EM에서 VAE로의 evolution**:
```
Classic EM (exact posterior, iterative)
    ↓
Variational EM (approximate posterior, iterative)
    ↓
Amortized VI / VAE (approximate posterior, parametric, joint)
    ↓
Normalizing Flow / Diffusion (flexible posterior/generator, joint)
```

이 evolution이 **latent variable learning의 deep learning 시대 진입**을 나타냄.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Maximum Likelihood for Graphical Models](./01-mle-for-graphical-models.md) | [📚 README](../README.md) | [03. Structure Learning ▶](./03-structure-learning.md) |

</div>
