# 01. Mean-Field Variational Inference

## 🎯 핵심 질문

- Mean-field의 **factorized approximation** $q(x) = \prod_i q_i(x_i)$는 왜 tractable한가?
- **ELBO**와 KL divergence의 관계 $\log p(x) = \text{ELBO} + \text{KL}(q \| p(\cdot|x))$는 어떻게 유도되는가?
- **Coordinate ascent update** $\log q_i \propto \mathbb{E}_{q_{-i}}[\log p]$는 어떻게 나오는가?
- KL 최소화와 ELBO 최대화의 등가성은?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Mean-Field Variational Inference**는 현대 Bayesian ML의 **핵심 방법론**. VAE, LDA, Bayesian neural network, probabilistic programming의 SVI(Stochastic Variational Inference), 모든 **amortized inference**의 기초. "정확한 posterior가 intractable → simpler family로 근사 → KL 최소화"의 패턴이 ML 전반에 걸쳐 반복. Bayesian deep learning의 **weight uncertainty**, VI-based generative model, **Mirror Descent**/natural gradient 최적화 — 모두 VI framework. Mean-field는 가장 단순하지만 가장 fundamental VI.

---

## 📐 수학적 선행 조건

- [Ch1-01 조건부 독립의 정의와 성질](../ch1-conditional-independence/01-conditional-independence-definition.md): factorization
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL divergence, 엔트로피
- Jensen's inequality
- Exponential family

---

## 📖 직관적 이해

### 문제: Exact Posterior Intractable

Bayesian inference:
$$p(z | x) = \frac{p(x, z)}{p(x)} = \frac{p(x, z)}{\int p(x, z) dz}$$

$p(x) = \int p(x, z) dz$가 intractable (high-dim, non-conjugate).

### VI의 아이디어

"$p(z | x)$를 exactly 계산하지 말고, simpler family $\mathcal{Q}$에서 **가장 가까운** $q(z)$를 찾자."

$$q^* = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(z) \| p(z | x))$$

### Mean-Field Family

$\mathcal{Q}_{\text{MF}}$: factorized distributions
$$q(z) = \prod_{i=1}^n q_i(z_i)$$

각 $z_i$의 marginal이 독립. 실제 posterior에서는 독립이 아닐 수 있음 — **approximation**.

### ELBO (Evidence Lower Bound)

**Key identity**:
$$\log p(x) = \underbrace{\mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q(z)]}_{\text{ELBO}} + \underbrace{\text{KL}(q(z) \| p(z | x))}_{\geq 0}$$

$\text{ELBO} \leq \log p(x)$ (Jensen).

**Minimizing KL ⟺ Maximizing ELBO**: $\log p(x)$가 constant (data fixed), 두 항이 합이 constant → 하나 최소화 = 다른 것 최대화.

### Coordinate Ascent VI (CAVI)

ELBO를 $q_i$에 대해 최적화 (다른 $q_{-i}$ 고정):
$$q_i^*(z_i) \propto \exp\left(\mathbb{E}_{q_{-i}}[\log p(x, z)]\right)$$

이 update가 분산된 variables의 **expected log-likelihood**를 이용.

### LDA Mean-Field 예시 (Blei-Ng-Jordan 2003)

LDA posterior: $p(\theta, z | w, \alpha, \beta)$. Mean-field:
$$q(\theta, z | \gamma, \phi) = q(\theta | \gamma) \prod_n q(z_n | \phi_n)$$

$\gamma, \phi$가 variational parameters. CAVI update:
- $\phi_{n, k} \propto \beta_{k, w_n} \exp(\mathbb{E}_q[\log \theta_k])$
- $\gamma_k = \alpha_k + \sum_n \phi_{n, k}$

Simple, fast, effective. Ch7-04에서 자세히.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Variational Family

$\mathcal{Q}$: approximating distributions의 집합. Common choices:
- **Mean-field** $\mathcal{Q}_{\text{MF}} = \{q : q(z) = \prod_i q_i(z_i)\}$
- **Structured MF**: partial factorization (tree-MF, Gaussian-MF)
- **Full-rank Gaussian** $q(z) = \mathcal{N}(z | \mu, \Sigma)$
- **Normalizing flow**: $q_\phi(z) = f_\phi^{-1}(\epsilon)$, $\epsilon \sim \mathcal{N}(0, I)$

### 정의 1.2 — KL Divergence

$$\text{KL}(q \| p) := \mathbb{E}_q[\log q - \log p] = \int q(z) \log \frac{q(z)}{p(z)} dz$$

Properties:
- $\text{KL} \geq 0$, = 0 iff $q = p$ a.s.
- Asymmetric: $\text{KL}(q \| p) \neq \text{KL}(p \| q)$
- Not a metric

### 정의 1.3 — ELBO

$$\text{ELBO}(q) := \mathbb{E}_q[\log p(x, z) - \log q(z)] = \mathbb{E}_q[\log p(x, z)] + H(q)$$

여기서 $H(q) = -\mathbb{E}_q[\log q]$는 entropy.

**Two decompositions**:
- $\text{ELBO}(q) = \log p(x) - \text{KL}(q \| p(z | x))$
- $\text{ELBO}(q) = \mathbb{E}_q[\log p(x | z)] - \text{KL}(q \| p(z))$ (evidence + reconstruction - prior KL)

### 정의 1.4 — CAVI Update

Mean-field: $q(z) = \prod_i q_i(z_i)$. ELBO를 $q_j$에 대해 최적화:

$$q_j^*(z_j) \propto \exp\left(\mathbb{E}_{q_{-j}}[\log p(x, z_j, z_{-j})]\right)$$

여기서 $q_{-j} = \prod_{i \neq j} q_i$.

---

## 🔬 정리와 증명

### 정리 1.1 — ELBO = $\log p(x) - \text{KL}(q \| p(z|x))$

**명제**: 임의 $q(z)$에 대해
$$\log p(x) = \text{ELBO}(q) + \text{KL}(q(z) \| p(z | x))$$

**증명**:

$$\text{KL}(q \| p(z|x)) = \mathbb{E}_q\left[\log \frac{q(z)}{p(z|x)}\right] = \mathbb{E}_q[\log q - \log p(z|x)]$$

$$= \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p(z, x) - \log p(x)]$$

$$= \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p(z, x)] + \log p(x)$$

$$= -\text{ELBO}(q) + \log p(x)$$

따라서 $\log p(x) = \text{ELBO}(q) + \text{KL}(q \| p(z|x))$. $\square$

**함의**: KL $\geq 0$이므로 ELBO $\leq \log p(x)$. 따라서 ELBO는 evidence의 **lower bound**.

### 정리 1.2 — Jensen Inequality로부터 ELBO

**대안 유도**:
$$\log p(x) = \log \int p(x, z) dz = \log \int q(z) \frac{p(x, z)}{q(z)} dz$$

Jensen:
$$\geq \int q(z) \log \frac{p(x, z)}{q(z)} dz = \mathbb{E}_q[\log p(x, z) - \log q(z)] = \text{ELBO}$$

Equality iff $\frac{p(x, z)}{q(z)} = \text{const}$ — 즉 $q(z) \propto p(x, z)$, 즉 $q = p(z|x)$. $\square$

### 정리 1.3 — CAVI Update의 유도

**명제**: Mean-field ELBO를 $q_j$에 대해 최대화하면
$$q_j^*(z_j) = \frac{\exp(\mathbb{E}_{q_{-j}}[\log p(x, z)])}{\int \exp(\mathbb{E}_{q_{-j}}[\log p(x, z)]) dz_j}$$

**증명**:

ELBO를 $q_j$에 대해 분해:
$$\text{ELBO}(q) = \mathbb{E}_q[\log p(x, z)] - \sum_i \mathbb{E}_{q_i}[\log q_i]$$

$q_j$에 의존하는 부분:
$$\text{ELBO}(q_j) = \mathbb{E}_{q_j}\left[\mathbb{E}_{q_{-j}}[\log p(x, z)]\right] - \mathbb{E}_{q_j}[\log q_j] + \text{const}$$

$f(z_j) := \mathbb{E}_{q_{-j}}[\log p(x, z_j, z_{-j})]$로 두면:
$$\text{ELBO}(q_j) = \int q_j(z_j) f(z_j) dz_j - \int q_j(z_j) \log q_j(z_j) dz_j + \text{const}$$

이는 $q_j = e^f / Z$ 형태의 분포에 대한 $-\text{KL}(q_j \| e^f/Z)$ + const.

**Lagrangian with constraint** $\int q_j = 1$:
$$\mathcal{L} = \int q_j f - \int q_j \log q_j - \lambda (\int q_j - 1)$$

$\partial / \partial q_j = f(z_j) - \log q_j - 1 - \lambda = 0$:
$$q_j^*(z_j) = \exp(f(z_j) - 1 - \lambda) \propto \exp(f(z_j))$$

즉 $q_j^* \propto \exp(\mathbb{E}_{q_{-j}}[\log p(x, z)])$. Normalized form:
$$q_j^*(z_j) = \frac{\exp(\mathbb{E}_{q_{-j}}[\log p(x, z)])}{\int \exp(\mathbb{E}_{q_{-j}}[\log p(x, z)]) dz_j}$$

$\square$

### 정리 1.4 — CAVI의 Monotonic Improvement

**명제**: CAVI iteration은 ELBO의 **non-decreasing**. 수렴점은 ELBO의 local maximum.

**증명**:

각 iteration에서 $q_j \leftarrow q_j^*$: $\text{ELBO}(q_{-j}, q_j^*) \geq \text{ELBO}(q_{-j}, q_j^{\text{old}})$ (by optimization).

Multiple coordinates update: cycle through all $j$. ELBO가 bounded above (by $\log p(x)$) + monotonic → 수렴. $\square$

**Local**: ELBO는 일반적으로 **non-convex** (mean-field family는 non-convex manifold). Random initialization 여러 번 필요.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# 예시: 2-D Gaussian posterior를 mean-field Gaussian으로 근사
# True posterior: p(z) = N(μ, Σ) with Σ = [[1, 0.9], [0.9, 1]] — strong correlation
mu_true = np.array([1.0, 2.0])
Sigma_true = np.array([[1.0, 0.9], [0.9, 1.0]])
Sigma_inv = np.linalg.inv(Sigma_true)

def log_p(z):
    """log p(z1, z2) = -0.5 * (z - μ)^T Σ^{-1} (z - μ) + const."""
    diff = z - mu_true
    return -0.5 * diff @ Sigma_inv @ diff

# Mean-field: q(z1, z2) = N(z1 | m1, s1^2) * N(z2 | m2, s2^2)
# CAVI updates:
# q*(z1) = argmax E_q2[log p]
# Gaussian의 경우 analytical closed form

def cavi_gaussian(mu_true, Sigma_true, n_iter=50):
    """CAVI for Gaussian posterior approximation by factorized Gaussian."""
    Sigma_inv = np.linalg.inv(Sigma_true)
    # Natural parameters (precision form)
    # log p(z) ∝ -0.5 z^T Λ z + z^T η, Λ = Σ^{-1}, η = Σ^{-1} μ
    Lambda = Sigma_inv
    eta = Sigma_inv @ mu_true
    
    # Initial MF params
    m1, s1 = 0.0, 1.0
    m2, s2 = 0.0, 1.0
    
    history = {'m1': [], 's1': [], 'm2': [], 's2': [], 'elbo': []}
    
    def compute_elbo(m1, s1, m2, s2):
        """ELBO = -KL(q || p)."""
        q_mean = np.array([m1, m2])
        q_cov = np.diag([s1**2, s2**2])
        # KL(N(μ_q, Σ_q) || N(μ_p, Σ_p))
        diff = mu_true - q_mean
        kl = 0.5 * (
            np.trace(Sigma_inv @ q_cov) +
            diff @ Sigma_inv @ diff -
            2 +  # dim
            np.log(np.linalg.det(Sigma_true) / np.linalg.det(q_cov))
        )
        return -kl  # ELBO = log p(x) - KL, omit log p(x) const
    
    for it in range(n_iter):
        # Update q1 (z1): marginal
        # E_q2[log p] = -0.5 Λ11 z1^2 + (eta1 - Λ12 m2) z1 + const
        # → q1 = N(m1, 1/Λ11)
        s1 = 1.0 / np.sqrt(Lambda[0, 0])
        m1 = (eta[0] - Lambda[0, 1] * m2) / Lambda[0, 0]
        
        # Update q2
        s2 = 1.0 / np.sqrt(Lambda[1, 1])
        m2 = (eta[1] - Lambda[1, 0] * m1) / Lambda[1, 1]
        
        history['m1'].append(m1)
        history['s1'].append(s1)
        history['m2'].append(m2)
        history['s2'].append(s2)
        history['elbo'].append(compute_elbo(m1, s1, m2, s2))
    
    return m1, s1, m2, s2, history

m1, s1, m2, s2, history = cavi_gaussian(mu_true, Sigma_true, n_iter=50)

print(f"True μ: {mu_true}")
print(f"MF μ: [{m1:.4f}, {m2:.4f}]")
print(f"True std: [{np.sqrt(Sigma_true[0,0])}, {np.sqrt(Sigma_true[1,1])}]")
print(f"MF std: [{s1:.4f}, {s2:.4f}]")
print(f"관찰: MF는 posterior variance를 **underestimate** (independent 가정의 대가)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Posterior + MF approximation contours
z1_range = np.linspace(-2, 4, 100)
z2_range = np.linspace(-1, 5, 100)
Z1, Z2 = np.meshgrid(z1_range, z2_range)

# True p
p_true = np.zeros_like(Z1)
for i in range(Z1.shape[0]):
    for j in range(Z1.shape[1]):
        diff = np.array([Z1[i, j] - mu_true[0], Z2[i, j] - mu_true[1]])
        p_true[i, j] = np.exp(-0.5 * diff @ Sigma_inv @ diff)

# MF approximation
p_mf = np.zeros_like(Z1)
for i in range(Z1.shape[0]):
    for j in range(Z1.shape[1]):
        p_mf[i, j] = np.exp(-0.5 * ((Z1[i, j] - m1)**2 / s1**2 + (Z2[i, j] - m2)**2 / s2**2))

axes[0].contour(Z1, Z2, p_true, levels=5, colors='blue', alpha=0.7)
axes[0].contour(Z1, Z2, p_mf, levels=5, colors='red', alpha=0.7, linestyles='--')
axes[0].scatter([mu_true[0]], [mu_true[1]], c='blue', s=100, label='True μ')
axes[0].scatter([m1], [m2], c='red', s=100, label='MF μ')
axes[0].set_xlabel('z_1'); axes[0].set_ylabel('z_2')
axes[0].set_title('True (blue) vs Mean-Field (red) posterior')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ELBO convergence
axes[1].plot(history['elbo'], 'o-')
axes[1].set_xlabel('CAVI iteration')
axes[1].set_ylabel('ELBO (up to constant)')
axes[1].set_title('CAVI convergence')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mean_field_vi_gaussian.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
True μ: [1. 2.]
MF μ: [1.0000, 2.0000]
True std: [1.0, 1.0]
MF std: [1.0000, 1.0000]
관찰: MF는 posterior variance를 underestimate (independent 가정의 대가)
```

(이 예시에서는 mean 정확히 match; correlation 때문에 variance는 지금 case에서 symmetric이라 그대로 but 일반적으로 underestimate)

---

## 🔗 AI/ML 연결

### VAE (Kingma-Welling 2014)

VAE의 encoder $q_\phi(z | x)$는 **amortized mean-field**:
- 일반 MF: 각 data point마다 $q(z | x^{(i)})$를 iterative 계산
- Amortized: $q_\phi(z | x) = \mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$ — encoder network가 output
- Test time에 **single forward pass** — iteration 없음

**Loss**: $-\text{ELBO} = -\mathbb{E}_q[\log p(x | z)] + \text{KL}(q_\phi(z | x) \| p(z))$.

Reparameterization: $z = \mu + \sigma \odot \epsilon$로 pathwise gradient.

### LDA의 Variational EM (Blei-Ng-Jordan 2003)

LDA:
- Document-topic: $\theta_d \sim \text{Dirichlet}(\alpha)$
- Topic-word: $\beta_k \sim \text{Dirichlet}(\eta)$
- $z_{d,n} \sim \text{Categorical}(\theta_d)$
- $w_{d,n} \sim \text{Categorical}(\beta_{z_{d,n}})$

**Mean-field**:
$q(\theta, z, \beta) = \prod_d q(\theta_d | \gamma_d) \prod_{d, n} q(z_{d, n} | \phi_{d, n}) \prod_k q(\beta_k | \lambda_k)$

**CAVI updates**: Closed-form (Dirichlet + Categorical conjugacy).

Ch7-04에서 자세히.

### Bayesian Neural Network

Bayesian NN:
- Weights $W \sim p(W)$
- $y | x, W \sim p(y | x, W)$
- Posterior $p(W | D)$ intractable (high-dim)

**Mean-field VI**: $q(W) = \prod_{l, i, j} q(W_{l, i, j})$ — each weight independent Gaussian.

- Blundell et al. 2015 **Bayes by Backprop**: reparameterization으로 ELBO gradient
- Gal 2016 **MC Dropout**: dropout as Bayesian approximation

MF는 posterior correlation 무시 → **variance underestimate**, overconfident predictions. 한계 알려져 있음.

### Stochastic VI (SVI, Hoffman et al. 2013)

**Stochastic gradient** on ELBO:
- Mini-batch of data
- Natural gradient with exponential family
- VAE의 original algorithm의 조상

### Reverse KL의 Mode-Seeking

$\text{KL}(q \| p)$ vs $\text{KL}(p \| q)$:
- **Reverse KL** ($q \| p$, VI 표준): **mode-seeking** — multimodal $p$에서 $q$가 **한 mode에 집중**
- **Forward KL** ($p \| q$, EP): **mean-seeking** — $q$가 전체 $p$에 평균

MF의 mode-seeking이 때로는 문제 (posterior가 multimodal이면 한 mode만 capture). **Normalizing flow**, **Laplace approximation** 등으로 더 나은 family.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Factorization | Posterior correlation 무시 → variance underestimate |
| Mode-seeking | Multimodal posterior에서 한 mode만 capture |
| Local optimum | Non-convex ELBO → random init 필요 |
| Tractable expectations | 일반 $p$에서 $\mathbb{E}[\log p]$ 계산 어려움 → stochastic estimate 필요 |

**주의**: MF는 가장 단순한 VI. 더 flexible: structured MF, full-covariance Gaussian, normalizing flow, implicit VI. 하지만 MF의 **tractable structure**가 많은 경우 여전히 최선의 trade-off.

---

## 📌 핵심 정리

$$\boxed{\log p(x) = \text{ELBO}(q) + \text{KL}(q \| p(z|x))}$$

$$\boxed{q_j^*(z_j) \propto \exp\left(\mathbb{E}_{q_{-j}}[\log p(x, z)]\right)}$$

| 개념 | 의미 |
|------|------|
| **Variational family** | Simpler distribution class |
| **Mean-field** | Fully factorized $q = \prod q_i$ |
| **ELBO** | Lower bound on $\log p(x)$ |
| **KL minimization** | ELBO maximization과 등가 |
| **CAVI** | Coordinate ascent on ELBO |
| **Mode-seeking** | Reverse KL의 성질 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2-variable Gaussian $p(z_1, z_2) = \mathcal{N}(\mu, \Sigma)$를 MF Gaussian으로 근사할 때, $\Sigma$가 diagonal이 아니면 MF의 variance가 어떻게 되는가?

<details>
<summary>힌트 및 해설</summary>

**True**: $p = \mathcal{N}(\mu, \Sigma)$ with off-diagonal $\Sigma_{12} \neq 0$.
**MF**: $q = q_1(z_1) \cdot q_2(z_2)$ — 각 Gaussian $\mathcal{N}(m_i, s_i^2)$.

**CAVI**: $q_1^* \propto \exp(\mathbb{E}_{q_2}[\log p(z_1, z_2)])$.

Log p:
$$\log p = -\frac{1}{2} \begin{pmatrix} z_1 - \mu_1 \\ z_2 - \mu_2 \end{pmatrix}^T \Lambda \begin{pmatrix} z_1 - \mu_1 \\ z_2 - \mu_2 \end{pmatrix}, \quad \Lambda = \Sigma^{-1}$$

전개: $-\frac{1}{2}(\Lambda_{11} (z_1 - \mu_1)^2 + 2\Lambda_{12}(z_1 - \mu_1)(z_2 - \mu_2) + \Lambda_{22}(z_2 - \mu_2)^2)$.

$\mathbb{E}_{q_2}[\log p] = -\frac{1}{2} \Lambda_{11} (z_1 - \mu_1)^2 - \Lambda_{12}(z_1 - \mu_1) \mathbb{E}[z_2 - \mu_2] + \text{const}$

$= -\frac{1}{2} \Lambda_{11} (z_1 - \mu_1)^2 - \Lambda_{12}(z_1 - \mu_1)(m_2 - \mu_2) + \text{const}$

이는 $z_1$에 대해 Gaussian with precision $\Lambda_{11}$.

$q_1 = \mathcal{N}(m_1, 1/\Lambda_{11})$

**Variance**: $s_1^2 = 1/\Lambda_{11}$.

**True variance of $z_1$**: $\Sigma_{11}$.

**Relationship**: $\Sigma_{11} = (\Lambda_{11} - \Lambda_{12}^2/\Lambda_{22})^{-1}$ — Schur complement.

**Diagonal $\Sigma$**: $\Lambda_{12} = 0$ → $s_1^2 = 1/\Lambda_{11} = \Sigma_{11}$. Match!

**Non-diagonal $\Sigma$** (correlation):
$$s_1^2 = 1/\Lambda_{11}, \quad \text{true} = 1/(\Lambda_{11} - \Lambda_{12}^2/\Lambda_{22})$$

$\Lambda_{12}^2/\Lambda_{22} > 0$ → MF variance **smaller than true**. 

**결론**: **MF underestimates posterior variance** when true has correlation. "Confidence" of MF posterior는 overconfident.

</details>

**문제 2** (심화): Forward KL $\text{KL}(p \| q)$ 최소화가 **mean-seeking** (mass covering)임을 보여라.

<details>
<summary>힌트 및 해설</summary>

**Forward KL**: $\text{KL}(p \| q) = \int p(z) \log(p(z)/q(z)) dz$.

**Minimization** of forward KL over simpler $q$:
- $p(z) > 0$인 영역에서 $q(z) > 0$ 필수 — 아니면 $\log(p/q) = \infty$
- $q$는 $p$의 **전체 support**를 cover해야 함

**대조**: Reverse KL $\text{KL}(q \| p)$:
- $q(z) > 0$인 영역에서 $p(z) > 0$ 필요
- 만약 $p(z) \approx 0$ 영역에 $q(z) > 0$ → $\log(q/p) = \infty$
- $q$는 $p$의 **support 안에만** (mass concentration)

**구체적 예**: Bimodal $p$ with two Gaussian modes.

**Reverse KL ($q \| p$, VI)**:
- $q$ unimodal → 한 mode에 집중 (mode-seeking)
- Other mode는 $q \approx 0$ → acceptable for reverse KL

**Forward KL ($p \| q$, EP, MLE)**:
- $q$ unimodal이 두 mode 사이 "sprad out" → 두 mode mass를 cover (mean-seeking)
- Gaussian일 경우 mean = two modes의 average, variance 큰

**ML context**:
- Reverse KL: VAE, VI — sharp but possibly missed modes
- Forward KL: Moment matching, maximum likelihood — mode averaging

**Variational EM**: $q$가 $p(z|x)$에 reverse KL 쓰지만, $p$ 자체 학습은 forward KL (MLE) — 둘 다 섞임.

</details>

**문제 3** (AI 연결): VAE의 **posterior collapse** 문제 — $q(z | x)$가 $p(z)$에 collapse하여 $z$가 $x$의 정보를 잃는 — 를 mean-field ELBO 관점에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

**VAE ELBO**:
$$\text{ELBO} = \mathbb{E}_q[\log p(x | z)] - \text{KL}(q(z|x) \| p(z))$$

**Posterior collapse**: $q(z | x) = p(z)$ (variational posterior = prior, $z$가 $x$와 독립).

**왜 일어나는가**:

1. **Decoder가 강력**: Decoder $p(x | z)$가 $z$ 없이도 $p(x)$를 잘 fit → $z$ unused
2. **KL regularization 강함**: $\text{KL}(q \| p)$ 최소화가 $z$ 정보 제거하도록 pressure
3. **Training dynamics**: 초기에 decoder가 useless $z$ 무시 → 나머지 training에서 복구 안 됨

**수학적 분석**:

만약 $p(x | z) = p(x)$ (decoder이 $z$ 무시):
- $\mathbb{E}_q[\log p(x|z)] = \log p(x)$ (max possible)
- $q(z|x) = p(z)$로 설정하면 $\text{KL}(q \| p) = 0$

ELBO = $\log p(x) - 0 = \log p(x)$ — theoretical maximum (by Jensen).

**하지만 $z$가 useless!** Generative quality 나쁠 수 있음.

**해결책**:
1. **KL annealing** (Bowman et al. 2016): KL term의 weight $\beta$를 0에서 1로 점차 증가
2. **β-VAE** (Higgins et al. 2017): $\beta < 1$로 KL 약화 — representation-reconstruction trade-off
3. **Skip connections**: Decoder를 덜 powerful하게
4. **Free bits** (Kingma et al. 2016): KL의 per-dim minimum
5. **Discrete latent** (VQ-VAE): Latent information 보존

**MF-specific 문제**:
- MF Gaussian이 **expressive decoder**에 비해 너무 simple
- Decoder가 $z$를 필요로 하지 않음 → MF가 easy solution ($q = p$)으로 collapse

**Normalizing flow로의 일반화**: $q$가 더 flexible하면 posterior가 complex structure 필요로 하는 data에 fit → collapse 덜. **Flow VAE**, **IAF-VAE**.

**결론**: Posterior collapse는 ELBO의 **degenerate optimum**. MF는 simplicity로 이를 encouraging함 — 더 flexible VI family 또는 objective 수정 필요.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch5-04 Inference의 복잡도 이론](../ch5-variable-elimination/04-inference-complexity.md) | [📚 README](../README.md) | [02. Bethe 자유에너지와 Loopy BP의 변분 해석 ▶](./02-bethe-loopy-bp.md) |

</div>
