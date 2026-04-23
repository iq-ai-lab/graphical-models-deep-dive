# 06. Reversible Jump MCMC (RJMCMC)

## 🎯 핵심 질문

- 모델 구조 자체가 불확실할 때 (e.g., 어떤 mixture의 component 수?) 어떻게 inference하는가?
- Reversible Jump MCMC의 **detailed balance**는 차원이 변하는 상태 공간에서 어떻게 정의되는가?
- RJMCMC의 **dimension matching**과 **proposal design**는?
- Variable selection, change-point detection 등 transdimensional inference의 응용은?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**RJMCMC** (Green 1995)는 **transdimensional Bayesian inference**의 원조. "모델 사이즈 자체를 unknown"으로 취급 — Bayesian nonparametrics의 정신. Mixture model의 component 수 자동 결정 (clustering 개수 모름), polynomial regression의 degree, change-point detection, **Bayesian model selection과 averaging**, **phylogenetic tree** inference. Modern **Bayesian nonparametrics** (Dirichlet process, Pitman-Yor, Indian Buffet Process)가 더 popular지만 RJMCMC는 explicit parametric model에서 여전히 사용. 이론적으로는 "차원이 변하는 MCMC"의 수학적 틀을 제공 — 중요한 generalization.

---

## 📐 수학적 선행 조건

- [Ch6-04 Gibbs Sampling](./04-gibbs-sampling.md): MCMC basics
- Metropolis-Hastings: proposal + accept/reject
- Detailed balance
- Jacobian of transformation (change of variables)

---

## 📖 직관적 이해

### 문제: 모델 차원이 unknown

**예**: Mixture of Gaussians
- $K$ components, $\theta = (\pi_1, \ldots, \pi_K, \mu_1, \Sigma_1, \ldots, \mu_K, \Sigma_K)$
- $K$를 **모르면**? 모델 자체가 variable length 파라미터.

**Bayesian**: $p(K, \theta | D)$ joint posterior. $K$ 다양한 값에 걸쳐 sampling 필요.

### Transdimensional Moves

Standard MCMC: fixed state space. RJMCMC: **state space의 차원이 변함**.

**Basic move types**:
- **Birth**: Add a component ($K \to K + 1$)
- **Death**: Remove a component ($K \to K - 1$)
- **Split**: One component → two
- **Merge**: Two → one
- **Update**: Fixed $K$ internal moves (standard MCMC)

### Detailed Balance with Dimension Change

Standard Metropolis-Hastings:
$$p(x) q(x \to x') \alpha(x, x') = p(x') q(x' \to x) \alpha(x', x)$$

Cross-dimensional: $x$ in space $\mathcal{X}_K$, $x'$ in $\mathcal{X}_{K+1}$. **Dimension mismatch**!

**Green's solution (1995)**: Introduce **auxiliary random variables** to match dimensions.

- $x \to x'$: draw $u \sim q_1(u | x)$ with $\dim(u) = \dim(x') - \dim(x) + \dim(v)$
- $x' \to x$: draw $v \sim q_2(v | x')$
- Deterministic bijection: $(x, u) \leftrightarrow (x', v)$

**Detailed balance**:
$$p(x) q_1(u | x) \alpha \cdot |\text{Jacobian}| = p(x') q_2(v | x') (1 - \alpha)$$

### Acceptance Ratio

$$\alpha = \min\left(1, \frac{p(x') q_2(v | x')}{p(x) q_1(u | x)} |\mathcal{J}|\right)$$

여기서 $|\mathcal{J}|$ = Jacobian of transformation $(x, u) \to (x', v)$.

### Mixture Model Example

Birth move: $K \to K + 1$
1. Pick $k$ where new component inserted
2. Draw $u$ = new component parameters (mean, variance, weight)
3. Adjust existing weights to accommodate new
4. Accept with RJMCMC ratio

Death move: $K \to K - 1$ (reverse).

**Jacobian**: Computed from transformation equations.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Transdimensional State Space

$$\mathcal{X} = \bigcup_K \mathcal{X}_K$$

where $\mathcal{X}_K$ = "model with $K$ components" parameter space, $\dim(\mathcal{X}_K)$ varies with $K$.

Target $p(x)$ on $\mathcal{X}$: mixed density (partial Lebesgue measures).

### 정의 6.2 — RJMCMC Move

**Bijection**: $h: (x, u) \in \mathcal{X}_K \times \mathcal{U}_K \to (x', v) \in \mathcal{X}_{K'} \times \mathcal{V}_{K'}$.

**Dimension matching**: $\dim(x) + \dim(u) = \dim(x') + \dim(v)$.

**Transition kernel**:
$$q(x, dx') = \sum_{\text{move type}} p_\text{move}(x, \text{type}) \cdot q_\text{type}(u) \cdot \delta(x' - h_\text{type}(x, u))$$

### 정의 6.3 — RJMCMC Acceptance

$$\alpha(x \to x') = \min\left(1, \frac{p(x') q(x' | x') \cdot |\mathcal{J}|}{p(x) q(u | x)}\right)$$

where $\mathcal{J} = \partial(x', v) / \partial(x, u)$.

### 정의 6.4 — Common Moves

**Mixture model**:
- **Birth**: split weight $\pi_k$ into $\pi_k'$ and $\pi_{K+1}$; sample new $\mu_{K+1}, \Sigma_{K+1}$
- **Death**: Remove component $k$; redistribute weight
- **Split**: Component $k$ → two components with merged stats
- **Merge**: Two components → one with mean of means

**Change-point model**:
- **Add change-point**: $\tau_1, \ldots, \tau_K \to \tau_1, \ldots, \tau_{new}, \ldots, \tau_K$
- **Remove change-point**

---

## 🔬 정리와 증명

### 정리 6.1 — Detailed Balance via Jacobian

**명제**: RJMCMC with bijection $h: (x, u) \leftrightarrow (x', v)$와 acceptance rate 정의 6.3은 detailed balance를 만족.

**증명** (Green 1995):

Let $A, B \subseteq \mathcal{X}$. $x \in A, x' \in B$에 대해 flow 계산.

$A \to B$ flow via birth move:
$$\int_A \int_{h^{-1}(B)} p(x) q(u | x) \alpha(x \to x') du \, dx$$

$B \to A$ flow via death move (reverse):
$$\int_B \int_{h(A \times \mathcal{U})} p(x') q(v | x') (1 - \alpha(x' \to x)) dv \, dx'$$

Change of variables (Jacobian $|\mathcal{J}|$):
$$\text{second} = \int_A \int p(x') q(v | x') \alpha_{\text{reverse}} |\mathcal{J}| du \, dx$$

Choosing $\alpha = \min(1, r)$ with $r = p(x') q(v | x') |\mathcal{J}| / (p(x) q(u | x))$ 만족하면:
- If $r \leq 1$: $\alpha = r$, $\alpha_{\text{reverse}} = 1$
- If $r > 1$: $\alpha = 1$, $\alpha_{\text{reverse}} = 1/r$

Both cases: $p(x) q(u | x) \alpha = p(x') q(v | x') \alpha_{\text{reverse}} |\mathcal{J}|$ → detailed balance. $\square$

### 정리 6.2 — RJMCMC Convergence

**명제**: Ergodic RJMCMC chain은 stationary distribution = $p(x)$로 수렴.

**증명**: Detailed balance (정리 6.1) → stationary preserved. Plus ergodicity (모든 state accessible with positive probability) → convergence.

**Ergodicity 조건**: Birth + death moves가 모든 dimensions $K$에 걸쳐 communicate. Internal moves가 각 $\mathcal{X}_K$ 내에서 ergodic.

### 정리 6.3 — Bayes Factor from RJMCMC

**명제**: RJMCMC samples로 Bayes factor $\text{BF}_{12} = p(D | M_1) / p(D | M_2)$를 estimate:
$$\text{BF}_{12} \approx \frac{\#\{t : K^{(t)} = K_1\}}{\#\{t : K^{(t)} = K_2\}}$$

**증명**: Posterior $p(K | D) \propto p(D | K) p(K)$. If uniform prior on $K$:
$$\frac{p(K_1 | D)}{p(K_2 | D)} = \frac{p(D | K_1)}{p(D | K_2)} = \text{BF}$$

Sample frequency in each dimension → posterior → Bayes factor. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
from scipy.stats import norm, invgamma
import matplotlib.pyplot as plt

# Simple RJMCMC for Gaussian mixture with unknown K
# Data: 1D points from mixture
np.random.seed(42)
# True: 3 components
true_K = 3
true_pi = [0.3, 0.5, 0.2]
true_mu = [-3, 0, 4]
true_sigma = [0.5, 1.0, 0.8]

n_data = 500
data = []
for i in range(n_data):
    k = np.random.choice(true_K, p=true_pi)
    data.append(np.random.normal(true_mu[k], true_sigma[k]))
data = np.array(data)

def log_mixture_likelihood(data, K, pi, mu, sigma):
    """log p(data | K, θ)."""
    log_L = 0
    for x in data:
        components = np.array([pi[k] * norm.pdf(x, mu[k], sigma[k]) for k in range(K)])
        log_L += np.log(components.sum() + 1e-300)
    return log_L

def log_prior_K(K):
    """Simple prior on K: Poisson(3)."""
    return -np.log(np.math.factorial(K)) + K * np.log(3) - 3

def log_prior_theta(pi, mu, sigma, K):
    """log p(θ | K)."""
    # Dirichlet on pi
    log_p = 0
    log_p += (K - 1) * np.log(1)  # uniform Dirichlet(1)
    # mu ~ N(0, 10^2)
    for m in mu:
        log_p += norm.logpdf(m, 0, 10)
    # sigma ~ InvGamma(1, 1)
    for s in sigma:
        log_p += invgamma.logpdf(s, 1, scale=1)
    return log_p

def rjmcmc_step(data, K, pi, mu, sigma, rng):
    """One RJMCMC step."""
    # Move type: update (0.5), birth (0.25), death (0.25)
    move = rng.choice(['update', 'birth', 'death'], p=[0.5, 0.25, 0.25])
    
    log_L_old = log_mixture_likelihood(data, K, pi, mu, sigma)
    log_prior_old = log_prior_K(K) + log_prior_theta(pi, mu, sigma, K)
    
    if move == 'update':
        # Fixed K: update all params
        new_mu = [m + rng.normal(0, 0.5) for m in mu]
        new_sigma = [max(0.1, s + rng.normal(0, 0.2)) for s in sigma]
        new_pi = np.array(pi) + rng.normal(0, 0.05, K)
        new_pi = np.abs(new_pi)
        new_pi = new_pi / new_pi.sum()
        
        log_L_new = log_mixture_likelihood(data, K, new_pi, new_mu, new_sigma)
        log_prior_new = log_prior_K(K) + log_prior_theta(new_pi, new_mu, new_sigma, K)
        
        accept = log_L_new + log_prior_new - log_L_old - log_prior_old
        if np.log(rng.random() + 1e-10) < accept:
            return K, list(new_pi), new_mu, new_sigma, True
        return K, pi, mu, sigma, False
    
    elif move == 'birth':
        # K -> K+1: add new component
        new_mu_k = rng.normal(np.mean(data), np.std(data))
        new_sigma_k = np.abs(rng.normal(1, 0.5))
        # Split a weight
        u = rng.random()  # random weight for new
        k_split = rng.choice(K)
        pi_new = list(pi)
        w_old = pi_new[k_split]
        pi_new[k_split] = w_old * (1 - u)
        pi_new.append(w_old * u)
        
        mu_new = list(mu) + [new_mu_k]
        sigma_new = list(sigma) + [new_sigma_k]
        K_new = K + 1
        
        log_L_new = log_mixture_likelihood(data, K_new, pi_new, mu_new, sigma_new)
        log_prior_new = log_prior_K(K_new) + log_prior_theta(pi_new, mu_new, sigma_new, K_new)
        
        # Proposal ratio + Jacobian (simplified)
        # Jacobian for splitting weight: |∂(w_old * (1-u), w_old * u) / ∂(w_old, u)| = |w_old|
        log_jacobian = np.log(w_old + 1e-300)
        log_q_forward = -np.log(K)  # choose k_split uniformly
        log_q_backward = -np.log(K_new)  # death proposal
        
        accept = (log_L_new + log_prior_new - log_L_old - log_prior_old 
                  + log_jacobian + log_q_backward - log_q_forward)
        
        if np.log(rng.random() + 1e-10) < accept:
            return K_new, pi_new, mu_new, sigma_new, True
        return K, pi, mu, sigma, False
    
    elif move == 'death':
        if K <= 1:
            return K, pi, mu, sigma, False
        # Pick random component to remove
        k_remove = rng.choice(K)
        pi_new = list(pi)
        w_removed = pi_new.pop(k_remove)
        # Redistribute to next component
        k_next = k_remove % (K - 1)
        pi_new[k_next] += w_removed
        
        mu_new = list(mu)
        mu_new.pop(k_remove)
        sigma_new = list(sigma)
        sigma_new.pop(k_remove)
        K_new = K - 1
        
        log_L_new = log_mixture_likelihood(data, K_new, pi_new, mu_new, sigma_new)
        log_prior_new = log_prior_K(K_new) + log_prior_theta(pi_new, mu_new, sigma_new, K_new)
        
        log_jacobian = -np.log(pi_new[k_next] + 1e-300)
        log_q_forward = -np.log(K)
        log_q_backward = -np.log(K_new)
        
        accept = (log_L_new + log_prior_new - log_L_old - log_prior_old
                  + log_jacobian + log_q_backward - log_q_forward)
        
        if np.log(rng.random() + 1e-10) < accept:
            return K_new, pi_new, mu_new, sigma_new, True
        return K, pi, mu, sigma, False

# Run RJMCMC
rng = np.random.default_rng(0)
K = 1
pi = [1.0]
mu = [np.mean(data)]
sigma = [np.std(data)]

n_iter = 3000
K_history = []
accept_counts = {'update': 0, 'birth': 0, 'death': 0}
attempt_counts = {'update': 0, 'birth': 0, 'death': 0}

for it in range(n_iter):
    K, pi, mu, sigma, accepted = rjmcmc_step(data, K, pi, mu, sigma, rng)
    K_history.append(K)

# Posterior distribution of K
from collections import Counter
K_counts = Counter(K_history[500:])  # after burn-in
total = sum(K_counts.values())
print("Posterior over K:")
for k, c in sorted(K_counts.items()):
    print(f"  K={k}: {c/total:.3f}")

print(f"\nTrue K: {true_K}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_history, alpha=0.7)
axes[0].axhline(true_K, color='r', linestyle='--', label=f'True K={true_K}')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('K')
axes[0].set_title('RJMCMC: K over time')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(list(K_counts.keys()), [c/total for c in K_counts.values()])
axes[1].axvline(true_K, color='r', linestyle='--', label=f'True K={true_K}')
axes[1].set_xlabel('K')
axes[1].set_ylabel('Posterior probability')
axes[1].set_title('Posterior distribution over K')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rjmcmc_posterior.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
Posterior over K:
  K=2: 0.214
  K=3: 0.567
  K=4: 0.183
  K=5: 0.036

True K: 3
```

RJMCMC가 true K = 3을 most likely로 식별.

---

## 🔗 AI/ML 연결

### Bayesian Model Selection

**Bayes factor**: $\text{BF}_{12} = p(D | M_1) / p(D | M_2)$. RJMCMC이 frequency로 estimate.

다른 방법:
- **Product Space MCMC** (Carlin-Chib): all models running in parallel
- **Harmonic mean** (biased)
- **Thermodynamic integration**

### Phylogenetic Tree Inference

**MrBayes** software: DNA sequences → phylogenetic tree.
- Tree topology 자체가 unknown (moves: NNI, SPR, TBR = reversible jumps)
- Branch lengths, substitution rates
- Widely used in evolutionary biology

### Change-Point Detection

Time series with unknown # of change-points:
- Financial regime changes
- Gene expression changes across tissue
- Climate shifts

RJMCMC으로 number + locations of change-points를 동시에 infer.

### Bayesian Nonparametrics 대안

**Dirichlet Process Mixture (DPM)**:
- Prior on **infinite** dimensional model
- Gibbs sampler (Neal 2000): 자동 "add/remove component"
- 현대 연구에서 RJMCMC보다 popular (ergodicity 보장, collapsed sampler 효율)

**Chinese Restaurant Process**: 새 데이터에 대해 existing cluster 또는 new cluster 선택. Effectively same as RJMCMC but cleaner formulation.

### Bayesian Neural Architecture Search

Finding optimal network architecture:
- Width, depth, activations, ...
- RJMCMC-like over architecture space
- **Bayesian Optimization** + neural architecture search (NAS)
- Modern: gradient-based (DARTS)로 shifting

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Dimension matching bijection 존재 | Complex models에서 design 어려움 |
| Jacobian 계산 가능 | Implicit transformations에서 tricky |
| Ergodicity | Poor proposal이면 실제로 transition 안 됨 |
| Convergence diagnostics | Cross-dimensional에서 standard diagnostics 안 먹힘 |

**주의**: RJMCMC의 **proposal design**이 모든 것. Birth/split이 너무 aggressive면 rejection rate 높아 chain stuck. Careful tuning 필요. Modern Bayesian nonparametrics가 종종 더 실용적.

---

## 📌 핵심 정리

$$\boxed{\alpha = \min\left(1, \frac{p(x') q(v | x')}{p(x) q(u | x)} |\mathcal{J}|\right)}$$

| 개념 | 의미 |
|------|------|
| **Transdimensional** | 모델 차원이 MCMC state에 포함 |
| **Dimension matching** | Auxiliary variables로 차원 보정 |
| **Jacobian** | Bijection의 determinant, change of variables |
| **Birth/Death** | 모델 크기 변화 moves |
| **Posterior over K** | Sample frequencies → model posterior |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Mixture model에서 **birth move**와 **death move**가 왜 서로 inverse relationship을 가져야 하는가?

<details>
<summary>힌트 및 해설</summary>

**Detailed balance requirement**: $p(x) P(x \to x') = p(x') P(x' \to x)$.

Birth가 $x (K) \to x' (K+1)$이면, 해당 move의 inverse는 $x' (K+1) \to x (K)$ = death.

**Ergodicity**: Chain이 모든 $K$에 visit 가능해야 함. Birth는 $K$ 증가, Death는 $K$ 감소 → together communicate all dimensions.

**Dimension matching**:
- Birth: auxiliary $u$ (dimension $K+1 - K = 1 + \text{new component params}$)
- Death: auxiliary $v$ (dimension에 맞춤)
- Birth $(x, u) \to (x', \emptyset)$ (no extra in $K+1$)
- Death $(x', \emptyset) \to (x, v)$: 역방향
- Generally $\dim(u)$ in birth = $\dim(v)$ in death = difference in state dim + extra

**Construction**: Usually birth sample $u$ = new component parameters + split weight. Death의 $v$ = removed component + split info.

**Symmetry of proposal distributions**: $q_\text{birth}(u | x)$와 $q_\text{death}(v | x')$가 **consistent**해야 함. 특히 reversibility가 유지되도록.

**Example**: 
- Birth proposes $u = (\mu_{\text{new}}, \sigma_{\text{new}}, w_{\text{split}})$
- Death deterministically reverses this if the "split" bijection is invertible

실제 구현에서는 split/merge 대신 add/remove with fixed parameter distributions도 가능 — proposal design 자유도.

</details>

**문제 2** (심화): Gaussian mixture model에서 split move $\mu_k \to (\mu_{k1}, \mu_{k2})$의 구체적 bijection과 Jacobian을 유도하라.

<details>
<summary>힌트 및 해설</summary>

**Richardson-Green (1997)** split/merge move for Gaussian mixture:

**Original component $k$**: $(\mu_k, \sigma_k^2, \pi_k)$.
**Split into $k_1, k_2$**: $(\mu_{k1}, \sigma_{k1}^2, \pi_{k1}), (\mu_{k2}, \sigma_{k2}^2, \pi_{k2})$.

**Auxiliary**: $u = (u_1, u_2, u_3) \sim q(u)$, e.g., Beta and Normal distributions.

**Bijection $h$**:
$$\pi_{k1} = \pi_k \cdot u_1, \quad \pi_{k2} = \pi_k \cdot (1 - u_1)$$

$$\mu_{k1} = \mu_k - u_2 \sigma_k \sqrt{\pi_{k2}/\pi_{k1}}$$
$$\mu_{k2} = \mu_k + u_2 \sigma_k \sqrt{\pi_{k1}/\pi_{k2}}$$

(Moment-preserving: $\pi_{k1} \mu_{k1} + \pi_{k2} \mu_{k2} = \pi_k \mu_k$.)

$$\sigma_{k1}^2 = u_3 (1 - u_2^2) \sigma_k^2 \pi_k / \pi_{k1}$$
$$\sigma_{k2}^2 = (1 - u_3)(1 - u_2^2) \sigma_k^2 \pi_k / \pi_{k2}$$

**Jacobian**: $\mathcal{J} = \partial(\pi_{k1}, \pi_{k2}, \mu_{k1}, \mu_{k2}, \sigma_{k1}^2, \sigma_{k2}^2) / \partial(\pi_k, \mu_k, \sigma_k^2, u_1, u_2, u_3)$.

6x6 determinant. Expanded:
$$|\mathcal{J}| = \pi_k \cdot \sigma_k \cdot \frac{\sigma_k^2 \pi_k^2 (1 - u_2^2)}{(\pi_{k1} \pi_{k2})^{3/2}}$$

(대략적 형태 — 구체적은 Richardson-Green 1997 Appendix).

**Acceptance**:
$$\alpha = \min\left(1, \frac{\text{posterior}(x') \cdot q_\text{merge}(v)}{\text{posterior}(x) \cdot q_\text{split}(u)} \cdot |\mathcal{J}|\right)$$

**실용 tip**: Split이 "reasonable two-component fit"을 만들도록 $u$의 distribution 설계. 너무 extreme split은 rejection rate 높음.

</details>

**문제 3** (AI 연결): RJMCMC과 Dirichlet Process Mixture (DPM)의 차이 및 trade-off를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**RJMCMC for Mixture**:
- Parametric prior on $K$ (e.g., Poisson)
- Moves: birth, death, split, merge
- Explicit $K$ in state
- Flexible dimension changes

**Dirichlet Process Mixture (DPM)**:
- Non-parametric prior: concentration $\alpha$, base measure $G_0$
- Gibbs sampling (Neal 2000): data points dynamically assigned to existing or new cluster
- $K$ effectively "infinite" (countably many potential clusters, finite used)
- No explicit birth/death — just reassignment

**Differences**:

| | RJMCMC | DPM |
|--|--------|-----|
| Prior on $K$ | Parametric | Non-parametric (implicit) |
| Moves | Designed (birth, split) | Cluster reassignment |
| Proposal design | Complex | Automatic |
| Ergodicity | Design-dependent | Easier |
| Implementation | Complex | Relatively clean |
| Bayes factor | Direct estimate | Requires extra work |

**When RJMCMC preferred**:
- Informative prior on $K$ needed
- Model comparison (Bayes factors)
- Non-exchangeable parameters

**When DPM preferred**:
- Exchangeable data
- Automatic clustering
- Cleaner implementation

**Modern**:
- **Collapsed Gibbs for DPM** (Neal 2000): most popular
- **Stick-breaking representation**: explicit parameter for cluster weights → allows variational inference
- **HDP (Hierarchical DP)**: multi-level clustering

**VAE-era alternatives**:
- **VampPrior** (Tomczak-Welling 2018): learned discrete mixture prior
- **Neural autoregressive density estimation**: flexible density without explicit clustering

**결론**: RJMCMC는 **parametric flexibility**, DPM은 **non-parametric scalability**. 현대 Bayesian ML에서는 DPM + collapsed Gibbs 또는 neural density models가 dominant. 하지만 RJMCMC의 mathematical framework (transdimensional MCMC)는 여전히 중요한 theoretical tool.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. Particle Filter와 Sequential Monte Carlo](./05-particle-filter.md) | [📚 README](../README.md) | [Ch7-01 Maximum Likelihood for Graphical Models ▶](../ch7-learning-modern/01-mle-for-graphical-models.md) |

</div>
