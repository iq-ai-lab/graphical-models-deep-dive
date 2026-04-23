# 04. Gibbs Sampling on Graphical Models

## 🎯 핵심 질문

- **Markov blanket**이 Gibbs sampling에서 왜 local computation을 가능하게 하는가?
- MRF/BN에서 각각 Markov blanket은 어떻게 정의되고 무엇이 포함되는가?
- Gibbs sampling이 정확한 posterior로 수렴함을 어떻게 보이는가 (detailed balance)?
- Block Gibbs, collapsed Gibbs 같은 variant는 언제 필요한가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Gibbs Sampling**은 MCMC의 가장 popular form, 특히 PGM에서. **LDA** (topic model)의 collapsed Gibbs sampler는 가장 효율적 inference, **Ising model simulation**, **Boltzmann machine training** (contrastive divergence), **image denoising**, **Bayesian mixture model** 모두 Gibbs 기반. Modern **score-based diffusion models**의 reverse sampling도 Langevin-like (Gibbs의 연속 extension). **Variational inference와의 trade-off** — VI는 fast but biased, Gibbs는 slow but asymptotically exact. Gibbs를 이해하면 **sampling-based approximate inference**의 장단점이 명확해진다.

---

## 📐 수학적 선행 조건

- [Ch1-04 Markov Random Field](../ch1-conditional-independence/04-markov-random-field.md): MRF conditional independence
- [Ch1-05 Moralization](../ch1-conditional-independence/05-moralization.md): BN Markov blanket
- Markov chain basics: stationary distribution, detailed balance
- Conditional distribution

---

## 📖 직관적 이해

### Gibbs Sampling의 기본 아이디어

Joint $p(x_1, \ldots, x_n)$ 샘플링 어려움. Conditional $p(x_i | x_{-i})$ 샘플링 쉬움 (저차원).

**Gibbs algorithm**:
1. Initialize $x^{(0)}$
2. For $t = 1, 2, \ldots$:
   - For $i = 1, \ldots, n$ (in order or random):
     - $x_i^{(t)} \sim p(x_i | x_{-i}^{(t)})$

결과: $\{x^{(t)}\}$가 $p(x)$로 수렴.

### Markov Blanket의 역할

$p(x_i | x_{-i})$ 계산 시, **모든 다른 변수**가 필요한가?

**Markov blanket 정리**: $p(x_i | x_{-i}) = p(x_i | x_{\text{MB}(i)})$

- **MRF**: $\text{MB}(i) = N(i)$ (이웃)
- **BN**: $\text{MB}(i) = \text{pa}(i) \cup \text{ch}(i) \cup \text{co-pa}(i)$

**Local computation**: Conditional sampling은 $x_i$의 Markov blanket만 필요 → **efficient** for sparse graphs.

### MRF Gibbs (Ising Example)

Ising model:
$$p(x) \propto \exp\left(\sum_{(i,j) \in E} J_{ij} x_i x_j\right), x_i \in \{-1, +1\}$$

$p(x_i | x_{-i}) = p(x_i | x_{N(i)})$:
$$p(x_i = 1 | x_{N(i)}) = \sigma\left(2 \sum_{j \in N(i)} J_{ij} x_j\right)$$

Sigmoid of weighted sum of neighbors. **매우 빠름**.

### BN Gibbs (Student Network)

$p(G | D, I, S, L) = p(G | D, I, L)$ (co-parents of L through G).

Full Markov blanket of $G$: parents $\{D, I\}$ + children $\{L\}$ + co-parents $\{}$ (L has no other parent).

$p(G | D, I, L) \propto p(G | D, I) \cdot p(L | G)$ (local factors).

### Collapsed Gibbs

**Idea**: 일부 variable을 **analytically integrate out**, 나머지를 Gibbs.

**LDA example**:
- Variables: $\theta$ (document-topic), $\phi$ (topic-word), $z$ (topic assignments)
- **Collapsed**: integrate out $\theta, \phi$ (Dirichlet conjugate → closed form)
- Sample only $z$ via Gibbs
- Mixing faster, more accurate

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Gibbs Sampling Algorithm

Target: $p(x_1, \ldots, x_n)$.

**Iteration**:
```
x^{(0)} = initial
for t = 1, 2, ...:
    for i = 1 to n:
        x_i^{(t)} ~ p(x_i | x_1^{(t)}, ..., x_{i-1}^{(t)}, x_{i+1}^{(t-1)}, ..., x_n^{(t-1)})
```

**Random scan**: random $i$ at each step.  
**Systematic scan**: $i = 1, 2, \ldots, n$ in order.

### 정의 4.2 — Markov Blanket

**MRF**: $\text{MB}(v) := N(v)$ (neighbors in undirected graph).

**BN**: $\text{MB}(v) := \text{pa}(v) \cup \text{ch}(v) \cup \{\text{co-parents of v's children}\}$.

**Property**: $X_v \perp\!\!\!\perp X_{V \setminus \text{MB}(v) \cup \{v\}} | X_{\text{MB}(v)}$.

### 정의 4.3 — Block Gibbs

변수를 block $B_1, \ldots, B_K$로 나누고 block 단위 샘플링:
$$X_{B_k}^{(t)} \sim p(X_{B_k} | X_{\text{rest}}^{(t)})$$

**이점**: Block 내 correlated variables를 동시 샘플 → 더 빠른 mixing.

### 정의 4.4 — Collapsed Gibbs

일부 변수 $Z$ 분석적으로 integrate out:
$$p(X, Y) = \int p(X, Y, Z) dZ \quad \text{(closed form via conjugacy)}$$

$X, Y$에 대해서만 Gibbs sample. 종종 **더 효율**(mixing, variance).

---

## 🔬 정리와 증명

### 정리 4.1 — Markov Blanket (MRF)

**명제**: MRF에서 $p(x_v | x_{-v}) = p(x_v | x_{N(v)})$.

**증명**: 

MRF 인수분해: $p(x) \propto \prod_C \phi_C(x_C)$.

$p(x_v | x_{-v}) \propto p(x) \propto \prod_C \phi_C(x_C)$.

$C$가 $v$를 포함하지 않으면 $\phi_C$는 $x_v$ 무관 → constant in $x_v$, cancel out after normalization.

$v$를 포함하는 clique은 $v$와 $v$의 이웃만 포함 (clique 정의).

$$p(x_v | x_{-v}) \propto \prod_{C \ni v} \phi_C(x_C)$$

이는 $x_v$와 $N(v)$만 의존 → $p(x_v | x_{N(v)})$. $\square$

### 정리 4.2 — Markov Blanket (BN)

**명제**: BN에서 $\text{MB}(v) = \text{pa}(v) \cup \text{ch}(v) \cup \text{co-pa}(v)$.

**증명**:

BN factorization: $p(x) = \prod_w p(x_w | x_{\text{pa}(w)})$.

$p(x_v | x_{-v})$를 계산하기 위해, $x_v$ 포함하는 factor:
- $p(x_v | x_{\text{pa}(v)})$: $v$ 와 parents
- $p(x_w | x_{\text{pa}(w)})$ for $w \in \text{ch}(v)$: $v$는 parent, 다른 parents가 co-parents

$$p(x_v | x_{-v}) \propto p(x_v | x_{\text{pa}(v)}) \prod_{w \in \text{ch}(v)} p(x_w | x_{\text{pa}(w)})$$

이는 $v$, parents, children, co-parents만 의존. $\square$

**왜 co-parents 필요**: Child $w$의 CPT $p(x_w | x_{\text{pa}(w)})$에 $v$와 co-parents 모두 등장. $v$의 conditional을 계산할 때 co-parents value가 필요.

### 정리 4.3 — Gibbs의 Stationary Distribution = $p$

**명제**: Gibbs sampling의 Markov chain의 stationary distribution = target $p(x)$.

**증명**:

Detailed balance 확인:
$$p(x) P(x \to x') = p(x') P(x' \to x)$$

Gibbs에서 $x \to x'$ transition은 한 변수 $x_i$만 변경: $x_i \to x_i'$, $x_{-i} = x'_{-i}$.

$$P(x \to x') = p(x'_i | x_{-i})$$

$$p(x) P(x \to x') = p(x_i, x_{-i}) p(x'_i | x_{-i}) = p(x_{-i}) p(x_i | x_{-i}) p(x'_i | x_{-i})$$

$$p(x') P(x' \to x) = p(x'_i, x_{-i}) p(x_i | x_{-i}) = p(x_{-i}) p(x'_i | x_{-i}) p(x_i | x_{-i})$$

$= p(x) P(x \to x')$. Detailed balance holds for single-variable update. 

Composing updates: systematic scan이 각 변수를 한 번씩 → stationary preserved. $\square$

**Ergodicity**: Chain이 ergodic이면 $x^{(t)} \to p$ 수렴. Reducibility: 모든 state pair 간 positive probability path 필요 (지속적으로 $x_i$ update 가능성).

### 정리 4.4 — Mixing Time

**Mixing time** $\tau_{\text{mix}}$: total variation distance $\leq \epsilon$에 도달하는 시간.

**Spectral gap**: $\gamma = 1 - \lambda_2$ where $\lambda_2$ = second largest eigenvalue.

$$\tau_{\text{mix}} \leq \frac{1}{\gamma} \log\left(\frac{1}{\epsilon}\right)$$

**Bad scaling**: $\gamma \to 0$ for strongly correlated variables (Ising at critical temperature, long chain). **Slow mixing**.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# Ising model Gibbs sampling
def gibbs_ising(L, J, h=0.0, n_sweeps=2000, burn_in=500, seed=0):
    """L x L Ising model Gibbs sampling."""
    rng = np.random.default_rng(seed)
    x = rng.choice([-1, 1], size=(L, L))
    
    samples = []
    magnetization = []
    
    for sweep in range(n_sweeps):
        # Systematic scan
        for i in range(L):
            for j in range(L):
                # Markov blanket: 4 neighbors (periodic)
                neighbors_sum = (x[(i-1) % L, j] + x[(i+1) % L, j] + 
                                 x[i, (j-1) % L] + x[i, (j+1) % L])
                # p(x_ij = 1 | blanket) = sigmoid(2 J neighbors_sum + 2 h)
                local_field = J * neighbors_sum + h
                p_plus = 1 / (1 + np.exp(-2 * local_field))
                x[i, j] = 1 if rng.random() < p_plus else -1
        
        if sweep >= burn_in:
            samples.append(x.copy())
            magnetization.append(x.mean())
    
    return np.array(samples), magnetization

# 여러 temperature에서 실험
L = 8
n_sweeps = 2000
burn_in = 500

temperatures = [0.5, 1.0, 1.5, 2.27, 3.0]  # 2.27 is critical
fig, axes = plt.subplots(1, len(temperatures), figsize=(20, 4))

for ax, T in zip(axes, temperatures):
    J = 1.0 / T
    samples, mag = gibbs_ising(L, J, n_sweeps=n_sweeps, burn_in=burn_in)
    
    avg_mag = np.abs(np.mean(mag))
    
    ax.imshow(samples[-1], cmap='gray', vmin=-1, vmax=1)
    ax.set_title(f'T={T}, |M|={avg_mag:.2f}')
    ax.axis('off')

plt.suptitle('Ising model Gibbs samples (final configuration)')
plt.tight_layout()
plt.savefig('ising_gibbs.png', dpi=120, bbox_inches='tight')
plt.show()

# Mixing time demonstration at critical
print("Mixing behavior near critical temperature (T=2.27):")
J_critical = 1 / 2.27
samples, mag = gibbs_ising(L, J_critical, n_sweeps=3000, burn_in=0)
plt.figure(figsize=(10, 4))
plt.plot(mag[:2000], linewidth=0.5)
plt.axvline(500, color='r', linestyle='--', label='Burn-in end')
plt.xlabel('Sweep')
plt.ylabel('Magnetization')
plt.title(f'Ising magnetization trace at T={2.27} (critical)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gibbs_mixing.png', dpi=120, bbox_inches='tight')
plt.show()

print("\n관찰: T=2.27 근처에서 magnetization이 slow oscillation → mixing time 증가")
print("T << Tc (저온): 빠른 수렴하지만 symmetric broken state에 trap → double peak")
print("T >> Tc (고온): symmetric, fast mixing")

# Gibbs의 correctness 검증: 작은 grid에서 brute force와 비교
L_small = 3
J = 0.5
samples_small, _ = gibbs_ising(L_small, J, n_sweeps=50000, burn_in=5000)

# Brute force: enumerate all 2^9 states
from itertools import product
total_mag = 0
Z = 0
for config in product([-1, 1], repeat=L_small**2):
    state = np.array(config).reshape(L_small, L_small)
    energy = 0
    for i in range(L_small):
        for j in range(L_small):
            energy += J * state[i, j] * state[(i+1) % L_small, j]
            energy += J * state[i, j] * state[i, (j+1) % L_small]
    p = np.exp(energy)
    Z += p
    total_mag += p * state.mean()

exact_mag = total_mag / Z
gibbs_mag_small = samples_small.mean()

print(f"\n3x3 Ising J=0.5:")
print(f"Gibbs estimate of <M>: {gibbs_mag_small:.4f}")
print(f"Exact <M>: {exact_mag:.4f}")
print(f"Error: {abs(gibbs_mag_small - exact_mag):.4f}")
```

**출력 예시**:
```
Ising model Gibbs samples (final configuration) 
  [visual of 5 different T configurations]

관찰: T=2.27 근처에서 magnetization이 slow oscillation → mixing time 증가
T << Tc (저온): 빠른 수렴하지만 symmetric broken state에 trap → double peak
T >> Tc (고온): symmetric, fast mixing

3x3 Ising J=0.5:
Gibbs estimate of <M>: 0.0012
Exact <M>: 0.0000
Error: 0.0012
```

Gibbs가 정확한 marginal에 수렴하지만, critical temperature 근처에서 mixing이 느려짐.

---

## 🔗 AI/ML 연결

### LDA Collapsed Gibbs (Griffiths-Steyvers 2004)

LDA variables: $\theta_d, \phi_k, z_{d, n}$.

**Collapsed**: Integrate out $\theta, \phi$ (Dirichlet conjugate):
$$p(z_{d, n} = k | z_{-(d, n)}, w) \propto \frac{n_{d, k}^{-} + \alpha}{\sum_{k'} (n_{d, k'}^{-} + \alpha)} \cdot \frac{n_{k, w_{d, n}}^{-} + \eta}{\sum_{w'} (n_{k, w'}^{-} + \eta)}$$

여기서 $n_{d, k}^{-}$ = document $d$에서 topic $k$의 count (current assignment 제외).

**매우 빠름**: $p(z_{d, n})$ 계산이 count 기반, $O(K)$ per token. Ch7-04에서 자세히.

### Boltzmann Machine / RBM Training

**Contrastive Divergence** (Hinton 2002):
- Gibbs chain을 **짧게** (k=1 step)
- Biased but practical gradient estimator

**Persistent Contrastive Divergence** (Tieleman 2008):
- Gibbs chain을 **유지** across mini-batches
- Less biased, slower mixing

**Boltzmann training**:
- Gradient = data mean - model mean (of statistics)
- Model mean ≈ Gibbs sample average
- Deep Boltzmann Machines, RBM layer pretraining

### Image Denoising

Noisy image $Y$ → clean $X$ via MRF:
$$p(X | Y) \propto p(Y | X) p(X)$$
$$= \prod_i p(y_i | x_i) \prod_{(i,j) \in E} \phi_{ij}(x_i, x_j)$$

**Gibbs denoising**:
- Initialize $X = Y$
- Sample $x_i | $ neighbors + observation
- Run for many iterations, take average

Classical before deep learning took over.

### Bayesian Mixture Model Clustering

Finite mixture: $p(x | \theta) = \sum_k \pi_k p(x | \theta_k)$.

**Gibbs sampler for mixture**:
- Alternately sample: cluster assignment, cluster parameters
- Dirichlet process mixture (infinite K): **Chinese Restaurant Process** sampling

Widely used in Bayesian nonparametrics before VAE era.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Conditional tractable | $p(x_i \| x_{-i})$ sampling 어려우면 Metropolis-Hastings |
| Sufficient mixing | Critical phenomena에서 exponentially slow |
| Ergodicity | Reducible chains (multiple modes)에서 trap |
| Discrete or simple continuous | 복잡한 continuous는 HMC 필요 |

**주의**: Mixing time이 **spectral gap**에 의해 결정 — strongly correlated / critical systems에서 exponentially large. **Parallel tempering**, **swendsen-wang**, **auxiliary variable methods** 등으로 improvement.

---

## 📌 핵심 정리

$$\boxed{\text{MRF: } \text{MB}(v) = N(v); \quad \text{BN: } \text{MB}(v) = \text{pa} \cup \text{ch} \cup \text{co-pa}}$$

$$\boxed{x_i^{(t)} \sim p(x_i | x_{\text{MB}(i)}^{(t)})}$$

| 개념 | 의미 |
|------|------|
| **Markov blanket** | Local conditional 정의에 필요한 변수 |
| **Detailed balance** | Stationary distribution = target |
| **Ergodicity** | Convergence to target |
| **Mixing time** | Spectral gap에 의해 결정 |
| **Collapsed Gibbs** | 일부 변수 integrate out → faster mixing |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2-cycle BN $A \to B, B \to C, A \to C$에서 $A$의 Markov blanket을 구하라.

<details>
<summary>힌트 및 해설</summary>

**BN structure**: $A$ → $B$, $B$ → $C$, $A$ → $C$. 즉 $A$ has 2 children: $B$ and $C$. $B$ has child $C$. (No cycles; this is DAG.)

**Markov blanket of $A$**:
- Parents of $A$: $\emptyset$
- Children of $A$: $\{B, C\}$
- Co-parents of $A$'s children:
  - Co-parents with child $B$: $\emptyset$ ($B$ has only $A$ as parent)
  - Co-parents with child $C$: parents of $C$ minus $A$ = $\{B\}$

$\text{MB}(A) = \{B, C\} \cup \{B\} = \{B, C\}$.

**Conditional**: $p(A | B, C) = p(A | \text{MB}(A)) \propto p(A) \cdot p(B | A) \cdot p(C | A, B)$.

(Note $B$는 child, $C$의 co-parent; $B$가 이미 children에 포함되어 있음.)

**Gibbs update**: Sample $A$ from $p(A | B, C)$ — tractable (few terms, closed form or enumeration).

</details>

**문제 2** (심화): Critical temperature의 Ising model에서 Gibbs의 **mixing time이 exponentially large**인 이유를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Phase transition at $T_c$**: Ising model은 $T < T_c$에서 spontaneous symmetry breaking — positive or negative magnetization, two equilibrium states.

**At $T_c$**: Critical slowing down — 상태들 간 transition이 느려짐. **Correlation length diverges** — $\xi \to \infty$.

**Gibbs의 문제**:
1. Local update만 (single spin flip)
2. 상태 간 transition이 **all spins를 flip**해야 함 → $O(L^2)$ correlated flips
3. $T_c$ 근처에서 이 flips가 coherent하게 일어날 확률 → exponentially small in $L$

**수학적**: Cheeger inequality
$$\tau_{\text{mix}} \leq \frac{\log(1/\min p)}{\lambda_2}$$
where $\lambda_2$ = spectral gap. $T_c$에서 $\lambda_2 \sim L^{-z}$ with $z \approx 2$ (dynamic critical exponent). $L$ increase → mixing $O(L^2)$.

**Sharp limit**: **Below $T_c$** with broken symmetry, mixing is **exponential** in $L$ — 한 phase에서 다른 phase로의 transition이 free energy barrier 통과 필요.

**해결**:
1. **Swendsen-Wang**: cluster flip — O(1) mixing time at $T_c$!
2. **Wolff algorithm**: single-cluster version, efficient
3. **Parallel tempering**: multiple temperature chains, swap
4. **Cluster-based MCMC**: 강하게 correlated variables를 동시 update

**ML 관점**: Boltzmann machine training도 비슷한 문제. Deep networks의 complex energy landscape가 Gibbs mixing을 어렵게.

</details>

**문제 3** (AI 연결): RBM (Restricted Boltzmann Machine) training에서 **Contrastive Divergence k=1** (1 Gibbs step)이 충분히 작동하는 이유는?

<details>
<summary>힌트 및 해설</summary>

**RBM structure**: Visible $v$, hidden $h$, energy $E(v, h) = -v^T W h - b^T v - c^T h$. **Bipartite** → conditional $p(h | v)$ and $p(v | h)$ both factorize:
- $p(h_j = 1 | v) = \sigma(\sum_i W_{ij} v_i + c_j)$
- $p(v_i = 1 | h) = \sigma(\sum_j W_{ij} h_j + b_i)$

**Gradient of log-likelihood**:
$$\nabla_\theta \log p(v) = \mathbb{E}_{p(h | v)}[\nabla_\theta E] - \mathbb{E}_{p(v, h)}[\nabla_\theta E]$$

- First term (positive phase): **tractable** (bipartite conditional)
- Second term (negative phase): **intractable** (full joint sampling)

**Naive Gibbs**: Run long Gibbs chain for each gradient step → too expensive.

**CD-k (Hinton 2002)**:
- Start Gibbs at data $v^{(0)} = v_{\text{data}}$
- Run $k$ Gibbs steps → $(v^{(k)}, h^{(k)})$
- Approximate negative phase with this sample

**CD-1 works because**:

1. **Biased but consistent direction**: CD의 gradient가 true gradient와 같은 **general direction** (Hinton 2002 showed). Bias는 있지만 파라미터 update가 올바른 방향.

2. **Regularization effect**: Bias가 implicit regularization 역할. Overfit 방지.

3. **Starting at data**: $v^{(0)} = v_{\text{data}}$가 already reasonable sample of $p(v)$ (대부분 training data가 model에 맞게 학습됨 → data distribution ≈ model distribution at convergence).

4. **Small $k$ = low variance**: $k$ 클수록 variance $\uparrow$, $k$ 작을수록 bias $\uparrow$. Trade-off.

5. **Empirical success**: Hinton et al.의 experiments에서 CD-1, CD-10 모두 좋은 feature 학습.

**한계**:
- CD는 **not convergent to MLE** — biased
- Persistent CD (PCD): data-free Gibbs chain 유지 → asymptotically less biased
- Score matching, NCE 등 alternative objective

**Modern**: Deep Boltzmann Machines의 hierarchical hidden layers에서는 CD insufficient → layer-wise pretraining, moment matching 등.

**결론**: CD-1은 "**quick and dirty but empirically working**" — 이론적 이해보다 practical utility가 앞섬. 이론적 justification은 ongoing research (e.g., Carreira-Perpinan-Hinton 2005).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Expectation Propagation (EP)](./03-expectation-propagation.md) | [📚 README](../README.md) | [05. Particle Filter와 Sequential Monte Carlo ▶](./05-particle-filter.md) |

</div>
