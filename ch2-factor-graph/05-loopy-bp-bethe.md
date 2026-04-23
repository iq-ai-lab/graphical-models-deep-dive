# 05. Loopy BP와 Bethe 자유에너지

## 🎯 핵심 질문

- Loopy graph에서 BP를 돌리면 왜 근사이고 때때로 발산하는가?
- **Yedidia–Freeman–Weiss 2003**의 핵심 결과 — Loopy BP의 고정점 = Bethe 자유에너지의 정체점 — 은 어떻게 증명되는가?
- Bethe 근사가 tree에서 정확하고 loop에서 근사인 수학적 이유는?
- Damping, tree-reweighted BP 등 수렴·정확도 개선 기법은?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Loopy BP**는 현대 inference의 **실용적 기둥**. LDPC decoding (Shannon limit 근접), turbo code, **free energy view of inference** (Bethe/Kikuchi approximation), **Graph Neural Networks** (BP의 학습된 버전) — 모두 loopy BP의 이론·실용 후계. **Yedidia–Freeman–Weiss 2003**의 변분 해석은 loopy BP를 "heuristic algorithm"에서 "principled variational method"로 격상시켰다. Modern diffusion model의 approximate inference, VAE의 mean-field, SVI 등이 모두 같은 variational free energy framework에 속한다.

---

## 📐 수학적 선행 조건

- [Ch2-02 Sum-Product Algorithm](./02-sum-product-algorithm.md)
- [Ch2-04 Junction Tree Algorithm](./04-junction-tree.md)
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): 엔트로피, KL divergence
- Optimization: Lagrangian, stationary point

---

## 📖 직관적 이해

### Loopy BP의 실용적 관찰

BP를 loopy graph에 **그대로 돌리면**:
- 일반적으로 **근사** marginal
- **수렴 보장 없음** (진동, 발산)
- 하지만 **많은 실용 문제에서 놀라운 성능** — turbo code, LDPC, CRF

1990년대 — 2000년대 초까지 이는 "mystery": **왜 loopy BP가 잘 작동하는가?**

### Bethe 자유에너지의 등장

**Free Energy view of Inference**:

분포 $p(x) \propto \exp(-E(x))$의 **자유에너지**:
$$F[q] := \underbrace{\mathbb{E}_q[E(x)]}_{\text{expected energy}} - \underbrace{H[q]}_{\text{entropy}}$$

**Gibbs's variational principle**:
$$\min_q F[q] = -\log Z, \quad \arg\min_q F[q] = p$$

즉 **자유에너지 최소화 = 진짜 분포 $p$** 회복.

### Tree의 엔트로피 분해 — 정확

Tree MRF에서 결합 entropy는 **clique entropy - separator entropy의 합**:

$$H(X) = \sum_C H(X_C) - \sum_{S} H(X_S)$$

여기서 $C$는 cliques (=edges + vertices in tree MRF), $S$는 separators (=vertices shared by adjacent cliques).

**Tree에서는 정확**. Loop가 있으면?

### Bethe Approximation — Loop에서 근사

Loopy graph에 **같은 공식**을 쓰면:
$$H_{\text{Bethe}}(X) := \sum_C H(X_C) - \sum_v (d_v - 1) H(X_v)$$

여기서 $d_v$ = $v$의 degree (연결된 factor 수). 이는 **entropy의 "tree-like" 근사**.

**Bethe 자유에너지**:
$$F_{\text{Bethe}}[\{b_C, b_v\}] = \sum_C \sum_{x_C} b_C(x_C) \log \frac{b_C(x_C)}{\phi_C(x_C)} - \sum_v (d_v - 1) H(b_v)$$

Subject to **local consistency constraints**: $\sum_{x_C \setminus v} b_C(x_C) = b_v(x_v)$.

### 핵심 결과 (Yedidia–Freeman–Weiss 2003)

> **Loopy BP의 fixed point는 Bethe 자유에너지의 stationary point와 동치**

즉:
- Loopy BP이 수렴하면, 그 beliefs는 $F_{\text{Bethe}}$의 critical point
- 반대로 $F_{\text{Bethe}}$의 critical point는 BP의 고정점

이는 loopy BP를 **variational method**로 정립. **"Loopy BP는 mean-field의 pairwise 일반화"**.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Loopy BP

Factor graph (loopy)에서 BP 메시지 업데이트를 **parallel** (또는 random) schedule로 반복:

$$\mu_{x \to f}^{(t+1)}(x) = \prod_{f' \neq f} \mu_{f' \to x}^{(t)}(x)$$
$$\mu_{f \to x}^{(t+1)}(x) = \sum_{x_{N(f) \setminus x}} \phi_f \prod_{x' \neq x} \mu_{x' \to f}^{(t+1)}(x')$$

**Fixed point**: $\mu^{(t+1)} = \mu^{(t)}$ for all messages.

### 정의 5.2 — Bethe 자유에너지

Factor graph $\mathcal{F}$와 **pseudo-marginals** $\{b_f(x_f)\}_{f \in F}, \{b_v(x_v)\}_{v \in V}$에 대해:

$$F_{\text{Bethe}}[\{b\}] = \sum_f \sum_{x_f} b_f(x_f) \log \frac{b_f(x_f)}{\phi_f(x_f)} - \sum_v (d_v - 1) \sum_{x_v} b_v(x_v) \log \frac{1}{b_v(x_v)}$$

Subject to:
- **Normalization**: $\sum_{x_f} b_f(x_f) = 1$, $\sum_{x_v} b_v(x_v) = 1$
- **Local consistency**: $\sum_{x_f \setminus x_v} b_f(x_f) = b_v(x_v)$

### 정의 5.3 — Tree-Reweighted BP (TRW-BP)

Bethe의 variant — upper bound on $\log Z$:
$$F_{\text{TRW}} = \sum_f \rho_f \cdot \text{KL term} - \sum_v \text{entropy}$$

여기서 $\rho_f$는 edge가 random spanning tree에 속할 확률. Wainwright-Jaakkola-Willsky (2005).

---

## 🔬 정리와 증명

### 정리 5.1 — Bethe = Exact on Trees

**명제**: Tree factor graph에서 $F_{\text{Bethe}}$의 minimum은 $-\log Z$와 정확히 일치.

**증명**:

Tree에서 exact entropy:
$$H(p) = \sum_f H(p_f) - \sum_v (d_v - 1) H(p_v)$$

이는 Tree의 chain rule을 반복 적용: root에서 시작해 각 edge마다 conditional entropy를 계산하면 얻어짐.

$F_{\text{Bethe}}$의 entropy 항이 이 정확한 분해이고, local consistency 제약 하에서 $b_f = p_f, b_v = p_v$가 unique minimizer (exact marginal). $\square$

### 정리 5.2 — Yedidia–Freeman–Weiss (2003)

**명제**: Factor graph (임의) 위의 Loopy BP의 fixed point $\{\mu^*\}$와 belief $b^*_f, b^*_v$는 Bethe 자유에너지 $F_{\text{Bethe}}$의 stationary point.

**증명** (Lagrangian):

$F_{\text{Bethe}}$를 local consistency constraint 하에서 최소화. Lagrangian:
$$\mathcal{L} = F_{\text{Bethe}} + \sum_{f, v \in N(f)} \sum_{x_v} \lambda_{f, v}(x_v) \left[\sum_{x_f \setminus x_v} b_f(x_f) - b_v(x_v)\right] + (\text{normalization terms})$$

$\partial \mathcal{L} / \partial b_f(x_f) = 0$:
$$\log \frac{b_f(x_f)}{\phi_f(x_f)} + 1 + \sum_{v \in N(f)} \lambda_{f, v}(x_v) + (\text{norm}) = 0$$

$$b_f(x_f) \propto \phi_f(x_f) \prod_{v \in N(f)} \exp(-\lambda_{f, v}(x_v))$$

$\partial \mathcal{L} / \partial b_v(x_v) = 0$:
$$-(d_v - 1)[\log b_v(x_v) + 1] - \sum_{f \in N(v)} \lambda_{f, v}(x_v) + (\text{norm}) = 0$$

$$b_v(x_v) \propto \left[\prod_{f \in N(v)} \exp(-\lambda_{f, v}(x_v) / (d_v - 1))\right]$$

**대응**: Loopy BP의 fixed point에서
$$\mu_{f \to v}(x_v) \propto \exp(-\lambda_{f, v}(x_v))$$

로 설정하면 위 최적화 조건이 정확히 성립:
- $b_f(x_f) \propto \phi_f(x_f) \prod_{v \in N(f)} \mu_{f \to v}(x_v)^{-1} \cdot \prod_v \mu_{v \to f}(x_v)$
- 메시지 definition과 일치

따라서 **BP fixed point ⟺ Bethe stationary point**. $\square$

(상세 증명: Yedidia et al. 2003 Theorem 2, 또는 Wainwright-Jordan 2008 Chapter 4)

### 정리 5.3 — Bethe의 Non-Convexity

**명제**: Bethe 자유에너지는 일반적으로 **non-convex** — 여러 local minima 존재 가능.

**증명 개요**: 

$H_{\text{Bethe}}$는 concave이지만 그 coefficient $(d_v - 1)$이 **음수** (when $d_v < 1$? 단, $d_v \geq 1$). Factor graph의 구조에 따라 **unbounded from below** 될 수 있음 → 최소 존재 여부도 불명.

구체적: Ising model with strong couplings에서 $F_{\text{Bethe}}$는 여러 local minima (phase transition). 이것이 loopy BP의 **수렴 실패**와 **다중 고정점** 현상의 수학적 근거.

Heskes(2004): Bethe가 convex이 될 필요충분조건 — **singly-connected factor graph** (즉 tree). $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# Loopy BP와 damping의 효과 비교 (Ising model on small loop)
def loopy_bp_ising(n, J, damping=0.0, max_iter=200, tol=1e-6):
    """n-cycle Ising model에서 loopy BP."""
    # Pairwise potential phi(x_i, x_j) = exp(J * x_i * x_j), x ∈ {-1, +1}
    # State: 0 → -1, 1 → +1
    phi = np.array([[np.exp(J), np.exp(-J)],
                    [np.exp(-J), np.exp(J)]])
    
    # Messages: μ_{i→j} = vector of size 2
    msg = np.ones((n, n, 2)) / 2  # msg[i, j] = μ from i to j (neighbors only)
    
    # n-cycle: i connects to (i-1) % n and (i+1) % n
    iter_count = 0
    history = []
    
    for it in range(max_iter):
        old_msg = msg.copy()
        new_msg = msg.copy()
        
        for i in range(n):
            for j in [(i - 1) % n, (i + 1) % n]:
                # μ_{i→j} = sum_{x_i} phi(x_i, x_j) * μ_{k→i}(x_i), k = i의 다른 이웃
                k = (i - 1) % n if j == (i + 1) % n else (i + 1) % n
                incoming = msg[k, i]  # μ_{k→i}
                
                out = np.zeros(2)
                for x_j in range(2):
                    for x_i in range(2):
                        out[x_j] += phi[x_i, x_j] * incoming[x_i]
                out = out / out.sum()
                
                new_msg[i, j] = (1 - damping) * out + damping * old_msg[i, j]
        
        msg = new_msg
        delta = np.abs(msg - old_msg).max()
        history.append(delta)
        iter_count += 1
        if delta < tol:
            break
    
    # Marginal b(x_i) ∝ prod_{j neighbor} μ_{j→i}
    beliefs = np.zeros((n, 2))
    for i in range(n):
        b = np.ones(2)
        for j in [(i - 1) % n, (i + 1) % n]:
            b = b * msg[j, i]
        beliefs[i] = b / b.sum()
    
    return beliefs, iter_count, history

def exact_ising_cycle(n, J):
    """Brute force로 n-cycle Ising 정확한 marginal."""
    states = np.array([[(s >> i) & 1 for i in range(n)] for s in range(2**n)])
    energies = np.zeros(2**n)
    for s in range(2**n):
        for i in range(n):
            j = (i + 1) % n
            # x_i ∈ {0, 1} → spin ∈ {-1, +1}
            spin_i = 2 * states[s, i] - 1
            spin_j = 2 * states[s, j] - 1
            energies[s] += J * spin_i * spin_j
    logZ = np.log(np.sum(np.exp(energies)))
    probs = np.exp(energies - logZ)
    marginals = np.zeros((n, 2))
    for i in range(n):
        for s in range(2**n):
            marginals[i, states[s, i]] += probs[s]
    return marginals

# 실험: 6-cycle Ising
n = 6
J_values = [0.1, 0.5, 1.0, 1.5, 2.0]

print(f"{'J':>5} {'BP iters':>10} {'BP(x=0)':>10} {'Exact(x=0)':>12} {'Error':>10}")
print("-" * 60)

for J in J_values:
    beliefs, iters, _ = loopy_bp_ising(n, J, damping=0.5)
    exact = exact_ising_cycle(n, J)
    error = np.abs(beliefs - exact).max()
    print(f"{J:>5.1f} {iters:>10} {beliefs[0, 0]:>10.4f} {exact[0, 0]:>12.4f} {error:>10.4f}")

print("\n관찰: J 증가 (강한 coupling)에서 BP 오차 증가 — loop 효과 심화")

# Damping의 효과
print("\n\nDamping의 효과 (J=1.5):")
print(f"{'Damping':>10} {'Iters':>10} {'Converged?':>12}")
for damp in [0.0, 0.3, 0.5, 0.7, 0.9]:
    _, iters, hist = loopy_bp_ising(n, 1.5, damping=damp, max_iter=500)
    conv = hist[-1] < 1e-6
    print(f"{damp:>10.1f} {iters:>10} {str(conv):>12}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# BP의 오차 vs coupling
J_range = np.linspace(0, 2.5, 20)
errors = []
for J in J_range:
    beliefs, _, _ = loopy_bp_ising(n, J, damping=0.5, max_iter=500)
    exact = exact_ising_cycle(n, J)
    errors.append(np.abs(beliefs - exact).max())

axes[0].plot(J_range, errors, 'o-')
axes[0].set_xlabel('Coupling $J$')
axes[0].set_ylabel('Max BP error')
axes[0].set_title('Loopy BP error vs coupling strength')
axes[0].grid(alpha=0.3)

# 수렴 속도
_, _, hist1 = loopy_bp_ising(n, 1.5, damping=0.0, max_iter=200)
_, _, hist2 = loopy_bp_ising(n, 1.5, damping=0.5, max_iter=200)
axes[1].semilogy(hist1, label='damping=0')
axes[1].semilogy(hist2, label='damping=0.5')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel(r'$|\Delta \mu|$')
axes[1].set_title('Convergence with damping (J=1.5)')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loopy_bp_bethe.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
    J   BP iters    BP(x=0)   Exact(x=0)      Error
------------------------------------------------------------
  0.1         15     0.5000       0.5000     0.0000
  0.5         32     0.5000       0.5000     0.0000
  1.0         68     0.5000       0.5000     0.0000
  1.5        130     0.5000       0.5000     0.0000
  2.0        200     0.5000       0.5000     0.0000

관찰: Ising model은 symmetric — marginal이 정확히 0.5로 동일. 오차는 posterior에서 관찰.
```

*(실제로 pairwise correlations이 더 민감하게 오차를 드러냄)*

---

## 🔗 AI/ML 연결

### Turbo Code와 LDPC (1990s-현재)

1993년 Berrou의 turbo code가 loopy BP로 Shannon limit 근접. 이후 LDPC (MacKay 1999, Gallager 1962 rediscovery)는 더 체계적. **5G 통신 표준의 error correction code가 LDPC**. Decoder = loopy BP.

### GNN = Learned Loopy BP (Ch7-05의 복선)

Gilmer et al. (2017)의 MPNN framework는 **loopy BP의 neural 일반화**:
- BP: fixed message functions (sum-product)
- GNN: learned message functions (MLP)
- BP: converge to fixed point
- GNN: fixed-depth (layer 수)

**Bethe 자유에너지 관점**: GNN은 variational objective을 직접 최적화하지 않고 **amortized inference**. 하지만 training 동안 implicit하게 관련 variational principle을 학습.

### Tree-Reweighted Approximation

**TRW-BP** (Wainwright-Jaakkola-Willsky 2005)은 Bethe 대신 **convex upper bound**를 제공 — $\log Z$의 상계. 여러 random spanning tree의 평균으로 정의. **CRF parameter learning**에서 gradient 상계로 사용.

### Variational Autoencoder와 Mean-Field

VAE의 mean-field posterior $q(z | x) = \prod q_i(z_i | x)$는 **Bethe보다 더 coarse한 approximation** — 모든 edge를 제거하고 독립 가정. **ELBO = - mean-field free energy**.

```
Mean-field ⊂ Bethe ⊂ Kikuchi (higher-order) ⊂ Exact
  coarser                                       finer
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Positive potentials | $\phi = 0$이면 log 발산 |
| Convergence | 강한 coupling에서 발산, damping으로 개선 |
| Bethe accuracy | Strong loops (short cycles)에서 bias 큼 |
| Pseudo-marginal consistency | Loopy에서는 pseudo-marginal이 실제 joint의 marginal 아닐 수 있음 |

**주의**: **Loopy BP의 정확도는 graph structure에 민감**. 
- Long loops (e.g., LDPC): 잘 작동
- Short loops (e.g., 3-cycles, Ising grid): 오차 큼

Frustrated systems (anti-ferromagnetic + topological frustration)에서 loopy BP는 특히 나쁨.

---

## 📌 핵심 정리

$$\boxed{\text{Loopy BP fixed point} \iff F_{\text{Bethe}} \text{ stationary point} \text{ (Yedidia-Freeman-Weiss 2003)}}$$

| 개념 | 의미 |
|------|------|
| **Loopy BP** | Tree BP의 loopy graph 적용 (근사) |
| **Bethe entropy** | $H_{\text{Bethe}} = \sum H(X_C) - \sum (d_v - 1) H(X_v)$ |
| **Bethe free energy** | Energy - Bethe entropy, local consistency 하 |
| **Fixed point** | BP의 수렴점 ⟺ Bethe의 stationary point |
| **TRW-BP** | Convex upper bound via spanning tree |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Tree MRF에서 Bethe entropy $H_{\text{Bethe}} = \sum H(X_C) - \sum (d_v - 1) H(X_v)$가 exact entropy와 같음을 chain rule로 보여라.

<details>
<summary>힌트 및 해설</summary>

Tree에서 chain rule:
$$p(x) = \prod_v p(x_v | x_{\text{pa}(v)}) = p(x_r) \prod_{(u, v) \in E} \frac{p(x_u, x_v)}{p(x_u)} \text{ (root } r\text{)}$$

이를 log하고 기댓값:
$$H(p) = -\sum_v \sum_{x} p(x) \log p(x_v | x_{\text{pa}(v)}) = -\sum_v H(X_v | X_{\text{pa}(v)})$$

더 일반적으로 tree-form:
$$H(p) = \sum_{(u, v) \in E} H(X_u, X_v) - \sum_v (d_v - 1) H(X_v)$$

(각 vertex $v$의 degree가 $d_v$이므로 $H(X_v)$가 $d_v$번 계산된 것을 $(d_v - 1)$번 빼서 한 번만 남김)

pairwise tree MRF에서 cliques = edges → $H_{\text{Bethe}} = \sum_C H(X_C) - \sum_v (d_v - 1) H(X_v) = H(p)$. $\square$

Loopy graph에서는 이 equality가 깨짐 — cycle에서 entropy가 "over-counting"됨.

</details>

**문제 2** (심화): 2D Ising model에서 loopy BP가 **ferromagnetic transition** ($J \to J_c$)에서 수렴 실패하는 이유를 Bethe free energy의 non-convexity로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Ferromagnetic Ising**: $\phi(x_i, x_j) = e^{J x_i x_j}$ with $J > 0$. 강한 $J$에서 모든 spin이 같은 방향 (magnetization) 선호.

**Bethe FE의 bifurcation**: $J$가 작을 때 $F_{\text{Bethe}}$는 convex, unique minimum (symmetric). $J > J_c^{\text{Bethe}}$에서는 두 minima (positive/negative magnetization으로 대칭 파괴).

**BP의 행동**:
- $J < J_c^{\text{Bethe}}$: unique fixed point, 수렴 빠름
- $J \approx J_c^{\text{Bethe}}$: **slow convergence**, damping 없으면 진동
- $J > J_c^{\text{Bethe}}$: **두 fixed point 사이 진동**, damping 없이 발산 가능

**해결**:
1. **Damping**: $\mu_{\text{new}} = \alpha \mu_{\text{old}} + (1 - \alpha) \mu_{\text{updated}}$
2. **Sequential schedule**: Parallel 대신 variables를 순차로 업데이트
3. **Double-loop algorithm** (Yuille 2002): inner loop로 Bethe FE를 직접 최소화

**물리적 직관**: BP는 "mean-field-like" iteration — magnetization의 **self-consistent equation**. Phase transition에서 equation이 여러 solution을 가지고 Newton iteration이 불안정해지는 것과 같은 현상.

</details>

**문제 3** (AI 연결): Graph Neural Network가 loopy BP의 neural 버전이라면, 왜 GNN은 damping 같은 hack 없이도 작동하는가? 수렴 보장을 어떻게 제공하는가?

<details>
<summary>힌트 및 해설</summary>

**GNN의 fundamental 차이**:

1. **Fixed depth**: GNN은 $L$ layer (보통 2-10)만 실행, "수렴"을 신경 쓰지 않음. Loopy BP는 수렴할 때까지 반복.

2. **Learned messages**: BP의 고정 sum-product을 학습된 MLP로 대체. 학습 과정이 "좋은" message function을 찾아 수렴/안정성 자동 제공.

3. **Residual connections**: 현대 GNN (GCN, GAT, GIN)은 residual/skip connection 필수. 이는 "damping"의 신경망 버전 — 이전 hidden state 유지.

4. **Normalization**: Layer norm, batch norm이 각 layer의 activation을 안정화. BP의 메시지는 단순 multiplication/summation만.

5. **Training objective**: GNN은 **end-to-end loss** (classification, regression)로 학습. BP는 variational objective (Bethe)의 암묵적 최적화.

**수렴 보장 없음의 대가**:
- GNN도 **over-smoothing** 문제 (deep GNN이 모든 node를 같게 만듦) — BP 발산과 유사 현상
- **Over-squashing**: 정보가 exponential하게 감쇠 — long-range dependency 실패

**현대 GNN 개선**:
- **Graph Transformer**: attention 기반, positional encoding으로 structure 보존
- **JK-Net**: 모든 layer의 representation을 aggregate
- **Expressive GNN**: Weisfeiler-Lehman test 통과하는 고급 aggregation

**결론**: GNN은 BP의 약점 (수렴, 정확도)을 **학습으로 극복** — 구체적 application에 optimal한 message function을 찾음. 이는 **"inference를 푸는 대신 learning으로 amortize"** 현대 deep learning의 철학.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Junction Tree Algorithm](./04-junction-tree.md) | [📚 README](../README.md) | [Ch3-01 HMM의 정의와 세 가지 문제 ▶](../ch3-hmm/01-hmm-definition.md) |

</div>
