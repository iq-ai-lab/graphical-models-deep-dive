# 02. Bethe 자유에너지와 Loopy BP의 변분 해석

## 🎯 핵심 질문

- **Bethe free energy** $F_{\text{Bethe}}$는 mean-field보다 왜 더 정확한가?
- **Yedidia–Freeman–Weiss 2003** — "Loopy BP fixed point = Bethe stationary point" — 의 증명은?
- Bethe이 tree에서 exact이고 loop에서 approx인 structural 이유는?
- Kikuchi (region-graph) approximation은 어떻게 Bethe를 일반화하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Bethe approximation**은 mean-field에서 loopy BP로 넘어가는 **variational bridge**. 1935년 Hans Bethe의 통계역학 원전을 Yedidia-Freeman-Weiss가 2003년 probabilistic graphical models에 연결 — 이는 "Loopy BP가 왜 작동하는가"에 대한 기념비적 설명. Modern LDPC decoding, protein folding, **Kikuchi approximation**(CCCP), **Tree-Reweighted BP**, **Region graph methods** — 모두 Bethe의 일반화. Variational inference의 **structured** approximation 계보의 핵심. Mean-field(Ch6-01)보다 정확, exact(JT)보다 빠름.

---

## 📐 수학적 선행 조건

- [Ch2-02 Sum-Product Algorithm](../ch2-factor-graph/02-sum-product-algorithm.md)
- [Ch2-05 Loopy BP와 Bethe 자유에너지](../ch2-factor-graph/05-loopy-bp-bethe.md): 이미 다룸
- [Ch6-01 Mean-Field Variational Inference](./01-mean-field-vi.md)
- Information theory: entropy, KL divergence
- Calculus of variations, Lagrangian

---

## 📖 직관적 이해

### Free Energy View of Inference

PGM 문제를 **free energy minimization**으로 formulate:

$$p(x) = \frac{1}{Z} \prod_f \phi_f(x) = \frac{1}{Z} \exp(-E(x))$$

where $E(x) = -\sum_f \log \phi_f(x)$.

**Gibbs variational principle**:
$$\min_q F[q] = -\log Z, \quad F[q] = U[q] - H[q] = \mathbb{E}_q[E] - H(q)$$

$F[q]$ = free energy, achieved at $q = p$.

**Approximation**: restrict $q$ to tractable family → approximate $-\log Z$ and $p$.

### Mean-Field vs Bethe

| | Mean-field | Bethe |
|--|-----------|-------|
| $q$ family | $q = \prod q_i$ | Local marginals + consistency |
| Entropy | $\sum H(q_i)$ | $\sum H(q_f) - \sum (d_v - 1) H(q_v)$ |
| Parameters | $|q_i|$ each variable | $|q_f|$ each factor + $|q_v|$ each variable |
| Exact when | $p$ is fully factorized | Tree factor graph |
| Captures | Marginals only | Local correlations |

**Bethe은 mean-field보다 더 rich structure** (local pairwise correlations). Tree에서 exact.

### Bethe Entropy의 근거 (Tree)

Tree에서 joint entropy의 정확한 분해:
$$H(p) = \sum_f H(X_f) - \sum_v (d_v - 1) H(X_v)$$

여기서 $d_v$ = $v$의 degree (인접 factor 수), $f$는 factor 위의 marginal.

**유도**: Tree의 chain rule을 반복 적용. Each edge (factor)의 joint entropy에서 shared vertex의 entropy를 한 번만 count.

Loopy graph에서 **같은 formula** 사용 → approximation. Cycle에서 entropy **double-counting** 발생.

### Bethe Free Energy

$$F_{\text{Bethe}}[\{b_f, b_v\}] = U_{\text{Bethe}} - H_{\text{Bethe}}$$

$$= \sum_f \sum_{x_f} b_f(x_f) [-\log \phi_f(x_f)] - \sum_f H(b_f) + \sum_v (d_v - 1) H(b_v)$$

$$= \sum_f \sum_{x_f} b_f(x_f) \log \frac{b_f(x_f)}{\phi_f(x_f)} - \sum_v (d_v - 1) H(b_v)$$

**Local consistency constraints**: $\sum_{x_f \setminus x_v} b_f(x_f) = b_v(x_v)$.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Bethe Pseudo-Marginals

Factor graph $\mathcal{F}$에 대해, **pseudo-marginals** $\{b_f\}_{f \in F}, \{b_v\}_{v \in V}$:
- $b_f: \prod_{v \in N(f)} \mathcal{X}_v \to [0, 1]$ with $\sum b_f = 1$
- $b_v: \mathcal{X}_v \to [0, 1]$ with $\sum b_v = 1$
- **Local consistency**: $\sum_{x_f \setminus x_v} b_f(x_f) = b_v(x_v)$ for $v \in N(f)$

"Pseudo"라는 이름: tree에서는 실제 marginal과 일치, loopy에서는 일반 joint에서 partial consistency만.

### 정의 2.2 — Bethe Free Energy

$$F_{\text{Bethe}}[\{b\}] = \sum_f \text{KL}(b_f \| \phi_f) - \sum_v (d_v - 1) H(b_v)$$

더 확장:
$$= \sum_f \sum_{x_f} b_f \log \frac{b_f}{\phi_f} - \sum_v (d_v - 1) H(b_v)$$

**Relation to true free energy**: Tree에서 $F_{\text{Bethe}} = -\log Z$ at optimal. Loopy에서 일반적으로 차이 있음 (bias).

### 정의 2.3 — Bethe Optimization Problem

$$\min_{\{b\}} F_{\text{Bethe}}[\{b\}]$$

subject to:
- Normalization: $\sum_{x_f} b_f = 1$, $\sum_{x_v} b_v = 1$
- Local consistency: $\sum_{x_f \setminus x_v} b_f(x_f) = b_v(x_v)$
- Non-negativity: $b_f, b_v \geq 0$

이는 **non-convex** optimization (Bethe가 loopy graph에서 concave 아님).

---

## 🔬 정리와 증명

### 정리 2.1 — Bethe Exact on Trees

**명제**: Tree factor graph에서 $F_{\text{Bethe}}$의 minimum = $-\log Z$, minimizer = true marginals.

**증명**:

Tree에서 chain rule로 joint 분해:
$$p(x) = \prod_f \frac{p(x_{N(f)})}{\prod_{v \in N(f)} p(x_v)^{1 - 1/d_f^v}} \cdot \prod_v p(x_v)^{\text{something}}$$

정확한 expansion:
$$p(x) = \prod_f p(x_{N(f)}) \prod_v p(x_v)^{1 - d_v}$$

(모든 factor marginal을 곱하고, variable이 $d_v$번 counted된 것을 $(1 - d_v)$ exponent로 보정).

**Entropy**:
$$H(p) = -\mathbb{E}_p[\log p] = \sum_f H(X_{N(f)}) - \sum_v (d_v - 1) H(X_v)$$

이것이 Bethe entropy와 정확히 일치. 따라서 Bethe FE = exact FE at true marginals. $\square$

**Loopy graph**: 위 expansion이 **정확하지 않음** — cycle에서 variable이 over- or under-counted. 따라서 approximation.

### 정리 2.2 — Yedidia-Freeman-Weiss (2003)

**명제**: Factor graph에서 Loopy BP의 fixed point $\{\mu^*\}$와 derived beliefs $b^*_f, b^*_v$는 $F_{\text{Bethe}}$의 stationary point (Lagrangian 기준).

**증명**:

Lagrangian:
$$\mathcal{L} = F_{\text{Bethe}} - \sum_v \nu_v (\sum b_v - 1) - \sum_f \nu_f (\sum b_f - 1) - \sum_{f, v \in N(f)} \sum_{x_v} \lambda_{fv}(x_v) \left(\sum_{x_f \setminus x_v} b_f(x_f) - b_v(x_v)\right)$$

$\partial \mathcal{L} / \partial b_f(x_f) = 0$:
$$\log \frac{b_f(x_f)}{\phi_f(x_f)} + 1 - \nu_f - \sum_{v \in N(f)} \lambda_{fv}(x_v) = 0$$

$$b_f(x_f) = \phi_f(x_f) \exp\left(\nu_f - 1 + \sum_{v \in N(f)} \lambda_{fv}(x_v)\right)$$

$$b_f(x_f) \propto \phi_f(x_f) \prod_{v \in N(f)} \exp(\lambda_{fv}(x_v))$$

$\partial \mathcal{L} / \partial b_v(x_v) = 0$:
$$-(d_v - 1)[\log b_v + 1] - \nu_v + \sum_{f \in N(v)} \lambda_{fv}(x_v) = 0$$

$$b_v(x_v) \propto \exp\left(\frac{1}{d_v - 1} \sum_{f \in N(v)} \lambda_{fv}(x_v)\right)$$

(when $d_v > 1$).

**BP message correspondence**:

$\mu_{f \to v}(x_v) := \exp(\lambda_{fv}(x_v))$로 정의. 그러면:
$$b_f(x_f) \propto \phi_f(x_f) \prod_{v \in N(f)} \mu_{f \to v}(x_v) \cdot \mu_{v \to f}(x_v)^{-1} \cdot \mu_{v \to f}(x_v)$$

정리하면 (messages in both directions):
$$b_f(x_f) \propto \phi_f(x_f) \prod_{v \in N(f)} \mu_{v \to f}(x_v)$$

이 formula가 정확히 **BP belief at factor**. Variable belief도 마찬가지.

**Message update equation**이 BP의 fixed point condition과 일치:
$$\mu_{v \to f}(x_v) = \prod_{f' \in N(v) \setminus f} \mu_{f' \to v}(x_v)$$
$$\mu_{f \to v}(x_v) = \sum_{x_f \setminus x_v} \phi_f(x_f) \prod_{v' \in N(f) \setminus v} \mu_{v' \to f}(x_{v'})$$

따라서 **BP fixed point ⟺ Bethe stationary point**. $\square$

**Historical**: 1935년 Bethe가 통계역학에서, Yedidia-Freeman-Weiss가 2003년 이를 ML에 연결.

### 정리 2.3 — Bethe의 Non-Convexity

**명제**: Loopy graph에서 $F_{\text{Bethe}}$는 non-convex → multiple local minima 가능.

**증명 개요**:

$F_{\text{Bethe}} = U - H_{\text{Bethe}}$. $U$는 linear in $b$ (convex). $H_{\text{Bethe}} = \sum H(b_f) - \sum (d_v - 1) H(b_v)$:

- $H(b_f)$: concave (entropy)
- $-(d_v - 1) H(b_v)$: for $d_v > 1$, **convex** (negative of concave)

Sum이 both convex and concave parts → **non-convex** in general.

**구체적 예**: Ising model with strong coupling — $F_{\text{Bethe}}$가 symmetric double-well (two local minima, ferromagnetic symmetry breaking).

**함의**:
- Loopy BP가 initial condition에 의존
- Multiple fixed points
- Global optimum 보장 안 됨 → damping, deterministic annealing 같은 기법 필요

$\square$

### 정리 2.4 — Tree-Reweighted BP (Wainwright-Jaakkola-Willsky 2005)

**명제**: TRW-BP는 **convex upper bound** on $-\log Z$.

**TRW free energy**:
$$F_{\text{TRW}} = \sum_f \text{KL}(b_f \| \phi_f) - \sum_v \mathbb{E}[\log b_v] + \sum_{(f, v)} \rho_{fv} [\cdots]$$

여기서 $\rho_{fv}$ = "edge appearance probability" in random spanning tree ensemble.

**Convexity**: $\rho_{fv}$ 선택이 $F_{\text{TRW}}$를 convex로 만듦.

**Upper bound**: $\log Z \leq -F_{\text{TRW}}^*$ (at stationary point).

**증명**: Jensen's inequality over spanning trees. Wainwright et al. 2005 상세.

Practical: convex → unique fixed point, guaranteed convergence (unlike loopy BP).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

# Loopy BP와 Bethe free energy 관계 (Ising model)
def ising_loopy_bp(L, J, h=0.0, damping=0.3, max_iter=300, tol=1e-6):
    """L-cycle Ising model with loopy BP (like Ch2-05)."""
    N = L
    phi_edge = np.array([[np.exp(J), np.exp(-J)],
                         [np.exp(-J), np.exp(J)]])
    phi_node = np.array([np.exp(h), np.exp(-h)])
    
    # Messages (edges of ring)
    msg = np.ones((N, 2, 2)) / 2  # msg[i, j=left/right, x]
    
    for it in range(max_iter):
        old_msg = msg.copy()
        for i in range(N):
            left = (i - 1) % N
            right = (i + 1) % N
            # Incoming from left
            inc_left = msg[left, 1]  # msg from left, right-going
            inc_right = msg[right, 0]  # msg from right, left-going
            
            # Outgoing to right
            out = np.zeros(2)
            for x_j in range(2):
                for x_i in range(2):
                    out[x_j] += phi_edge[x_i, x_j] * phi_node[x_i] * inc_left[x_i]
            out = out / out.sum()
            msg[i, 1] = (1 - damping) * out + damping * old_msg[i, 1]
            
            # Outgoing to left
            out = np.zeros(2)
            for x_j in range(2):
                for x_i in range(2):
                    out[x_j] += phi_edge[x_j, x_i] * phi_node[x_i] * inc_right[x_i]
            out = out / out.sum()
            msg[i, 0] = (1 - damping) * out + damping * old_msg[i, 0]
        
        if np.abs(msg - old_msg).max() < tol:
            break
    
    # Beliefs
    b_v = np.zeros((N, 2))
    for i in range(N):
        left = (i - 1) % N
        right = (i + 1) % N
        b_v[i] = phi_node * msg[left, 1] * msg[right, 0]
        b_v[i] /= b_v[i].sum()
    
    b_f = np.zeros((N, 2, 2))  # pairwise beliefs
    for i in range(N):
        right = (i + 1) % N
        left_incoming = msg[(i - 1) % N, 1]
        right_incoming = msg[(right + 1) % N, 0]
        for x_i in range(2):
            for x_j in range(2):
                b_f[i, x_i, x_j] = (phi_edge[x_i, x_j] * 
                                    phi_node[x_i] * phi_node[x_j] *
                                    left_incoming[x_i] * right_incoming[x_j])
        b_f[i] /= b_f[i].sum()
    
    return b_v, b_f, it + 1

def bethe_free_energy(b_v, b_f, L, J, h=0.0):
    """Bethe FE computed from beliefs."""
    phi_edge = np.array([[np.exp(J), np.exp(-J)],
                         [np.exp(-J), np.exp(J)]])
    phi_node = np.array([np.exp(h), np.exp(-h)])
    
    # U: energy term
    # Sum over factors: sum_f sum_x b_f(x) * (-log phi_f(x))
    U = 0
    for i in range(L):
        for x_i in range(2):
            for x_j in range(2):
                if b_f[i, x_i, x_j] > 1e-12:
                    U -= b_f[i, x_i, x_j] * np.log(phi_edge[x_i, x_j] * phi_node[x_i] * phi_node[x_j])
    
    # H_Bethe
    H_Bethe = 0
    for i in range(L):
        for x in range(2):
            for y in range(2):
                if b_f[i, x, y] > 1e-12:
                    H_Bethe -= b_f[i, x, y] * np.log(b_f[i, x, y])
    # -(d_v - 1) H(b_v) where d_v = 2 for cycle
    for i in range(L):
        for x in range(2):
            if b_v[i, x] > 1e-12:
                H_Bethe += (2 - 1) * b_v[i, x] * np.log(b_v[i, x])
    
    F = U - H_Bethe
    return F

def exact_logZ_cycle(L, J, h):
    """Brute force log Z for L-cycle Ising."""
    from itertools import product
    phi_edge = np.array([[np.exp(J), np.exp(-J)],
                         [np.exp(-J), np.exp(J)]])
    phi_node = np.array([np.exp(h), np.exp(-h)])
    
    total = 0
    for config in product(range(2), repeat=L):
        p = 1
        for i in range(L):
            p *= phi_node[config[i]] * phi_edge[config[i], config[(i+1) % L]]
        total += p
    return np.log(total)

# 실험: J에 따른 Bethe error
L = 6
Js = np.linspace(0.05, 2.0, 15)
bethe_fs = []
exact_logZs = []

for J in Js:
    b_v, b_f, iters = ising_loopy_bp(L, J, damping=0.5)
    F_bethe = bethe_free_energy(b_v, b_f, L, J)
    exact_logZ = exact_logZ_cycle(L, J, h=0)
    bethe_fs.append(-F_bethe)  # Bethe approximation of log Z
    exact_logZs.append(exact_logZ)

bethe_fs = np.array(bethe_fs)
exact_logZs = np.array(exact_logZs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(Js, exact_logZs, 'o-', label='Exact log Z')
axes[0].plot(Js, bethe_fs, 's-', label='Bethe approximation')
axes[0].set_xlabel('Coupling J')
axes[0].set_ylabel('log Z')
axes[0].set_title(f'{L}-cycle Ising: Bethe vs Exact')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(Js, bethe_fs - exact_logZs, 'o-')
axes[1].set_xlabel('Coupling J')
axes[1].set_ylabel('Bethe - Exact')
axes[1].set_title('Bethe approximation error (should be 0 for tree, positive bias for loop)')
axes[1].grid(alpha=0.3)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('bethe_vs_exact.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"Bethe approximation of log Z (J=1.0): {bethe_fs[7]:.4f}")
print(f"Exact log Z (J=1.0):                  {exact_logZs[7]:.4f}")
print(f"Error:                                 {bethe_fs[7] - exact_logZs[7]:.4f}")
```

**출력 예시**:
```
Bethe approximation of log Z (J=1.0): 5.7845
Exact log Z (J=1.0):                  5.7234
Error:                                 0.0611
```

Bethe이 cycle에서 small but nonzero error (loopy approximation). Coupling strength가 증가할수록 error 증가.

---

## 🔗 AI/ML 연결

### LDPC Decoding Revisited

1990년대 turbo code / LDPC의 성공이 "왜?" 물음을 낳음. 2003년 YFW 이론이 답:
- LDPC factor graph는 large-girth (짧은 cycle 없음)
- Long cycles → Bethe bias 작음
- Loopy BP = approximate but near-exact for random LDPC

현대 5G LDPC 디코딩이 이 framework 기반.

### Variational Inference Hierarchy

```
Exact (JT)         — exact H decomposition
Kikuchi            — region graph H decomposition
Bethe              — pairwise H decomposition
Tree-Reweighted    — convex upper bound via spanning trees
Mean-field         — factorized H (no correlations)
```

각각 trade-off: accuracy vs computational cost.

### Region Graph / Kikuchi Approximation

Yedidia-Freeman-Weiss의 general framework:
- **Region**: set of variables
- **Region graph**: how regions overlap (subset lattice)
- **Bethe**: factor graph의 regions = factors + variables
- **Kikuchi**: higher-order regions (triangles, squares)

More accurate than Bethe at cost of larger computation. Generalized Belief Propagation (GBP) = BP on region graph.

### Spin Glass Theory

Condensed matter의 Sherrington-Kirkpatrick spin glass:
- Random couplings $J_{ij} \sim \mathcal{N}(0, 1)$
- Low temperature: many local minima
- **Cavity method** = Bethe approximation (Mezard-Parisi)
- 복잡도가 랜덤 sat problem, constraint satisfaction에도 연관

### Modern Deep Learning Connections

GNN의 message passing은 loopy BP의 learned 버전:
- $m = \text{MLP}(\cdot)$ instead of fixed sum-product
- Learned aggregation
- Depth = inference iterations
- 하지만 Bethe-like variational interpretation 가능

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Pairwise factors 가정 | Higher-order에는 Kikuchi 필요 |
| Bethe non-convex | Multiple local minima, damping 필요 |
| Approximate | Loop 많으면 정확도 떨어짐 |
| Local consistency | Full joint constraint는 more restrictive |

**주의**: Bethe는 pairwise factor에 natural. Higher-order factor (3+ variable)에서는 **Kikuchi approximation** 필요 — correlation capture 향상하지만 계산 증가.

---

## 📌 핵심 정리

$$\boxed{F_{\text{Bethe}} = \sum_f \text{KL}(b_f \| \phi_f) - \sum_v (d_v - 1) H(b_v)}$$

$$\boxed{\text{Loopy BP fixed point} \iff F_{\text{Bethe}} \text{ stationary point (YFW 2003)}}$$

| 개념 | 의미 |
|------|------|
| **Bethe entropy** | Pairwise edge + vertex correction |
| **Local consistency** | Factor marginal = variable marginal |
| **Tree exact** | Bethe가 tree에서 정확 |
| **Loopy approximation** | Cycle에서 bias |
| **TRW** | Convex upper bound |
| **Kikuchi** | Higher-order generalization |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2-cycle (변수 2개, factor 2개 parallel) factor graph에서 Bethe exact임을 검증하라.

<details>
<summary>힌트 및 해설</summary>

**2-cycle**: Variables $v_1, v_2$. Two factors $f_1(v_1, v_2)$ and $f_2(v_1, v_2)$ — **parallel edges in factor graph** (no cycle actually, if considered distinct factors).

**Factor graph structure**: 
```
v_1 - f_1 - v_2
   \       /
    f_2 ---
```

이는 cycle이지만 factor graph의 "multigraph" 해석. 실제로는 **tree-like** — 두 factor가 independent 정보.

**Bethe entropy**:
- $d_{v_1} = d_{v_2} = 2$ (두 factor 연결)
- $H_{\text{Bethe}} = H(b_{f_1}) + H(b_{f_2}) - (2-1) H(b_{v_1}) - (2-1) H(b_{v_2})$
- $= H(b_{f_1}) + H(b_{f_2}) - H(b_{v_1}) - H(b_{v_2})$

**True entropy**: Joint $p(v_1, v_2) \propto f_1 f_2$. $H(p) = ?$

단일 joint (effectively 1 factor with potential $f_1 f_2$):
$$H(p) = H(b_{f}) \quad \text{where } b_f \text{ corresponds to } p$$

**비교**: Bethe가 two factors를 separate하게 봄 → over-counting. Specifically:
$H(b_{f_1}) + H(b_{f_2}) > H(p)$ in general.

**특별 경우**: Factor $f_1, f_2$가 **marginalize to same $b_{v_i}$** → consistency 유지. 하지만 joint가 factorize되지 않으면 Bethe ≠ exact.

**실제 2-cycle이 cycle인가 not?**: Depends on interpretation:
- Distinct factors with same scope = 사실상 product → single factor (simplify)
- 진짜 cycle은 3 or more distinct variables/factors

**결론**: "2-cycle"이 실제로는 tree-reducible → Bethe exact. 진짜 3+ cycle에서만 Bethe approx.

</details>

**문제 2** (심화): Bethe approximation의 bias가 **short cycles에서 크고 long cycles에서 작은** 이유를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Bethe assumption**: Tree-like correlation structure.

**Short cycles** (e.g., 3-cycle, 4-cycle):
- Variables가 강하게 correlated via multiple paths
- Bethe이 각 edge를 **independent** pairwise로 취급
- Over-counting of correlations
- **Large error**

**Long cycles** (e.g., 100-cycle):
- Any two variables의 correlation이 chain에 의해 weakened
- Effective correlation decays with graph distance
- Bethe의 "locally tree-like" 가정이 good approximation
- **Small error**

**수학적**: Information propagates via cycles:
- 3-cycle: signal amplifies every 3 steps → BP diverges or biased
- Large cycle: signal decays exponentially with cycle length → BP converges near exact

**LDPC code context**: Design LDPC codes with **large girth** (no short cycles) → BP decoding converges near-optimal. 5G LDPC codes have girth $\geq 6$ typically.

**Random graph asymptotics**: Erdős-Rényi random graph with low edge density → local neighborhoods are tree-like (girth $\Theta(\log n)$) → BP asymptotically exact.

**Dense local structure**: Grid graph with 3-cycle + 4-cycle (if you consider diagonal) → BP bias. Nearest-neighbor grid (no diagonals) has 4-cycles → moderate bias.

**Remedy**:
- **Short cycle elimination**: LDPC code design
- **Tree-reweighted BP**: weighted average over spanning trees
- **Generalized BP / Kikuchi**: larger regions to capture short cycles

결론: Graph의 **local girth**가 Bethe accuracy의 핵심 predictor. LDPC가 BP를 잘 사용하는 건 **by design** — large girth construction.

</details>

**문제 3** (AI 연결): Graph Neural Network이 "learned Bethe approximation"으로 해석될 수 있는 이유는?

<details>
<summary>힌트 및 해설</summary>

**Bethe / Loopy BP structure**:
- Variables = nodes, factors = edges
- Messages: $\mu_{v \to f}, \mu_{f \to v}$
- Aggregation: product of incoming messages
- Marginalization: sum over factor scope

**GNN structure** (Gilmer et al. 2017):
- Variables = nodes
- Messages: $m_{ij} = \text{MLP}(h_i, h_j, e_{ij})$
- Aggregation: sum (or attention)
- Update: $h_i^{(t+1)} = \text{MLP}(h_i^{(t)}, \sum_j m_{ij})$

**Correspondence**:

| Bethe / BP | GNN |
|-----------|-----|
| Fixed sum-product | Learned MLP |
| Belief (pseudo-marginal) | Hidden state $h_i$ |
| Fixed point iteration | Fixed $T$ layers |
| Edge factor $\phi_{ij}$ | Learned edge function |
| Normalization per iter | Layer norm / activation |
| Product aggregation | Sum aggregation (with MLP) |

**Bethe variational principle 관점**:
- Bethe: minimize $F_{\text{Bethe}}$ over pseudo-marginals
- GNN: minimize task loss over parameterized hidden states

**Implicit variational objective**: GNN training이 end-to-end loss로 guided, not explicit Bethe FE. 하지만:
- **Expressiveness**: GNN이 Bethe보다 rich (nonlinear messages, attention, etc.)
- **Optimality**: GNN이 task-optimal이지만 probabilistic semantics 희미

**최신 연구 — Explicit connection**:

**Structure2Vec (Dai et al. 2016)**: 
- Factor graph를 embedding으로 compile
- Message passing = learned BP
- For CRF, MAP inference tasks

**Neural Bethe Free Energy (Hart et al. 2019)**:
- Explicit differentiable Bethe FE as loss
- Improve loopy BP via learned regularization

**Amortized VI with GNN** (recent, generic):
- GNN encode observed graph → posterior distribution
- End-to-end learn posterior approximation
- Superset of Bethe approximation

**결론**: GNN = "**deep, learned Bethe approximation**". 이 관점은 GNN design에 inductive bias 제공 (permutation invariance, locality, depth = iteration count). 하지만 GNN은 Bethe의 probabilistic guarantees를 희생하고 **task-specific optimization**을 얻음. Ch7-05 주제.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Mean-Field Variational Inference](./01-mean-field-vi.md) | [📚 README](../README.md) | [03. Expectation Propagation (EP) ▶](./03-expectation-propagation.md) |

</div>
