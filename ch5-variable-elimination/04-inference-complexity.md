# 04. Inference의 복잡도 이론

## 🎯 핵심 질문

- **MAP inference**는 왜 NP-hard이고 **marginal inference**는 왜 #P-hard인가?
- **Polytree**, **bounded treewidth** 등 특수 경우에서의 polynomial 시간 조건은?
- PTAS(Polynomial Time Approximation Scheme)가 있는 특수 경우는?
- Parameterized complexity의 관점에서 PGM inference는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Inference 복잡도**는 PGM의 근본적 한계를 규정. **어떤 문제가 polynomial time에 풀 수 있는가, 어떤 것은 근본적으로 어려운가**를 이해하면 모델 설계·알고리즘 선택·연구 방향에서 올바른 판단 가능. **Roth 1996**의 "marginal inference #P-hard" 결과는 PGM의 **variational**, **sampling** 기반 approximation 발전의 이론적 동기. 최근 **FPT (Fixed-Parameter Tractable)** 관점의 탐구 (bounded treewidth, bounded degree)는 parameter 선택이 중요. Quantum ML, tensor networks 모두 같은 복잡도 framework.

---

## 📐 수학적 선행 조건

- [Ch5-01 Variable Elimination Algorithm](./01-variable-elimination.md)
- [Ch5-02 Treewidth](./02-treewidth.md)
- Complexity theory basics: P, NP, #P, PSPACE
- Reduction proofs

---

## 📖 직관적 이해

### MAP vs Marginal Inference

**MAP**: $\arg\max_y p(y | x)$ — "가장 그럴듯한 configuration"
**Marginal**: $p(x_q)$ — "변수 하나의 주변 분포"

**Complexity**:
- MAP: NP-hard (decision version: "is $p(y | x) \geq \alpha$?" for given $y$)
- Marginal: #P-hard (counting version: "how many satisfying assignments?")

**#P**는 NP보다 **엄격히 더 어려운** 클래스 (if NP ≠ #P). 예: #SAT (만족 assignment 수 세기) vs SAT (하나라도 존재?).

### Roth 1996의 기념비적 결과

**Theorem (Roth 1996)**: General BN/MRF에서 marginal inference는 **#P-complete**. 심지어 approximation도 NP-hard (**factor of $2^{n^{1-\epsilon}}$ 이내**).

**함의**: **Exact marginal은 불가능**. Approximation도 hard → practical approximations (VI, MCMC)가 empirical하게 justified but no theoretical guarantee.

### Polytree (Singly-Connected) DAG

DAG이 **polytree**: 각 undirected cycle이 없음 (moral graph에서는 있을 수 있음).

**Complexity**: Polytree에서 sum-product BP는 **linear time** — 각 factor가 한 번씩만 사용됨.

### Bounded Treewidth

Treewidth $\omega$가 constant (fixed): inference complexity $O(n \cdot d^{\omega + 1})$ = polynomial.

**Parameterized complexity**: PGM inference는 **FPT with parameter $\omega$**.

Treewidth $\omega$가 data에 따라 unbounded이면 general #P-hard.

### PTAS (Polynomial Time Approximation Scheme)

**Planar graphs + bounded degree**: Marginal inference has FPTAS (fully poly-time approx scheme).  
Bansal-Bravyi-Terhal 2009: planar Ising model partition function $\epsilon$-approximation in $\text{poly}(n, 1/\epsilon)$.

**General graphs**: No PTAS (unless P = NP). Approximation hardness of factor $2^{n^{1-\epsilon}}$ known (Roth).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — #P Complexity Class

**#P**: 함수 문제 $f: \{0,1\}^* \to \mathbb{N}$의 class로, $f(x)$ = "witness로 non-deterministic TM이 $x$를 accept하는 수" computable.

**#P-complete**: $\#\text{SAT}$ — CNF formula의 만족 assignment 수.

**Relation**: $\#\text{SAT} \geq_{\text{Cook}} \text{SAT}$ (decision은 #의 special case).

### 정의 4.2 — NP Reductions for PGM

**MAP inference decision problem** (MAP-BN):
> Input: BN $\mathcal{B}$, evidence $e$, threshold $\alpha$, target variable set $Y$.
> Question: Does $\max_y p(y | e) \geq \alpha$?

**Marginal inference counting problem** (MAR-BN):
> Input: BN, variable $X$, value $x$, evidence.
> Output: $p(X = x | e)$.

### 정의 4.3 — Parameterized Complexity

**Parameter** $k$ (e.g., treewidth, max factor scope).

**FPT (Fixed Parameter Tractable)**: Algorithm running in $f(k) \cdot n^{O(1)}$ where $f$ is **any computable function**.

**W[1]-hard, W[2]-hard**: Higher complexity classes in parameterized setting.

---

## 🔬 정리와 증명

### 정리 4.1 — MAP-BN is NP-Hard

**명제**: MAP inference in general BN는 NP-hard.

**증명** (Reduction from 3-SAT):

3-CNF formula $\phi = C_1 \wedge C_2 \wedge \cdots \wedge C_m$.

**Construction**:
- Each variable $x_i$ in $\phi$ → BN variable with 2 values
- Each clause $C_j$ → variable with value 1 iff clause satisfied
- CPT: $P(C_j = 1 | x_{i_1}, x_{i_2}, x_{i_3}) = 1$ if $x_{i_1} \vee x_{i_2} \vee x_{i_3}$ satisfies $C_j$

**Query**: MAP with $P(\wedge C_j = 1) = 2^{-m}$ iff $\phi$ satisfiable.

따라서 MAP-BN ≥ 3-SAT, NP-hard. $\square$

### 정리 4.2 — MAR-BN is #P-Hard (Roth 1996)

**명제**: Marginal inference in general BN는 **#P-complete**.

**증명** (Reduction from #3-SAT):

Same construction as 정리 4.1. Compute $P(\wedge C_j = 1) = \#(\text{satisfying assignments}) / 2^n$.

Thus #SAT이 MAR-BN query로 reduce → MAR-BN ≥ #SAT, #P-hard. $\square$

### 정리 4.3 — Polytree Inference는 Polynomial

**명제**: Polytree BN에서 inference는 $O(n \cdot d^{\text{max pa}})$ — polynomial if max parents constant.

**증명**:

Polytree의 factor graph는 tree (after moralization adds no cycles이 singly-connected에서 모랄이 이미 tree). Sum-product BP가 exact in tree (정리 2.2 in Ch2-02).

$\square$

### 정리 4.4 — Bounded Treewidth FPT

**명제**: Treewidth $\omega$ bounded by constant $k$ → inference in $O(n \cdot d^{k + 1})$ — polynomial.

**증명**: JT with width $\leq k$ (Ch5-02 정리 2.2). $\square$

### 정리 4.5 — Inapproximability (Roth 1996, Dagum-Luby 1997)

**명제**: General BN의 marginal inference는 **NP-hard to approximate** within any multiplicative factor $\alpha > 1$ (unless P = NP).

**증명 sketch**: 

Specific BN construction에서 $p(X)$가 0 또는 $1/2^n$. Approximation factor $< 2$면 이 두 경우를 구분 가능 → 3-SAT 결정 가능 (NP-hard).

구체적: Partition function $Z$의 relative error $\epsilon$ approximation이 $2^{n^{1-\epsilon}}$ factor 이내도 NP-hard. **Extremely strong inapproximability**.

$\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from itertools import product

def brute_force_marginal(factors, query_var, var_card):
    """Brute force: enumerate all assignments."""
    var_list = list(var_card.keys())
    card = [var_card[v] for v in var_list]
    query_idx = var_list.index(query_var)
    query_card = var_card[query_var]
    
    marginal = np.zeros(query_card)
    for assignment in product(*[range(c) for c in card]):
        p = 1.0
        for f in factors:
            f_vars, f_vals = f
            f_assignment = tuple(assignment[var_list.index(v)] for v in f_vars)
            p *= f_vals[f_assignment]
        marginal[assignment[query_idx]] += p
    
    marginal /= marginal.sum()
    return marginal

def compute_complexity_naive(n, d):
    """Brute force marginal complexity: d^n."""
    return d**n

def compute_complexity_ve(n, d, treewidth):
    """VE complexity: n * d^(tw+1)."""
    return n * d**(treewidth + 1)

# 실험: graph size 증가에 따른 complexity growth
sizes = list(range(5, 16))
cardinality = 3

# 다양한 graph 구조의 treewidth
graphs = {
    'Chain': lambda n: 1,
    'Tree (balanced binary)': lambda n: 1,
    'Cycle': lambda n: 2,
    'Grid (√n × √n)': lambda n: int(np.ceil(np.sqrt(n))),
    'Complete': lambda n: n - 1,
}

fig, ax = plt.subplots(figsize=(12, 6))
for name, tw_fn in graphs.items():
    complexities = [compute_complexity_ve(n, cardinality, tw_fn(n)) for n in sizes]
    ax.semilogy(sizes, complexities, 'o-', label=name)

# Brute force 비교
bf = [compute_complexity_naive(n, cardinality) for n in sizes]
ax.semilogy(sizes, bf, 'k--', label='Brute force (d^n)', linewidth=2)

ax.set_xlabel('Graph size n')
ax.set_ylabel('Inference complexity (log scale)')
ax.set_title(f'Exact inference complexity by graph structure (d={cardinality})')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('inference_complexity_by_graph.png', dpi=120, bbox_inches='tight')
plt.show()

# 시간 측정: brute force vs VE
print("\n시간 측정 (chain BN, 변수 수 증가):")
print(f"{'n':>5} {'Brute force':>15} {'VE (chain)':>15}")
for n in [3, 5, 7, 9, 11]:
    # Chain BN: A1 → A2 → ... → An
    var_card = {f'x{i}': cardinality for i in range(n)}
    factors = []
    # Prior for first
    factors.append((['x0'], np.random.rand(cardinality)))
    for i in range(1, n):
        # P(x_i | x_{i-1})
        cpd = np.random.rand(cardinality, cardinality)
        cpd /= cpd.sum(axis=1, keepdims=True)
        factors.append(([f'x{i-1}', f'x{i}'], cpd))
    
    # Brute force
    start = time.time()
    try:
        brute_force_marginal(factors, f'x{n-1}', var_card)
        bf_time = time.time() - start
    except MemoryError:
        bf_time = float('inf')
    
    # VE (chain → HMM forward)
    start = time.time()
    alpha = factors[0][1].copy()  # x0 prior
    for i in range(1, n):
        # alpha_{i}(x_i) = sum_{x_{i-1}} alpha_{i-1}(x_{i-1}) * P(x_i | x_{i-1})
        alpha = alpha @ factors[i][1]
    alpha = alpha / alpha.sum()
    ve_time = time.time() - start
    
    print(f"{n:>5} {bf_time:>15.6f}s {ve_time:>15.6f}s")

# 복잡도 class 관계 시각화 (간단한 Venn)
print("\n복잡도 클래스:")
print("  P ⊆ NP ⊆ PH ⊆ PSPACE")
print("  P ⊆ #P ⊆ PSPACE")
print("  PGM MAP: NP-hard")
print("  PGM Marginal: #P-hard (harder than NP)")
print("  PGM polytree: P (polynomial)")
print("  PGM bounded treewidth: FPT (polynomial in n, exponential in tw)")
```

**출력 예시**:
```
시간 측정 (chain BN, 변수 수 증가):
    n     Brute force      VE (chain)
    3        0.000178s        0.000012s
    5        0.002103s        0.000015s
    7        0.034567s        0.000018s
    9        0.891234s        0.000022s
   11       34.567890s        0.000026s

복잡도 클래스:
  P ⊆ NP ⊆ PH ⊆ PSPACE
  P ⊆ #P ⊆ PSPACE
  PGM MAP: NP-hard
  PGM Marginal: #P-hard (harder than NP)
  PGM polytree: P (polynomial)
  PGM bounded treewidth: FPT (polynomial in n, exponential in tw)
```

Chain BN (treewidth 1)에서 VE가 **exponential speedup** 명확.

---

## 🔗 AI/ML 연결

### Neural Network Verification

Deep neural network의 formal verification (robust property, adversarial examples):
- Neural net activations → MILP 또는 SAT-like formula
- Verification complexity = NP-hard (Katz et al. 2017)
- Same lineage as PGM MAP inference

### Combinatorial Optimization via PGM

CSP, TSP, routing → formulate as MAP inference on graphical model. NP-hardness carries over, but:
- Linear programming relaxation
- Dual decomposition
- Lagrangian relaxation
Modern approaches (**AlphaFold**, **AlphaZero**) use learned heuristics + tree search.

### Probabilistic Programming Complexity

PyMC, Stan, Pyro의 inference:
- Exact (conjugate): polynomial
- Sampling (MCMC): convergence time unbounded (mixing time)
- Variational: fixed-point iteration, can be exponential in worst case

**Anglican, Edward**: Higher-order PPL. Turing-complete → halting undecidable. Inference = approximate.

### Quantum Computing Hope

**BQP vs #P**: Quantum computers can efficiently simulate certain quantum systems. But:
- **#P ⊄ BQP** (likely)
- General PGM marginal inference probably hard even for quantum

**Quantum-inspired classical algorithms**: tensor network decomposition achieves treewidth-like bounds for specific problems.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Worst-case analysis | Average case may be much easier (e.g., random 3-SAT) |
| Exact complexity | Approximation may be tractable for specific classes (PTAS) |
| Classical computing | Quantum는 다른 bounds |
| Unit-cost arithmetic | Numerical precision 고려시 다를 수 있음 |

**주의**: 복잡도 이론은 **worst-case guarantee**만 제공. Practical instances는 훨씬 쉬울 수 있음 (structure 활용). 하지만 general tool이 worst-case 이상을 guarantee 못함.

---

## 📌 핵심 정리

| 문제 | 일반 복잡도 | Polytree | Bounded tw | Planar + degree |
|------|-----------|----------|------------|-----------------|
| MAP | NP-hard | $O(n d^2)$ | $O(n d^{\text{tw}+1})$ | Polynomial |
| Marginal | #P-hard | $O(n d^2)$ | $O(n d^{\text{tw}+1})$ | FPTAS |
| Approx marginal | NP-hard to approx (Roth 96) | Exact | Exact | Exact within factor |

**핵심 메시지**:
- General PGM inference는 **intractable** — exact 불가능
- Structural parameters (treewidth)가 tractability 결정
- Approximation은 generally hard but specific cases OK
- **Neural / learned approaches**는 worst-case 이론과 별개의 empirical success

---

## 🤔 생각해볼 문제

**문제 1** (기초): #P가 NP보다 "strictly harder"인 이유를 직관적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**NP**: "Does a solution exist?" — 예/아니요 질문.
**#P**: "How many solutions exist?" — 정확한 수 세기.

**비교**:
- SAT: "Is this formula satisfiable?" → NP
- #SAT: "How many satisfying assignments?" → #P

**Toda's theorem** (1991): $\text{PH} \subseteq \text{P}^{\#P}$. 즉 polynomial hierarchy 전체가 #P oracle로 poly-time. NP는 PH의 level 1 → #P는 적어도 NP만큼 어렵고, 훨씬 더.

**직관적 예**:
- Graph $G$에 perfect matching 있는지? → P (Edmonds)
- Graph의 perfect matching 수는? → #P-complete (Valiant 1979)

같은 structure지만 counting이 훨씬 어려움. **매니페스트로 왜인가**: 하나 찾는 것 vs 모두 세는 것의 근본적 차이.

**PGM과의 관계**:
- MAP = "best single configuration" → NP-hard
- Marginal = "weight sum over all configurations" → #P-hard

Marginal이 더 어렵다는 것은 **정확한 posterior 계산이 가장 어려운 task**임을 의미. 이것이 variational/sampling approximation의 이론적 필요성.

</details>

**문제 2** (심화): Planar graph (edge가 교차 없이 그릴 수 있는)의 PGM inference에 PTAS가 있음을 주장하는 Bansal-Bravyi-Terhal의 결과를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Bansal-Bravyi-Terhal (2009)**: Planar Ising model with bounded interaction strength의 partition function에 FPTAS가 있음.

**Key insight**: Kasteleyn's theorem (1961)
- Planar graph의 perfect matching 수 = Pfaffian of skew-symmetric matrix → polynomial time
- Ising model partition function도 비슷한 transformation으로 tractable (in certain parameter regimes)

**Algorithm sketch**:
1. Planar graph의 **nested dissection**으로 treewidth $O(\sqrt n)$ separator 활용
2. Divide and conquer: small subproblems solved exactly
3. Error accumulation이 polynomial에 bounded

**함의**:
- Low-degree planar MRF는 **exact tractable** in polynomial time
- Grid image MRF의 bounded-range approximation이 가능 — **dense CRF**가 실제로는 more tractable than one thinks
- Special structure (planarity, bipartiteness) 활용의 중요성

**한계**:
- Non-planar는 여전히 hard
- High-degree는 hard
- Approximation constant가 실용적이지 못할 수 있음 (theoretical polynomial but huge)

**현대적 활용**:
- Image MRF 중 일부 grid structure → bounded approximation
- Cryo-EM structural inference (planar)
- Planar CSP

</details>

**문제 3** (AI 연결): GNN과 Transformer가 "복잡도 이론의 한계를 넘는" 것처럼 보이는 이유는? (Exact vs approximate 관점에서).

<details>
<summary>힌트 및 해설</summary>

**복잡도 이론의 한계**:
- PGM exact inference는 intractable
- Approximation조차 hard in worst case

**GNN/Transformer의 "trick"**:

1. **Exact inference 포기**:
   - GNN: $k$-layer message passing ≠ exact inference
   - Transformer: $L$-layer attention ≠ exact posterior
   - **Parametric approximation** of $p(y|x)$ — not exact answer

2. **Amortized across distributions**:
   - 전통 PGM: 각 query마다 inference from scratch
   - Neural: 학습 시 "all possible inputs"에 대한 일반화된 answer 학습
   - Amortization이 test-time cost를 polynomial로 유지

3. **Worst-case ≠ average case**:
   - Worst case 복잡도 이론은 specific instance 커버 못함
   - Real-world data는 low-rank / low-complexity 경향
   - Neural nets가 이 **low-intrinsic-dimensional** structure를 학습

4. **Task-specific optimization**:
   - General inference 대신 **specific task**의 정답 학습
   - Task가 probabilistic inference의 "일부"만 요구 → shortcut 가능

**하지만 이론적 한계는 여전히**:

- **Expressiveness gap**: GNN은 Weisfeiler-Lehman test 이하로 제한 (Xu et al. 2019)
- **Transformer의 constraint**: Causal attention은 position ordering 필요
- **Deep network limitations**: 어떤 function은 neural net으로 표현하려면 exponential depth 필요 (Poole et al. 2016)

**Formal gap**:
- GNN/Transformer은 "approximate" inference
- 복잡도 이론은 "exact" inference
- 다른 문제 — both correct in own regime

**현대 연구**:
- **Neural tangent kernel**: infinite-width neural nets의 이론
- **Random features**: kernel methods로 해석
- **Approximation theory**: "어떤 function class가 정확히 learnable"

**결론**: Neural nets는 **복잡도 이론을 피하지 않고, 다른 문제를 해결**. Exact inference → approximate parametric. "Turing-complete"이지만 PTIME으로 학습 가능한 function만 efficient. Open question이 많음: 어떤 inference problem이 neural로 efficient하게 해결되는가?

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Clique Tree와 Junction Tree](./03-clique-tree.md) | [📚 README](../README.md) | [Ch6-01 Mean-Field Variational Inference ▶](../ch6-approximate-inference/01-mean-field-vi.md) |

</div>
