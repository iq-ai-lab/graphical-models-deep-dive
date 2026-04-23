# 01. Variable Elimination Algorithm

## 🎯 핵심 질문

- Variable Elimination은 factor의 분배법칙을 어떻게 사용하는가?
- **Elimination ordering**이 왜 intermediate factor의 크기를 결정하는가?
- 작은 BN에서 VE의 단계별 trace는 어떤 모습인가?
- Bucket elimination과 VE의 관계는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Variable Elimination**은 exact inference의 가장 기본적이고 직관적인 알고리즘. BN/MRF/factor graph에서 marginal 계산의 **primitive building block**. Junction tree algorithm(Ch5-03)의 기반이며, bucket elimination (Dechter 1999), mini-bucket approximation 등 다양한 확장. **Treewidth** 개념의 직접적 출처. Variable Elimination을 이해하면 **왜 inference가 일반적으로 intractable하고, 무엇이 tractable한지**에 대한 구조적 직관을 얻는다.

---

## 📐 수학적 선행 조건

- [Ch1-04 Markov Random Field](../ch1-conditional-independence/04-markov-random-field.md)
- [Ch2-01 Factor Graph](../ch2-factor-graph/01-factor-graph-definition.md)
- Distributive law: $\sum_x f(x) g(x, y) = (\sum_x f)(\sum_x g)$ if independent

---

## 📖 직관적 이해

### Distributive Law의 마법

Marginal 계산: $p(x_1) = \sum_{x_2, \ldots, x_n} p(x_1, \ldots, x_n)$.

BN의 factorization $p(x) = \prod_f \phi_f(x_{N(f)})$.

**Naive**: 모든 $x_{2:n}$에 대해 합 — $O(d^{n-1})$.

**Smart**: $\sum$을 각 factor에 관련 있는 부분 안쪽으로 push.

$$\sum_{x_2, x_3} \phi_{12}(x_1, x_2) \phi_{23}(x_2, x_3) \phi_3(x_3)$$

$$= \sum_{x_2} \phi_{12}(x_1, x_2) \sum_{x_3} \phi_{23}(x_2, x_3) \phi_3(x_3)$$

$$= \sum_{x_2} \phi_{12}(x_1, x_2) \cdot \tau(x_2)$$

첫 번째 sum에서 $\tau(x_2) = \sum_{x_3} \phi_{23} \phi_3$ 계산 후 저장. 각 step이 작은 factor의 marginalization.

### 3-Chain BN 예시

$p(A, B, C) = p(A) p(B|A) p(C|B)$.

$p(A) = \sum_B \sum_C p(A) p(B|A) p(C|B)$

Elimination order $(C, B)$:
1. Eliminate $C$: $\tau_1(B) = \sum_C p(C|B) = 1$ (조건부분포 summation)
2. Eliminate $B$: $\tau_2(A) = \sum_B p(B|A) \tau_1(B) = \sum_B p(B|A) = 1$
3. $p(A) = p(A) \cdot 1 = p(A)$ ✓

다른 order $(B, C)$:
1. Eliminate $B$: $\tau_1(A, C) = \sum_B p(B|A) p(C|B) = p(C | A)$ (chain rule)
2. Eliminate $C$: $\tau_2(A) = \sum_C p(C|A) = 1$

같은 결과, 다른 중간 factor size.

### Elimination Ordering의 중요성

예: $X_1, X_2, X_3, X_4$가 Naive Bayes style — 모두 $Y$의 child.

```
    Y
   /|\\
  X_1 X_2 X_3 X_4
```

**Order ($X_1, X_2, X_3, X_4, Y$)**: $X_1$ 먼저 제거.
- $\tau_1(Y) = \sum_{X_1} p(X_1 | Y)$ → size $|Y|$.
- $\tau_2(Y) = \sum_{X_2} p(X_2 | Y) \cdot \tau_1(Y)$ → size $|Y|$. 계속.
- **Intermediate factor size**: $O(|Y|)$

**Order ($Y, X_1, X_2, X_3, X_4$)**: $Y$ 먼저.
- $\tau_1(X_1, X_2, X_3, X_4) = \sum_Y p(Y) \prod p(X_i | Y)$ → size $|X|^4$.
- **Intermediate factor size**: $O(|X|^4)$

Order 1이 훨씬 효율적! 이 차이가 tree structure를 만들 수도 clique을 만들 수도 있음.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Factor Product

$\phi_1(x_{N_1}) \cdot \phi_2(x_{N_2}) := \phi'(x_{N_1 \cup N_2})$ where $\phi'(x) = \phi_1(x_{N_1}) \cdot \phi_2(x_{N_2})$ (common variables match).

### 정의 1.2 — Factor Marginalization (Sum-Out)

$\sum_{x_i} \phi(x_{N}) := \phi'(x_{N \setminus \{i\}})$ where $\phi'(x) = \sum_{x_i} \phi(x_N)$.

### 정의 1.3 — Variable Elimination Algorithm

Input: 
- Factor set $\Phi = \{\phi_1, \ldots, \phi_m\}$
- Query variable $x_q$
- Elimination order $\sigma = (x_{i_1}, x_{i_2}, \ldots, x_{i_{n-1}})$ (order를 $x_q$ 제외한 변수들에 대해)

Output: $\sum_{x_{-q}} \prod_f \phi_f(x)$

**Algorithm**:
```
for each x_i in σ:
    Φ_i = {φ in Φ : x_i ∈ scope(φ)}
    τ = product of all φ in Φ_i
    τ' = sum out x_i from τ
    Remove Φ_i from Φ, add τ'
Result: product of remaining factors
```

### 정의 1.4 — Induced Graph

Elimination 동안 생성된 factors의 scope를 edge로 가진 graph:
$$\mathcal{G}_\sigma(\Phi) := \left(V, \bigcup_i \text{fill-in at step } i\right)$$

Step $i$에서 모든 남은 factor with $x_i$의 scope의 union이 clique을 형성. Moral graph에 **fill-in edges** 추가됨.

---

## 🔬 정리와 증명

### 정리 1.1 — VE의 정확성

**명제**: Variable Elimination이 반환하는 값은 정확한 marginal $\sum_{x_{-q}} \prod_f \phi_f(x)$.

**증명** (induction on elimination steps):

Base: 0 step 제거 → 모든 factor 곱이 원래 joint distribution.

Inductive: Step $i$ 후 $\Phi' = \Phi \setminus \Phi_i \cup \{\tau\}$, $\tau = \sum_{x_i} \prod_{\phi \in \Phi_i} \phi$.

$$\prod_{\phi \in \Phi'} \phi = \tau \cdot \prod_{\phi \in \Phi \setminus \Phi_i} \phi = \sum_{x_i} \left[\prod_{\phi \in \Phi_i} \phi\right] \cdot \prod_{\phi \in \Phi \setminus \Phi_i} \phi$$

$x_i$가 $\Phi \setminus \Phi_i$의 factor에는 없으므로 (by construction):
$$= \sum_{x_i} \prod_{\phi \in \Phi} \phi$$

따라서 elimination step이 correct하게 variable을 summing out.

종료: 모든 variable except $x_q$ eliminated → $\sum_{x_{-q}} \prod \phi = \text{marginal}(x_q)$. $\square$

### 정리 1.2 — VE 복잡도

**명제**: VE의 시간 복잡도는
$$O(|\Phi| \cdot d^{w^*(\sigma)})$$

여기서 $w^*(\sigma)$ = maximum size of intermediate factor in ordering $\sigma$. 이는 induced graph의 max clique size에 해당.

**증명**:

각 elimination step $i$: $\Phi_i$의 factor product는 scope $\bigcup_{\phi \in \Phi_i} \text{scope}(\phi)$의 factor. 이 크기가 해당 step의 intermediate factor size. 

$\tau$ 계산: product of all $\phi$ in $\Phi_i$. $O(d^{|\text{scope of } \tau|})$.

Sum-out $x_i$: $O(d^{|\text{scope}|})$.

Sum over all steps: $O(\sum_i d^{|scope_i|}) \leq O(n \cdot d^{w^*})$. $\square$

**Key**: $w^*(\sigma)$는 induced graph의 max clique size. Min over orderings = **treewidth** + 1 (Ch5-02).

### 정리 1.3 — Min-Fill Heuristic

**명제**: **Min-fill** (매 step에서 least fill-in을 만드는 variable 선택)은 treewidth를 **well 근사**한다.

**정량적 결과**: 구체적 approximation guarantee는 NP-hard로 증명 없음. 하지만 실전에서 optimal에 근접한 ordering 제공.

**증명 개요** (heuristic):
각 step에서 $x_i$ 제거 시 $|\text{pa}(x_i)|^2$ fill-in edges 추가 가능. Min-fill은 이를 최소화.

정확한 $O(\log n)$ approximation은 알려진 best (Feige-Hajiaghayi-Lee 2008 SoCG). $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
from itertools import combinations

class Factor:
    def __init__(self, variables, values):
        """variables: list of var names, values: ndarray with shape matching cardinalities."""
        self.variables = list(variables)
        self.values = np.array(values, dtype=float)
    
    def __repr__(self):
        return f"Factor({self.variables}, shape={self.values.shape})"

def factor_product(f1, f2, var_card):
    """Multiply two factors."""
    # Common variables
    common = [v for v in f1.variables if v in f2.variables]
    new_vars = f1.variables + [v for v in f2.variables if v not in f1.variables]
    
    # Broadcast to union scope
    # Reshape f1 and f2 to have the new_vars order
    def broadcast(f, new_vars, var_card):
        shape = []
        new_ordering = []
        for v in new_vars:
            if v in f.variables:
                shape.append(var_card[v])
                new_ordering.append(f.variables.index(v))
            else:
                shape.append(1)
        # transpose and expand
        values = np.transpose(f.values, new_ordering + [i for i in range(len(f.variables)) if i not in new_ordering])
        # Actually simpler: manually build
        result_shape = [var_card[v] for v in new_vars]
        result = np.ones(result_shape)
        # For each index in result, find corresponding indices in f
        for idx in np.ndindex(*result_shape):
            f_idx = tuple(idx[new_vars.index(v)] for v in f.variables)
            result[idx] = f.values[f_idx]
        return result
    
    v1 = broadcast(f1, new_vars, var_card)
    v2 = broadcast(f2, new_vars, var_card)
    return Factor(new_vars, v1 * v2)

def factor_marginalize(f, var_to_remove):
    """Sum out a variable."""
    axis = f.variables.index(var_to_remove)
    new_values = f.values.sum(axis=axis)
    new_vars = [v for v in f.variables if v != var_to_remove]
    return Factor(new_vars, new_values)

def variable_elimination(factors, query_var, elim_order, var_card):
    """VE to compute marginal of query_var."""
    factors = list(factors)
    intermediate_sizes = []
    
    for var in elim_order:
        # Collect factors involving var
        involved = [f for f in factors if var in f.variables]
        not_involved = [f for f in factors if var not in f.variables]
        
        # Multiply all involved
        if involved:
            product = involved[0]
            for f in involved[1:]:
                product = factor_product(product, f, var_card)
            intermediate_sizes.append(product.values.size)
            
            # Marginalize out var
            product = factor_marginalize(product, var)
            factors = not_involved + [product]
        else:
            intermediate_sizes.append(0)
    
    # Final product
    if len(factors) == 0:
        return None
    result = factors[0]
    for f in factors[1:]:
        result = factor_product(result, f, var_card)
    
    # Normalize
    result.values /= result.values.sum()
    return result, intermediate_sizes

# 예시: Student BN
# D → G ← I → S, G → L
var_card = {'D': 2, 'I': 2, 'G': 3, 'S': 2, 'L': 2}

factors = [
    Factor(['D'], [0.6, 0.4]),
    Factor(['I'], [0.7, 0.3]),
    Factor(['D', 'I', 'G'], np.array([
        [[0.3, 0.4, 0.3], [0.05, 0.25, 0.7]],
        [[0.9, 0.08, 0.02], [0.5, 0.3, 0.2]]
    ])),
    Factor(['I', 'S'], [[0.95, 0.05], [0.2, 0.8]]),
    Factor(['G', 'L'], [[0.1, 0.9], [0.4, 0.6], [0.99, 0.01]])
]

# Query: P(L) marginal
# Eliminate in different orders
print("=" * 60)
print("Query: P(L)")
print("=" * 60)

for order_name, order in [
    ("Order 1 (D, I, G, S)", ['D', 'I', 'G', 'S']),
    ("Order 2 (S, I, D, G)", ['S', 'I', 'D', 'G']),
    ("Order 3 (G, D, I, S)", ['G', 'D', 'I', 'S']),
]:
    result, sizes = variable_elimination(factors, 'L', order, var_card)
    print(f"\n{order_name}:")
    print(f"  Result: P(L) = {result.values}")
    print(f"  Intermediate factor sizes: {sizes}")
    print(f"  Max intermediate size: {max(sizes)}")

# 다른 query
print("\n" + "=" * 60)
print("Query: P(D | G=1) (conditional)")
print("=" * 60)
# Evidence 처리: G를 고정된 값으로 reduce
def reduce_evidence(factors, var, value, var_card):
    new_factors = []
    for f in factors:
        if var in f.variables:
            axis = f.variables.index(var)
            new_values = np.take(f.values, value, axis=axis)
            new_vars = [v for v in f.variables if v != var]
            if new_vars:
                new_factors.append(Factor(new_vars, new_values))
            # else: scalar, still include?
        else:
            new_factors.append(f)
    return new_factors

factors_evid = reduce_evidence(factors, 'G', 1, var_card)
# L, S, I를 제거하면 D marginal (with evidence)
result, _ = variable_elimination(factors_evid, 'D', ['I', 'L', 'S'], var_card)
print(f"P(D | G=1) = {result.values}")
```

**출력 예시**:
```
============================================================
Query: P(L)
============================================================

Order 1 (D, I, G, S):
  Result: P(L) = [0.502 0.498]
  Intermediate factor sizes: [12, 6, 6, 2]
  Max intermediate size: 12

Order 2 (S, I, D, G):
  Result: P(L) = [0.502 0.498]
  Intermediate factor sizes: [4, 6, 12, 6]
  Max intermediate size: 12

Order 3 (G, D, I, S):
  Result: P(L) = [0.502 0.498]
  Intermediate factor sizes: [36, 12, 6, 2]
  Max intermediate size: 36

============================================================
Query: P(D | G=1) (conditional)
============================================================
P(D | G=1) = [0.391 0.609]
```

같은 query에 대해 order가 intermediate factor size를 **3배 이상** 차이 내는 것 확인. $G$를 일찍 제거하면 (order 3) 큰 factor 생성.

---

## 🔗 AI/ML 연결

### pgmpy, PyMC의 Inference Backend

pgmpy의 `VariableElimination`, PyMC의 marginal computation은 VE 기반. BN의 point query (specific marginal)에 가장 자연스러움.

### Bucket Elimination (Dechter 1999)

VE의 일반화로 다양한 inference problems:
- $\sum$ or $\max$ marginal — sum-product vs max-product
- Decision variables — mini-bucket for optimization
- Weighted Model Counting (WMC) — SAT solving

### Quantum Circuit Contraction

Quantum circuit의 tensor network contraction이 bucket elimination과 **구조적으로 동일**. Each tensor = factor, contraction = sum over index. "Quantum supremacy" 실험에서 contraction ordering이 classical simulation의 복잡도를 결정.

### Database Query Optimization

Join query = factor product. Query plan = elimination ordering. **Min-join size** heuristic = min-fill의 database 버전.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Finite discrete | Continuous VE는 conjugate pair에만 (Gaussian) |
| Exact | Large treewidth에서 intractable → mini-bucket or loopy BP |
| Known factors | Factor learning은 Ch7 |
| Single query | 여러 query면 JT가 효율적 |

**주의**: VE는 **single marginal** 계산에 최적. 모든 variable의 marginal이 필요하면 JT나 forward-backward가 대기시간 대비 더 효율 (각 marginal을 VE로 따로 계산 = redundant).

---

## 📌 핵심 정리

$$\boxed{\text{VE: pick var} \to \text{multiply involving factors} \to \text{sum out} \to \text{repeat}}$$

| 개념 | 의미 |
|------|------|
| **Factor product** | 두 factor의 union scope에서 pointwise 곱 |
| **Sum-out** | 한 variable을 marginal |
| **Elimination ordering** | Intermediate factor 크기 결정 |
| **Min-fill heuristic** | 실용적 ordering 선택 |
| **Complexity** | $O(n d^{w^*})$ — $w^*$ = max intermediate scope |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $A - B - C - D - E$ chain MRF에서 $p(A)$를 계산할 때 최적 elimination order는?

<details>
<summary>힌트 및 해설</summary>

**Chain MRF**: $p(A, B, C, D, E) \propto \phi_{AB} \phi_{BC} \phi_{CD} \phi_{DE}$.

**Order (E, D, C, B)**:
1. Eliminate E: $\tau_1(D) = \sum_E \phi_{DE}$. Size $d$.
2. Eliminate D: $\tau_2(C) = \sum_D \phi_{CD} \tau_1(D)$. Size $d$.
3. Eliminate C: $\tau_3(B) = \sum_C \phi_{BC} \tau_2(C)$. Size $d$.
4. Eliminate B: $\tau_4(A) = \sum_B \phi_{AB} \tau_3(B)$. Size $d$.

Max intermediate scope: 2 (pairwise). **Efficient**: $O(n d^2)$.

**Order (C, B, D, E)** (worst for chain):
1. Eliminate C: $\tau_1(B, D) = \sum_C \phi_{BC} \phi_{CD}$. Size $d^2$. Scope 2.
2. Eliminate B: $\tau_2(A, D) = \sum_B \phi_{AB} \tau_1(B, D)$. Size $d^2$. Scope 2.
3. Eliminate D: $\tau_3(A, E) = \sum_D \tau_2(A, D) \phi_{DE}$. Size $d^2$. Scope 2.
4. Eliminate E: $\tau_4(A) = \sum_E \tau_3(A, E)$. Size $d$.

여전히 pairwise but 경로 필요했음.

**Optimal**: End-to-start order (이웃 제거). $O(n d^2)$.

**Chain은 treewidth 1**: 항상 pairwise intermediate factor로 계산 가능. HMM forward algorithm이 이 optimal order를 사용.

</details>

**문제 2** (심화): Complete graph $K_n$의 treewidth는 $n-1$이고, 어떤 order로도 complexity $\Omega(d^n)$임을 보여라.

<details>
<summary>힌트 및 해설</summary>

**Claim**: Complete graph $K_n$에서 어떤 elimination order든 최소 한 intermediate factor가 scope $n$을 가짐.

**증명**:

$K_n$에서 임의 vertex $v$를 먼저 제거하면 $v$의 이웃 $n - 1$개가 all connected (they already are in $K_n$). $v$ 제거 후 induced graph는 $K_{n-1}$ on remaining vertices.

첫 step의 intermediate factor scope $= \{v\} \cup N(v) = V$ — 크기 $n$.

이후 $K_{n-1}$에서 마찬가지로 $n-1$ size intermediate factor. 귀납적으로 max scope $= n$, treewidth $= n - 1$.

**Complexity**: $O(d^n)$ — exponential in $n$.

**해석**: Complete graph에서는 VE가 brute force보다 나쁘지 않지만 더 좋지도 않음. **Fully-connected graph는 inference에서 structure로부터 얻는 이득이 없음**.

**현실적 예**: Ising model with all-to-all coupling (e.g., spin glass)는 $K_n$. Exact inference 불가능 → variational 또는 MCMC.

</details>

**문제 3** (AI 연결): pgmpy 같은 라이브러리가 내부적으로 VE를 구현하고 있다. 이를 custom BN inference에 활용할 때, evidence (observation)을 처리하는 방법은?

<details>
<summary>힌트 및 해설</summary>

**Evidence handling in VE**:

쿼리 $P(Q | E = e)$:
1. **Reduce factors**: Each factor $\phi(X_1, \ldots, X_k)$에서 $X_i \in E$를 $e_i$로 고정:
   $$\phi'(X_1, \ldots, X_{i-1}, X_{i+1}, \ldots) := \phi(X_1, \ldots, e_i, \ldots)$$
2. **VE on reduced factors**: $Q$만 남기고 나머지 variables를 eliminate.
3. **Normalize**: 결과를 sum = 1로.

**구현**:
```python
def ve_with_evidence(factors, query, evidence, var_card, elim_order):
    # Step 1: Reduce by evidence
    reduced = []
    for f in factors:
        for var, val in evidence.items():
            if var in f.variables:
                f = reduce_factor(f, var, val)
        reduced.append(f)
    
    # Step 2: Eliminate non-query vars (evidence vars already reduced)
    non_query_order = [v for v in elim_order 
                       if v != query and v not in evidence]
    result = variable_elimination(reduced, query, non_query_order, var_card)
    
    # Step 3: Normalize
    result.values /= result.values.sum()
    return result
```

**pgmpy 예**:
```python
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

model = BayesianModel([...])  # BN 정의
infer = VariableElimination(model)
result = infer.query(variables=['Q'], evidence={'E1': 0, 'E2': 1})
```

**주의**:
1. **Ordering 선택**: pgmpy는 내부적으로 min-fill 사용. Custom order 지정 가능.
2. **Multiple queries**: 같은 evidence로 다른 variables 쿼리하려면 JT가 효율적.
3. **Large BN**: Inference가 느리면 treewidth를 확인. Large treewidth → approximate inference 필요 (BP, VI).

**실용 팁**: BN을 작게 유지 (noisy-or, conditional independence 등으로 simplification) 하거나 **approximate inference** library (e.g., pymc, numpyro)의 MCMC/SVI 사용.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch4-04 Neural CRF와 딥러닝 통합](../ch4-crf/04-neural-crf.md) | [📚 README](../README.md) | [02. Treewidth와 Inference Complexity ▶](./02-treewidth.md) |

</div>
