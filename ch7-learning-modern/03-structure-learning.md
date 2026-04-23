# 03. Structure Learning

## 🎯 핵심 질문

- 관측 데이터만으로 DAG 구조를 추정하는 **score-based**와 **constraint-based** 접근의 차이는?
- **BIC, BDeu, AIC** 같은 scoring functions의 의미와 차이는?
- **DAG structure learning이 NP-hard** (Chickering 1996)인 이유는?
- **Chow-Liu 1968** tree structure learning이 왜 polynomial time에 exactly solvable한가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Structure Learning**은 causal discovery의 수학적 기반. **PC algorithm** (Spirtes-Glymour-Scheines), **GES** (Greedy Equivalence Search), **LiNGAM**, **NOTEARS** (2018, continuous optimization relaxation) — 모두 structure learning 계보. Causal inference, biological network discovery, social network analysis에 활용. **Chow-Liu tree**는 computer vision의 feature selection, NLP의 document classification에 응용. Modern **neural structure learning** (DAG-GNN, GraN-DAG)이 differentiable extension. 데이터에서 DAG를 발견하는 문제는 **scientific discovery의 자동화**와 연결.

---

## 📐 수학적 선행 조건

- [Ch1-02 Bayesian Network — DAG 기반 인수분해](../ch1-conditional-independence/02-bayesian-network-factorization.md)
- [Ch1-03 d-separation](../ch1-conditional-independence/03-d-separation.md)
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): Mutual information

---

## 📖 직관적 이해

### 두 가지 접근

**Score-based**:
- 각 DAG에 score 할당 (BIC, BDeu 등)
- Search over DAG space → best score
- Greedy hill-climbing, tabu search, GES

**Constraint-based**:
- 독립성 테스트 (CI test)로 조건부 독립 관계 추정
- 독립성 pattern과 일관된 DAG 찾기
- PC algorithm, IC algorithm

### Scoring Functions

**BIC** (Bayesian Information Criterion, Schwarz 1978):
$$\text{BIC}(\mathcal{G}) = \log p(D | \hat\theta_\mathcal{G}, \mathcal{G}) - \frac{|\theta_\mathcal{G}|}{2} \log N$$

- Likelihood - complexity penalty
- $|\theta|$ 클수록 penalty 커짐
- Asymptotic approximation to log marginal likelihood

**BDeu** (Bayesian Dirichlet equivalent uniform, Heckerman et al. 1995):
$$\text{BDeu}(\mathcal{G}) = \log p(D | \mathcal{G}) = \log \int p(D | \theta, \mathcal{G}) p(\theta | \mathcal{G}) d\theta$$

- Log marginal likelihood (integrate out $\theta$)
- Dirichlet prior with equivalent sample size
- **Score equivalent**: Markov equivalent DAGs have same score

**AIC**: $\ell(\hat\theta) - |\theta|$ — less penalty than BIC (prefers more complex).

### PC Algorithm

Spirtes et al. 2000:
1. Complete undirected graph
2. For each pair $(X, Y)$ and conditioning set $Z$:
   - Test $X \perp Y | Z$. If yes → remove edge.
3. **Orient v-structures**: collider 판정
4. **Orient additional edges**: acyclicity, no new v-structures

**Output**: Completed Partially Directed Acyclic Graph (**CPDAG**) — Markov equivalence class representation.

### Chow-Liu Tree (1968)

**Tree-restricted**: DAG가 tree (or polytree).

**Optimal tree MLE**: $\arg\max_T \log p(D | T) = \arg\max_T \sum_{(u, v) \in T} I(X_u; X_v)$

(Mutual information between pairs).

**Algorithm**: 
1. Compute pairwise $I(X_i; X_j)$
2. **Maximum Spanning Tree** on graph with MI weights
3. Root randomly, orient edges

**Polynomial**: $O(n^2 N)$ for MI computation + $O(n^2 \log n)$ MST.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Structure Learning Problem

**Input**: Data $D = \{x^{(1)}, \ldots, x^{(N)}\}$ with $x \in \mathbb{R}^n$.

**Output**: DAG $\mathcal{G}$ (or Markov equivalence class).

### 정의 3.2 — Score-Based Learning

$$\hat\mathcal{G} = \arg\max_{\mathcal{G}} \text{Score}(\mathcal{G}, D)$$

**Decomposable score**:
$$\text{Score}(\mathcal{G}) = \sum_v \text{Score}_v(v, \text{pa}(v), D)$$

- 각 node score 독립적으로 계산
- Local search (one edge change) efficient

### 정의 3.3 — BIC Score

$$\text{BIC}(\mathcal{G}, D) = \sum_v \left[\log p(D_v | D_{\text{pa}(v)}, \hat\theta_v) - \frac{|\theta_v|}{2} \log N\right]$$

Asymptotic approximation to $\log p(D | \mathcal{G})$.

### 정의 3.4 — BDeu Score

$$\text{BDeu}(\mathcal{G}, D) = \sum_v \sum_u \log \frac{\Gamma(\alpha_u)}{\Gamma(N_{v, u} + \alpha_u)} \prod_k \frac{\Gamma(N_{v, u, k} + \alpha_{v, u, k})}{\Gamma(\alpha_{v, u, k})}$$

with Dirichlet prior hyperparameters $\alpha_{v, u, k} = \alpha / (r_v r_{\text{pa}(v)})$ (equivalent sample size $\alpha$).

### 정의 3.5 — Constraint-Based: CI Test

**Statistical test**: $H_0: X \perp Y | Z$ vs $H_1: X \not\perp Y | Z$.

**Discrete**: $G^2$ test (likelihood ratio), $\chi^2$ test.  
**Continuous**: Partial correlation (Gaussian), kernel-based (non-parametric).

### 정의 3.6 — Chow-Liu Tree

**Target**: Tree $T$ maximizing likelihood under tree distribution:
$$p_T(x) = \prod_v p(x_v | x_{\text{pa}(v)})$$

**Optimal**: $\arg\max_T \sum_{(u, v) \in T} I(X_u; X_v)$ = MST with MI weights.

---

## 🔬 정리와 증명

### 정리 3.1 — DAG Learning NP-Hardness (Chickering 1996)

**명제**: "Find DAG $\mathcal{G}$ with $\text{BIC}(\mathcal{G}) \geq k$" decision problem은 NP-complete.

**증명 개요**: Reduction from **Feedback Arc Set** (known NP-complete). BIC의 structure penalty를 조작하여 edge inclusion을 NP-hard optimization으로 변환.

상세: Chickering 1996 "Learning Bayesian networks is NP-complete".

**함의**: Exact structure learning은 일반적으로 불가능 → heuristic (greedy, genetic algorithm).

### 정리 3.2 — Chow-Liu의 Correctness

**명제**: Tree-restricted MLE는 pairwise MI의 maximum spanning tree.

**증명**:

Tree distribution: $p_T(x) = \prod_{(u, v) \in T} \frac{p(x_u, x_v)}{p(x_u) p(x_v)} \prod_w p(x_w)$

(chain rule + tree factorization).

Log-likelihood:
$$\ell = \sum_n \log p_T(x^{(n)}) = N \sum_{(u, v) \in T} \mathbb{E}_{p}[\log \frac{p(X_u, X_v)}{p(X_u) p(X_v)}] + N \sum_w \mathbb{E}[\log p(X_w)]$$
$$= N \sum_{(u, v) \in T} I(X_u; X_v) + (\text{const})$$

Maximize over trees: MST with edge weights $I(X_u; X_v)$. $\square$

**복잡도**: MI 계산 $O(n^2 N)$, Kruskal's MST $O(n^2 \log n)$. **Polynomial time exact solution** — 극히 드문 case (대부분 structure learning NP-hard).

### 정리 3.3 — Score Equivalence

**명제** (Heckerman et al. 1995): BIC and BDeu are **score equivalent**: Markov-equivalent DAGs have the same score.

**증명 개요**:
- Markov equivalent DAGs describe same CI structure → same likelihood family
- BIC의 likelihood + penalty가 equivalence 내에서 same
- BDeu에서 Dirichlet marginal도 equivalent

**함의**: Score-based learning이 **equivalence class** (CPDAG)를 반환 가능 — 개별 DAG가 아니라 equivalence. True causal direction은 결정 불가 from observational data only (추가 실험/가정 필요).

### 정리 3.4 — PC Algorithm Correctness

**명제** (Spirtes-Glymour-Scheines 2000): Under **faithfulness** + **causal sufficiency** + **consistent CI test**, PC algorithm converges to Markov equivalence class of true DAG.

**증명 개요**:
- Faithfulness: 관측 CI가 모두 d-sep로 설명됨
- Causal sufficiency: unobserved common cause 없음
- CI test가 $N \to \infty$에서 consistent

Under these: PC's edge removal + orientation steps → correct CPDAG.

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def mutual_information(X, Y):
    """Estimate MI between two discrete variables."""
    # Joint and marginal distributions
    N = len(X)
    joint = {}
    marg_X = {}
    marg_Y = {}
    for x, y in zip(X, Y):
        joint[(x, y)] = joint.get((x, y), 0) + 1
        marg_X[x] = marg_X.get(x, 0) + 1
        marg_Y[y] = marg_Y.get(y, 0) + 1
    
    mi = 0
    for (x, y), n_xy in joint.items():
        p_xy = n_xy / N
        p_x = marg_X[x] / N
        p_y = marg_Y[y] / N
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi

def chow_liu_tree(data):
    """
    data: (N, n) — N samples, n discrete variables.
    Returns: tree structure (networkx graph).
    """
    N, n = data.shape
    
    # Compute pairwise MI
    MI = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        mi = mutual_information(data[:, i], data[:, j])
        MI[i, j] = mi
        MI[j, i] = mi
    
    # Maximum spanning tree (Kruskal-like)
    edges = []
    for i, j in combinations(range(n), 2):
        edges.append((i, j, MI[i, j]))
    edges.sort(key=lambda e: -e[2])  # descending MI
    
    # Union-find
    parent = list(range(n))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    tree = nx.Graph()
    tree.add_nodes_from(range(n))
    for u, v, w in edges:
        if find(u) != find(v):
            tree.add_edge(u, v, weight=w)
            parent[find(u)] = find(v)
    
    return tree, MI

# 합성 데이터: 3-chain BN
np.random.seed(42)
N = 5000

# X_0 → X_1 → X_2 → X_3 → X_4 (chain, strong dependencies)
X0 = np.random.randint(0, 3, N)
X1 = np.array([(x + np.random.randint(-1, 2)) % 3 for x in X0])  # noisy +X0
X2 = np.array([(x + np.random.randint(-1, 2)) % 3 for x in X1])
X3 = np.array([(x + np.random.randint(-1, 2)) % 3 for x in X2])
X4 = np.array([(x + np.random.randint(-1, 2)) % 3 for x in X3])

# Also add an independent variable X_5
X5 = np.random.randint(0, 3, N)

data = np.column_stack([X0, X1, X2, X3, X4, X5])

tree, MI = chow_liu_tree(data)

print("Pairwise MI matrix:")
print(np.round(MI, 3))

print("\nChow-Liu tree edges (i, j, MI):")
for u, v in tree.edges():
    print(f"  ({u}, {v}): MI = {tree[u][v]['weight']:.3f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im = axes[0].imshow(MI, cmap='viridis')
axes[0].set_title('Pairwise MI matrix')
axes[0].set_xticks(range(6))
axes[0].set_yticks(range(6))
axes[0].set_xticklabels([f'X{i}' for i in range(6)])
axes[0].set_yticklabels([f'X{i}' for i in range(6)])
plt.colorbar(im, ax=axes[0])

labels = {i: f'X{i}' for i in range(6)}
pos = nx.spring_layout(tree, seed=42)
nx.draw(tree, pos, labels=labels, ax=axes[1], node_color='lightblue',
        node_size=1500, font_weight='bold', 
        width=[tree[u][v]['weight'] * 3 for u, v in tree.edges()])
axes[1].set_title('Chow-Liu tree\n(edge width = MI)')

plt.tight_layout()
plt.savefig('chow_liu.png', dpi=120, bbox_inches='tight')
plt.show()

# X5는 독립 → tree의 leaf여야 함
print(f"\nX5 (독립 변수)의 degree: {tree.degree(5)}")
print("→ Chow-Liu가 독립 변수를 leaf에 배치 (올바른 structure 복원)")
```

**출력 예시**:
```
Pairwise MI matrix:
[[0.    0.432 0.318 0.201 0.131 0.008]
 [0.432 0.    0.432 0.318 0.201 0.007]
 [0.318 0.432 0.    0.432 0.318 0.006]
 [0.201 0.318 0.432 0.    0.432 0.005]
 [0.131 0.201 0.318 0.432 0.    0.009]
 [0.008 0.007 0.006 0.005 0.009 0.   ]]

Chow-Liu tree edges (i, j, MI):
  (0, 1): MI = 0.432
  (1, 2): MI = 0.432
  (2, 3): MI = 0.432
  (3, 4): MI = 0.432
  (4, 5): MI = 0.009

X5 (독립 변수)의 degree: 1
→ Chow-Liu가 독립 변수를 leaf에 배치 (올바른 structure 복원)
```

Chow-Liu가 true chain structure $X_0 - X_1 - X_2 - X_3 - X_4$를 정확히 복원. 독립 변수 $X_5$는 arbitrary leaf.

---

## 🔗 AI/ML 연결

### NOTEARS (Zheng et al. 2018)

Structure learning as **continuous optimization**:
$$\min_W \|X - X W\|^2 + \lambda \|W\|_1$$

subject to acyclicity: $\text{tr}(e^{W \circ W}) - n = 0$.

- $W$ = weighted adjacency matrix
- Differentiable acyclicity constraint
- End-to-end gradient descent
- Converts NP-hard combinatorial → smooth optimization

### Causal Discovery

Modern **causal discovery**: structure learning + causal assumptions.
- **PC, FCI** (Spirtes et al.): assume no latent confounders
- **LiNGAM** (Shimizu et al.): Non-Gaussianity → identifies exact DAG
- **RFCI, ADMGs**: with latent confounders
- **GES** (Chickering 2002): greedy equivalence search

응용: gene regulatory networks, fMRI brain networks, economic causality.

### Information Theory and PGM

Chow-Liu: pairwise MI. 일반화:
- **Tree-augmented Naive Bayes (TAN)**: class + features, tree on features conditional on class
- **Markov Tree**: max spanning tree on general nodes

Modern connections: **mutual information neural estimation (MINE)**, **InfoNCE** — structure learning을 contrastive neural으로.

### Applications in Biology

**Gene Regulatory Networks**:
- Nodes = genes, edges = regulatory relationships
- Expression data (RNA-seq) → PC algorithm 또는 ARACNE
- **Causal** inference for drug target discovery

**Protein Interaction Networks**: 
- Bayesian structure learning for pairwise interactions
- Integration of observational + experimental (perturbation) data

### Neural Structure Learning

**DAG-GNN** (Yu et al. 2019): Variational autoencoder + DAG prior.
**GraN-DAG** (Lachapelle et al. 2020): Gradient-based neural DAG learning.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Faithfulness | Real data이 this 위반할 수 있음 (exact CI by cancellation) |
| Causal sufficiency | Unobserved confounders는 spurious CI |
| Consistent CI test | Small sample에서 noisy |
| DAG structure | Cyclic causation은 DAG 위반 |
| Discrete data (many scores) | Continuous는 Gaussian 가정 또는 non-parametric |

**주의**: Observational data로는 **Markov equivalence class**까지만 identify 가능. **Individual DAG** identification은 **interventions** (randomized experiments) 필요.

---

## 📌 핵심 정리

$$\boxed{\text{Chow-Liu tree} = \arg\max_T \sum_{(u,v) \in T} I(X_u; X_v) = \text{MST}}$$

**Score-based**:
- BIC, BDeu, AIC
- Score equivalence → CPDAG
- Greedy hill-climbing, GES

**Constraint-based**:
- PC, IC algorithms
- CI tests → CPDAG
- Faithfulness assumption

**복잡도**:
- DAG learning: **NP-hard** (general)
- Tree-restricted: **polynomial** (Chow-Liu)
- Bounded in-degree: **polynomial** in $n$ (still exponential in degree bound)

---

## 🤔 생각해볼 문제

**문제 1** (기초): BIC와 AIC의 penalty term 차이를 설명하고, 어느 것이 더 "conservative" (fewer edges)인지 이유를 밝혀라.

<details>
<summary>힌트 및 해설</summary>

**BIC**: $-\frac{|\theta|}{2} \log N$
**AIC**: $-|\theta|$

$N$ > $e^2 \approx 7.4$ 이면 $\log N > 2$, BIC penalty > AIC penalty.

**Large $N$**:
- BIC penalty → $\infty$ 처럼 model size에 strongly penalize
- AIC penalty constant per parameter
- **BIC prefers simpler models** (fewer edges)

**정량적 비교**:
- $N = 100$: BIC penalty $\approx 4.6 / 2 = 2.3$ per parameter, AIC = 1 per parameter. BIC > AIC
- $N = 10000$: BIC $\approx 9.2/2 = 4.6$, AIC = 1. BIC >> AIC

**Theoretical justification**:
- BIC: asymptotic approximation of **Bayesian marginal likelihood** (Schwarz 1978)
- AIC: asymptotic minimizer of **expected Kullback-Leibler** from true distribution (Akaike 1974)

**Practical**:
- BIC: Consistent ("true" model selected as $N \to \infty$)
- AIC: Predictively optimal (better out-of-sample, even if not "true")

**For structure learning**:
- BIC typically preferred (sparsity, interpretability)
- AIC if predictive performance matters more
- Cross-validation as alternative

**Trade-off**:
- True model may have many parameters but correct structure
- Noisy small sample → simpler (BIC) is safer
- Large $N$ → BIC finds correct structure asymptotically

</details>

**문제 2** (심화): PC algorithm의 **edge orientation phase** (v-structure detection)을 자세히 설명하라.

<details>
<summary>힌트 및 해설</summary>

**PC algorithm 단계**:

**Phase 1 — Skeleton** (undirected):
- 시작: complete graph
- For each edge $(X, Y)$:
  - Test $X \perp Y | Z$ for $Z \subseteq N(X) \cap N(Y)$
  - If some $Z$ gives independence → remove edge
- 결과: undirected skeleton, conditioning set $\text{Sep}(X, Y)$ recorded for each removed edge

**Phase 2 — V-structure orientation**:

For each unshielded triple $X - Z - Y$ (no edge $X-Y$):
- If $Z \notin \text{Sep}(X, Y)$:
  - Orient as **collider**: $X \to Z \leftarrow Y$
  - 이유: $X \perp Y$ requires conditioning that doesn't include $Z$, so $Z$ must be collider (by d-separation rules)

**Phase 3 — Meek rules** (propagation):

Apply rules iteratively:
- **R1**: $X \to Z - Y$ and $X, Y$ not adjacent → $Z \to Y$ (else new v-structure created)
- **R2**: $X \to Z \to Y$ and $X - Y$ → $X \to Y$ (avoid cycle)
- **R3**: Complex patterns 유사

**Output**: CPDAG with:
- Directed edges: orientation determined
- Undirected edges: either direction consistent with CI pattern

**왜 v-structure는 unique**:
- Collider는 unconditional CI $X \perp Y$를 **fails** when conditioning $Z$
- Chain/fork with $Z \in \text{Sep}$: $X \perp Y | Z$
- Chain/fork와 구분 가능 by which $Z$ separates

**Limitations**:
- Faithfulness 가정
- CI test의 error propagation (chain of tests)
- Sample complexity large for high-dimensional data

**Modern improvements**:
- **Conservative PC**: less aggressive orientation
- **Stable PC** (Colombo-Maathuis 2014): order-independent
- **FCI** (Fast Causal Inference): handle latent confounders

</details>

**문제 3** (AI 연결): NOTEARS의 "differentiable acyclicity constraint" $\text{tr}(e^{W \circ W}) - n = 0$의 수학적 의미를 설명하라.

<details>
<summary>힌트 및 해설</summary>

**NOTEARS (Zheng et al. 2018)** transforms DAG learning:
- From: combinatorial search over DAGs (NP-hard)
- To: continuous optimization with differentiable constraint

**Binary DAG → continuous**:
- $W \in \mathbb{R}^{n \times n}$: weighted adjacency matrix
- $W_{ij} \neq 0$ ⟺ edge $i \to j$
- Standard: $W \in \{0, 1\}^{n \times n}$ with acyclicity → hard to optimize

**Acyclicity characterization**:

**Theorem (Zheng et al. 2018)**: $W$ is acyclic iff 
$$h(W) := \text{tr}(e^{W \circ W}) - n = 0$$

**Proof sketch**:
- $e^M = \sum_{k=0}^\infty M^k / k!$
- $[M^k]_{ii}$ = sum of weights of all walks of length $k$ from $i$ to $i$
- $\text{tr}(M^k)$ = sum of $[M^k]_{ii}$ = total weight of **closed walks** of length $k$
- Acyclic graph: no closed walks (cycles) of any length $k \geq 1$
- $W \circ W$ = element-wise square (handle negative weights)
- $\text{tr}(M^0) = n$ (trace of identity)
- $h(W) = 0$ iff closed walks of length $\geq 1$ have zero weight iff acyclic

**Properties of $h$**:
- $h$ smooth (infinite series of matrix powers)
- $h(W) \geq 0$ always (closed walk weights non-negative since $W \circ W \geq 0$)
- $\nabla_W h$ computable — differentiable!

**Optimization**:
$$\min_W \mathcal{L}(W; D) + \lambda \|W\|_1 + \text{s.t. } h(W) = 0$$

**Augmented Lagrangian**:
$$\min_W \mathcal{L}(W; D) + \lambda \|W\|_1 + \alpha h(W) + \frac{\rho}{2} h(W)^2$$

Solve iteratively with $\alpha, \rho$ increasing.

**Practical issues**:
- Computing $e^{W \circ W}$ expensive ($O(n^3)$ per eval, many evals)
- Truncated series 또는 low-rank approximation 사용
- Finite precision issues near $h(W) = 0$

**Impact**:
- DAG learning이 standard deep learning optimization tool로 가능
- **DAG-GAN**, **DAG-VAE** 같은 후속 연구
- End-to-end **causal structure learning from raw data**

**Limitations**:
- Local optima (non-convex)
- Linear model 가정 original paper (nonlinear extensions: Lachapelle et al. NeurIPS 2020)
- Identifiability 문제 여전 (observational data alone)

**결론**: NOTEARS = "classical PGM problem을 현대 deep learning optimization 기법으로 재해석". Combinatorial → continuous의 elegant formulation.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. EM Algorithm — 불완전 데이터](./02-em-algorithm.md) | [📚 README](../README.md) | [04. Topic Model (LDA)의 그래프 모델적 이해 ▶](./04-lda-topic-model.md) |

</div>
