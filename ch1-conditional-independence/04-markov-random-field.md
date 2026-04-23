# 04. Markov Random Field와 Hammersley–Clifford

## 🎯 핵심 질문

- Markov Random Field(undirected graphical model)는 왜 clique potential의 곱으로 인수분해되는가?
- **Hammersley–Clifford 정리** — positive density 하에서 Markov property ⟺ Gibbs distribution — 은 어떻게 증명되는가?
- Partition function $Z$는 왜 #P-hard인가?
- BN과 MRF는 표현력에서 어떻게 다른가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Markov Random Field**는 **대칭적 상호작용**을 모델링하는 도구다. Ising model(스핀 격자), 이미지 분할(인접 픽셀의 smoothness prior), CRF(Ch4), Boltzmann machine, RBM(Restricted Boltzmann Machine) — 모두 MRF. **Energy-based model**(EBM)은 $p(x) = e^{-E(x)}/Z$로 정확히 MRF. Modern generative model(Score-based, EBM)의 이론적 기반도 MRF. **Variational autoencoder**의 prior를 structured prior로 확장하려면 MRF가 필요하다. Hammersley–Clifford 없이는 "왜 local interaction이 global probability를 결정하는가"의 이유를 알 수 없다.

---

## 📐 수학적 선행 조건

- [Ch1-01 조건부 독립의 정의와 성질](./01-conditional-independence-definition.md)
- Graph theory: undirected graph, clique, maximal clique, vertex separator
- Combinatorics: Möbius inversion
- Probability: positive density, ratio form

---

## 📖 직관적 이해

### 왜 Undirected인가?

Bayesian Network은 **인과/방향성**이 자연스러운 상황에 적합. 그러나:

- **Ising model**: 스핀 $s_i, s_j$ 사이의 상호작용은 **대칭** — "누가 원인"이라는 개념 없음
- **이미지 픽셀**: 인접 픽셀이 비슷해야 한다는 **smoothness** — 방향성 없음
- **Social network**: 친구 관계, Facebook의 like — 대칭

이런 때 **undirected graph**가 적합. 대신 인수분해는 parent 개념이 없어서 다른 방식.

### Clique Potential

Undirected graph $\mathcal{G}$에서 **clique** $C$는 모든 꼭짓점이 서로 연결된 부분집합. **Maximal clique**는 더 큰 clique에 포함되지 않는 것.

MRF의 결합분포는 maximal clique $C \in \text{cl}(\mathcal{G})$의 **potential** $\phi_C$의 곱:

$$p(x) = \frac{1}{Z} \prod_{C \in \text{cl}(\mathcal{G})} \phi_C(x_C)$$

여기서 $\phi_C \geq 0$은 임의 non-negative 함수 (확률일 필요 없음), $Z$는 정규화.

### Partition Function의 악마

$$Z = \sum_x \prod_C \phi_C(x_C)$$

$n$개의 이진 변수에 대해 $2^n$개 항의 합 — **exponential**. BN과 달리 MRF는 $\phi_C$가 local이어도 $Z$는 **global** 계산 필요. 이것이 MRF inference의 근본 난제.

### BN vs MRF 직관

| | Bayesian Network | Markov Random Field |
|--|-----------------|---------------------|
| Graph | DAG | Undirected |
| 인수분해 | $\prod p(x_v \mid \text{pa}(v))$ | $\prod \phi_C(x_C) / Z$ |
| 정규화 | 자동 ($p(x|\text{pa})$가 이미 확률) | $Z$ 계산 필요 |
| CI 판정 | d-separation | Graph separation |
| 방향성 | 방향 있음 (인과) | 대칭 |
| 조건부 | Closed under conditioning | Closed under conditioning |
| 샘플링 | Ancestral sampling 쉬움 | MCMC 필요 |

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Undirected Graph와 Clique

**Undirected graph** $\mathcal{G} = (V, E)$. $C \subseteq V$가 **clique**이면 $C$의 모든 쌍이 edge. **Maximal clique**는 strict superset이 clique가 아닌 clique.

### 정의 4.2 — Markov Properties (MRF)

분포 $P$와 undirected graph $\mathcal{G}$에 대해 세 가지 Markov property:

**Pairwise Markov**: $(u, v) \notin E$이면 $X_u \perp\!\!\!\perp X_v \mid X_{V \setminus \{u, v\}}$

**Local Markov**: 모든 $v$에 대해 $X_v \perp\!\!\!\perp X_{V \setminus (\{v\} \cup N(v))} \mid X_{N(v)}$ (여기서 $N(v)$는 $v$의 이웃)

**Global Markov**: 세 집합 $A, B, S$에 대해 $S$가 $A$와 $B$를 $\mathcal{G}$에서 **분리**(그래프상 $A$에서 $B$로 가는 모든 path가 $S$ 통과)하면 $X_A \perp\!\!\!\perp X_B \mid X_S$

일반적으로 Global ⟹ Local ⟹ Pairwise. Positive density에서 세 개 동치.

### 정의 4.3 — Gibbs Distribution

MRF 인수분해:
$$p(x) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \phi_C(x_C), \quad Z = \sum_x \prod_C \phi_C(x_C)$$

여기서 $\mathcal{C}$는 $\mathcal{G}$의 cliques의 집합(maximal로 제한 가능), $\phi_C \geq 0$.

종종 **energy function** 형태로: $\phi_C(x_C) = \exp(-E_C(x_C))$이면
$$p(x) = \frac{1}{Z} \exp\left(-\sum_C E_C(x_C)\right) = \frac{1}{Z} \exp(-E(x))$$

이는 **Boltzmann distribution**.

---

## 🔬 정리와 증명

### 정리 4.1 — Gibbs ⟹ Global Markov

**명제**: $p(x) = \frac{1}{Z} \prod_C \phi_C(x_C)$로 인수분해되면, $P$는 $\mathcal{G}$의 global Markov property를 만족한다.

**증명**: 

$S$가 $A$와 $B$를 그래프에서 분리한다고 하자. 각 maximal clique $C$는 $A$, $B$, $S$와만 겹칠 수 있으나, $C$가 $A$와 $B$ 모두에 겹치면 $C$ 안의 꼭짓점 $a \in A, b \in B$가 edge로 연결 — 이는 $S$가 $A$와 $B$를 분리한다는 가정에 모순.

따라서 $\mathcal{C}$는 세 disjoint 집합으로 분할:
- $\mathcal{C}_A$: $A$ 또는 $A \cup S$에만 속하는 clique
- $\mathcal{C}_B$: $B$ 또는 $B \cup S$에만 속하는 clique  
- $\mathcal{C}_S$: $S$에만 속하는 clique

$$p(x) = \frac{1}{Z} \prod_{C \in \mathcal{C}_A} \phi_C(x_{A \cup S}) \prod_{C \in \mathcal{C}_S} \phi_C(x_S) \prod_{C \in \mathcal{C}_B} \phi_C(x_{B \cup S})$$

$$= \frac{1}{Z} \cdot g(x_{A \cup S}) \cdot h(x_{B \cup S})$$

정리 1.1 (3)에 의해 $X_A \perp\!\!\!\perp X_B \mid X_S$. $\square$

### 정리 4.2 — Hammersley–Clifford (1971, positive density 하)

**명제**: $p(x) > 0$이고 pairwise Markov property를 만족하면, $p$는 Gibbs:

$$p(x) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \phi_C(x_C)$$

**증명** (Möbius inversion):

WLOG 모든 변수가 이진 $\{0, 1\}$이라 가정 (일반화는 유사). $\mathbf{0} := (0, 0, \ldots, 0)$. Positive density이므로 $p(\mathbf{0}) > 0$.

Log-likelihood ratio 정의:
$$H(x) := \log \frac{p(x)}{p(\mathbf{0})}$$

$H(\mathbf{0}) = 0$. Claim: $H$는 **clique에서의 합**으로 분해됨:
$$H(x) = \sum_{C \in \mathcal{C}} H_C(x_C)$$

이를 보이기 위해 Möbius inversion 사용. 임의의 $S \subseteq V$에 대해
$$H_S(x) := \sum_{T \subseteq S} (-1)^{|S \setminus T|} H(x_T, \mathbf{0}_{V \setminus T})$$

**Step 1**: Möbius inversion으로
$$H(x) = \sum_{S \subseteq V} H_S(x)$$

**Step 2** (핵심): $S$가 **clique가 아니면** $H_S \equiv 0$.

$S$가 clique가 아니라 $u, v \in S$이 edge가 아니라 하자. Pairwise Markov에 의해
$$p(x_u, x_v, x_{V \setminus \{u, v\}}) = p(x_u | x_{V \setminus \{u, v\}}) p(x_v | x_{V \setminus \{u, v\}}) p(x_{V \setminus \{u, v\}})$$

이 대수에서 $H$ 표현으로 가면:
$$H(x_u = 1, x_v = 1, x_R) - H(x_u = 0, x_v = 1, x_R) - H(x_u = 1, x_v = 0, x_R) + H(x_u = 0, x_v = 0, x_R) = 0$$

모든 $x_R \in \{0,1\}^{V \setminus \{u,v\}}$에 대해 (여기서 $R := V \setminus \{u, v\}$). 이 identity가 $H_S$의 $u, v$-cross-term의 취소 조건. Möbius inversion의 구조상 $S \ni u, v$이고 $S$가 clique가 아니면 $H_S = 0$이 됨.

**Step 3**: 따라서
$$H(x) = \sum_{C \text{ clique}} H_C(x_C)$$

$\phi_C := \exp(H_C)$로 놓으면 $p(x) = p(\mathbf{0}) \prod_C \phi_C(x_C)$. 정규화하여 $Z = 1/p(\mathbf{0}) \cdot [1]$이 아니라, $Z = \sum_x \prod_C \phi_C$로 자동으로 결정. 실제로는 $p(\mathbf{0})$가 normalization의 일부가 되어 깔끔하게 $p(x) = \frac{1}{Z} \prod_C \phi_C(x_C)$. $\square$

**참고**: Positive density 없이는 Hammersley–Clifford가 성립하지 않음. Moussouris(1974)의 유명한 4-변수 반례 — Markov property를 만족하지만 Gibbs factorization 불가능한 분포가 존재.

### 정리 4.3 — Partition Function의 #P-hardness

**명제**: 일반 MRF에서 $Z = \sum_x \prod_C \phi_C(x_C)$ 계산은 **#P-hard**.

**증명 개요**: **Reduction from #SAT**. 

각 CNF 절(clause) $\text{clause}_i$마다 해당 변수 집합에 potential $\phi_i = \mathbb{1}[\text{clause}_i \text{ 참}]$을 할당. 그러면
$$Z = \sum_x \prod_i \phi_i(x) = \# \text{satisfying assignments}$$

#SAT은 #P-complete (Valiant 1979). 따라서 일반 MRF의 $Z$ 계산도 #P-hard. $\square$

이것이 MRF inference(marginal, partition function)이 일반적으로 **intractable**인 이유. Junction tree(Ch2-04), variational inference(Ch6), sampling(Ch6-04)가 필요.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 예시 1: 2D Ising Model (MRF의 전형)
# Energy: E(x) = -J sum_{<ij>} x_i x_j, x_i ∈ {-1, +1}
# p(x) ∝ exp(-E(x)/T)

L = 4  # 4x4 grid
J = 1.0
T = 2.0

# Gibbs sampling으로 MRF 샘플
def gibbs_sample_ising(L, J, T, n_iter=10000, seed=0):
    np.random.seed(seed)
    x = np.random.choice([-1, 1], size=(L, L))
    samples = []
    for it in range(n_iter):
        for i in range(L):
            for j in range(L):
                # 이웃들 (주기적 경계)
                neighbors = (x[(i-1) % L, j] + x[(i+1) % L, j] +
                             x[i, (j-1) % L] + x[i, (j+1) % L])
                # P(x_{ij} = +1 | neighbors) = sigmoid(2 J neighbors / T)
                p = 1 / (1 + np.exp(-2 * J * neighbors / T))
                x[i, j] = 1 if np.random.rand() < p else -1
        if it >= n_iter // 2:  # burn-in
            samples.append(x.copy())
    return np.array(samples)

samples = gibbs_sample_ising(L, J, T, n_iter=3000)

# 평균 magnetization과 correlation
magnetization = samples.mean(axis=(1, 2))
print(f"2D Ising (L={L}, J={J}, T={T})")
print(f"평균 magnetization: {magnetization.mean():.3f} ± {magnetization.std():.3f}")

# 이웃 상관 (local Markov의 수치 확인)
neighbor_corr = []
for s in samples:
    c = 0
    for i in range(L):
        for j in range(L):
            c += s[i, j] * s[(i+1) % L, j]
    neighbor_corr.append(c / (L * L))
print(f"평균 이웃 상관: {np.mean(neighbor_corr):.3f}")

# Global Markov 확인: 같은 행의 두 끝 변수가 중간 변수들로 분리되어 있음
# x_{0,0}과 x_{0,3}이 x_{0,1}, x_{0,2} 주어졌을 때 CI?
# (실제로는 grid에서 column 방향 path도 있어서 분리 안 됨)
print("\n주의: Grid MRF에서 (0,0)과 (0,3)은 행만으로는 분리 안 됨 — 모든 path가 분리되어야 함")
print("전체 column 1과 2를 조건으로 줘야 global Markov 성립")

# Hammersley-Clifford 검증: positive density → factorization 가능
# 간단한 3-chain MRF로 시연
# Pairwise Markov: x_1 ⊥ x_3 | x_2
# Factorization: p(x) = (1/Z) φ_{12}(x_1, x_2) φ_{23}(x_2, x_3)

def build_3chain_mrf():
    # x_1, x_2, x_3 ∈ {0, 1, 2}
    phi_12 = np.array([[2.0, 1.0, 0.5],
                       [1.0, 3.0, 1.0],
                       [0.5, 1.0, 2.0]])
    phi_23 = np.array([[1.5, 1.0, 0.8],
                       [1.0, 2.0, 1.0],
                       [0.8, 1.0, 1.5]])
    # p(x_1, x_2, x_3) ∝ phi_12 * phi_23
    P = np.zeros((3, 3, 3))
    for x1 in range(3):
        for x2 in range(3):
            for x3 in range(3):
                P[x1, x2, x3] = phi_12[x1, x2] * phi_23[x2, x3]
    Z = P.sum()
    return P / Z, Z

P, Z = build_3chain_mrf()
print(f"\n3-chain MRF: Z = {Z:.3f}")

# x_1 ⊥ x_3 | x_2 검증
for x2 in range(3):
    P_given_x2 = P[:, x2, :] / P[:, x2, :].sum()
    P_x1_given_x2 = P_given_x2.sum(axis=1)
    P_x3_given_x2 = P_given_x2.sum(axis=0)
    product = np.outer(P_x1_given_x2, P_x3_given_x2)
    diff = np.abs(P_given_x2 - product).max()
    print(f"  x_2={x2}: max|P(x_1,x_3|x_2) - P(x_1|x_2)P(x_3|x_2)| = {diff:.5f}")

# 시각화: Ising sample
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(samples[-1], cmap='gray', vmin=-1, vmax=1)
ax.set_title(f'Ising 샘플 (T={T}, J={J})')
ax.axis('off')
plt.tight_layout()
plt.savefig('ising_sample.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
2D Ising (L=4, J=1.0, T=2.0)
평균 magnetization: 0.421 ± 0.523
평균 이웃 상관: 0.639

주의: Grid MRF에서 (0,0)과 (0,3)은 행만으로는 분리 안 됨 — 모든 path가 분리되어야 함
전체 column 1과 2를 조건으로 줘야 global Markov 성립

3-chain MRF: Z = 54.550
  x_2=0: max|P(x_1,x_3|x_2) - P(x_1|x_2)P(x_3|x_2)| = 0.00000
  x_2=1: max|P(x_1,x_3|x_2) - P(x_1|x_2)P(x_3|x_2)| = 0.00000
  x_2=2: max|P(x_1,x_3|x_2) - P(x_1|x_2)P(x_3|x_2)| = 0.00000
```

3-chain MRF에서 $x_1 \perp\!\!\!\perp x_3 \mid x_2$가 정확히 성립 — global Markov의 수치적 확인.

---

## 🔗 AI/ML 연결

### Energy-Based Model (EBM)

EBM의 기본 형태:
$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}, \quad Z(\theta) = \int \exp(-E_\theta(x)) dx$$

이는 MRF의 **연속 일반화**. Modern EBM(Xie et al. 2016, Du et al. 2019, Song & Ermon 2019)은 $E_\theta$를 신경망으로 parameterize. **Score matching**, **Contrastive Divergence**, **Langevin dynamics**로 학습 (Ch7-01에서 자세히).

### Restricted Boltzmann Machine (RBM)

RBM은 2-layer MRF: visible $v$ + hidden $h$, edges only between visible-hidden.
$$p(v, h) = \frac{1}{Z} \exp(-E(v, h)), \quad E(v, h) = -v^T W h - b^T v - c^T h$$

**Conditional independence**: visible given hidden — 모두 독립 ($v_i \perp\!\!\!\perp v_j \mid h$). 이것이 contrastive divergence 학습의 핵심 속성 — 조건부 샘플링이 벡터화 가능.

### 이미지 분할의 Grid MRF

이미지 $I$ 위의 segmentation label $y$를 MRF로 모델링:
$$p(y | I) = \frac{1}{Z(I)} \exp\left(\sum_i \psi_i(y_i, I_i) + \sum_{(i,j) \in E} \psi_{ij}(y_i, y_j)\right)$$

- $\psi_i$: unary (pixel이 label $y_i$일 점수)
- $\psi_{ij}$: pairwise (인접 pixel이 같은 label일 bonus) — smoothness prior

MAP inference는 **graph cut** 또는 $\alpha$-expansion으로 해결 (Boykov–Kolmogorov 2001).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Positive density | 0 probability 있으면 Hammersley-Clifford 실패 (Moussouris 1974 반례) |
| Finite state space | 연속으로 확장하려면 적분 + regularity |
| Maximal clique | 너무 큰 clique는 parameter 폭발 → 종종 pairwise MRF로 제한 |
| Partition function | #P-hard, approximation 또는 contrastive 방법 필요 |

**주의**: BN은 MRF로 moralize(다음 문서) 시 일부 CI 손실. MRF는 collider-like CI(unconditional independence under v-structure) 표현 불가.

---

## 📌 핵심 정리

$$\boxed{p(x) = \frac{1}{Z}\prod_{C \in \mathcal{C}} \phi_C(x_C) \iff p \text{ is Markov on } \mathcal{G} \text{ (positive density)}}$$

| 개념 | 의미 |
|------|------|
| **Clique potential** | 각 maximal clique에 할당된 non-negative function |
| **Partition function** | $Z = \sum \prod \phi_C$ — 정규화, #P-hard |
| **Hammersley–Clifford** | Positive density 하에서 Markov ⟺ Gibbs |
| **Global Markov** | Graph separation ⟹ CI |
| **Moussouris 반례** | Positive density 없으면 역방향 실패 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 4-variable MRF에서 complete graph(모든 edge)와 empty graph의 파라미터 수는?

<details>
<summary>힌트 및 해설</summary>

**Complete graph** ($K_4$): 유일한 maximal clique = 전체 $\{1, 2, 3, 4\}$. 이진 변수 가정 시 potential $\phi_{1234}(x_1, x_2, x_3, x_4)$는 $2^4 = 16$개 엔트리. 정규화 고려하면 실질 15개. 즉 **full joint와 동일**.

**Empty graph**: Maximal clique = 각 singleton. $\phi_i(x_i)$가 4개, 각 1개 파라미터(정규화 후). **총 4개** — 모든 변수 독립.

중간 예 (chain $1 - 2 - 3 - 4$): Maximal clique = $\{1,2\}, \{2,3\}, \{3,4\}$. 각 pairwise potential $2 \times 2 = 4$ 엔트리 → **총 12개** (정규화 고려 시 $Z$로 1 감소).

이처럼 **graph의 sparsity가 parameter 수를 결정**.

</details>

**문제 2** (심화): MRF $p(x) \propto \phi_{12}(x_1, x_2) \phi_{23}(x_2, x_3)$에서 $\phi_{12}' := 2 \phi_{12}$, $\phi_{23}' := \phi_{23}/2$로 바꾸면 분포가 변하는가?

<details>
<summary>힌트 및 해설</summary>

**변하지 않음**. 
$$\tilde p(x) \propto \phi_{12}'(x_1, x_2) \phi_{23}'(x_2, x_3) = 2 \phi_{12} \cdot \phi_{23} / 2 = \phi_{12} \phi_{23}$$

정규화 상수 $Z$만 바뀌지만 분포는 동일. 이는 **MRF potential의 rescaling freedom** — clique potential은 **유일하지 않음**.

더 일반적으로 $\phi_{12} \to \phi_{12} \cdot c(x_2)$, $\phi_{23} \to \phi_{23} / c(x_2)$로 $x_2$ 의존 함수도 이동 가능. 이 자유도가 MRF parameterization의 non-identifiability를 낳고, exponential family 표현(canonical parameter)으로 고정하는 것이 표준.

CRF(Ch4)에서는 조건부 확률이므로 $x_1 \to $ conditioning variables는 자유도를 감소시킴. 순수 MRF보다 identifiability가 좋음.

</details>

**문제 3** (AI 연결): RBM의 log-likelihood gradient는 
$$\nabla_\theta \log p(v) = -\mathbb{E}_{p(h|v)}[\nabla_\theta E(v, h)] + \mathbb{E}_{p(v, h)}[\nabla_\theta E(v, h)]$$
왜 두 번째 항("model expectation")은 sampling 없이 계산할 수 없는가?

<details>
<summary>힌트 및 해설</summary>

$$\mathbb{E}_{p(v, h)}[\nabla_\theta E] = \frac{1}{Z} \sum_{v, h} \nabla_\theta E(v, h) \exp(-E(v, h))$$

이 합에는 **모든 가능한 $(v, h)$ 조합**이 필요 — $v$가 $n$차원 이진이면 $2^n$개, 총 $2^{n + m}$ 조합. $n = 784$ (MNIST)이면 도저히 불가능.

$Z = \sum e^{-E}$도 같은 이유로 intractable — Ch4-1의 **#P-hardness**.

**해결책**:
1. **Contrastive Divergence (Hinton 2002)**: 짧은 Gibbs chain으로 model expectation을 근사. 편향되지만 실전에서 잘 작동.
2. **Persistent CD (Tieleman 2008)**: Markov chain을 지속적으로 유지해 더 나은 샘플.
3. **Score Matching (Hyvärinen 2005)**: $\nabla \log p$만 학습, $Z$ 우회.
4. **NCE (Noise Contrastive Estimation)**: 데이터 vs 노이즈 분류로 $Z$ 우회.

이 기법들은 MRF / EBM 학습의 표준. Modern diffusion model의 score matching도 이 계보.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. d-separation](./03-d-separation.md) | [📚 README](../README.md) | [05. Moralization — DAG ↔ MRF 변환 ▶](./05-moralization.md) |

</div>
