# 01. 조건부 독립의 정의와 성질

## 🎯 핵심 질문

- 조건부 독립 $A \perp\!\!\!\perp B \mid C$는 정확히 무엇을 의미하는가?
- 왜 "독립"의 세 가지 동치 서술 — $P(A,B|C) = P(A|C)P(B|C)$, $P(A|B,C) = P(A|C)$, $P(A,B,C) \propto g(A,C)h(B,C)$ — 가 모두 같은가?
- Semi-graphoid 공리(symmetry, decomposition, weak union, contraction)는 왜 "그래프"적 구조의 대수적 기반인가?
- Intersection property는 왜 positive density에서만 성립하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**조건부 독립**은 확률 그래프 모델의 **언어 자체**다. Bayesian Network의 DAG 인수분해, MRF의 clique potential, factor graph의 message passing — 모두 조건부 독립 가정을 그래프 구조로 번역한 것이다. **Naive Bayes**가 feature 간 조건부 독립을 가정하고, **HMM**이 Markov 가정(미래 ⊥ 과거 | 현재)을 쓰며, **Variational Autoencoder**가 posterior $q(z|x)$를 factorize 가능하게 만드는 것 — 이 모든 것이 "조건부 독립"을 **모델링 자유도**로 활용한다. 조건부 독립의 공리를 모르면 **Pearl의 인과 추론**에서 do-calculus의 규칙(intervention과 observation의 구분)을 제대로 이해할 수 없다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$, $\sigma$-대수, 측도론적 조건부 기댓값
- 동시분포와 주변분포: $P(A, B) = \sum_c P(A, B, c)$
- 조건부 확률: $P(A|B) = P(A,B)/P(B)$ when $P(B) > 0$
- 사건의 독립성: $A \perp\!\!\!\perp B \iff P(A \cap B) = P(A)P(B)$

---

## 📖 직관적 이해

### 독립 vs 조건부 독립

두 확률변수가 **독립**이란 관측이 서로에게 정보를 주지 않는 것:

$$X \perp\!\!\!\perp Y \iff P(X, Y) = P(X) P(Y)$$

**조건부 독립**은 한 단계 더 미묘하다 — $Z$를 알고 나면 $X$와 $Y$가 서로 정보를 주지 않는다:

$$X \perp\!\!\!\perp Y \mid Z \iff P(X, Y \mid Z) = P(X \mid Z) P(Y \mid Z)$$

핵심은 **조건을 주는 변수가 정보의 "통로"를 막는다**는 것. 세 가지 대표적 상황:

| 상황 | 일상 비유 |
|------|-----------|
| **Chain** ($X \to Z \to Y$) | "비가 오면 도로가 젖는다. 도로가 젖으면 사고가 난다." $Z$(젖은 도로)를 알면 $X$(비)는 $Y$(사고)에 추가 정보를 주지 않음 |
| **Fork** ($X \leftarrow Z \to Y$) | "기온이 오르면 아이스크림 판매가 늘고, 익사 사고도 늘어난다." $Z$(기온)를 알면 아이스크림↔익사의 상관관계 사라짐 |
| **Collider** ($X \to Z \leftarrow Y$) | "천재거나 부모의 인맥이 있으면 명문대 합격." $Z$(합격)을 **알면** 천재 $\perp\!\!\!\perp$ 인맥이 **깨짐** (explaining away!) |

이 세 패턴이 Ch1-03의 **d-separation**의 모든 것.

### 조건부 독립 ≠ 독립

**조건부 독립은 독립을 함의하지 않는다**. 반대도 마찬가지:

| 예시 | $X \perp\!\!\!\perp Y$? | $X \perp\!\!\!\perp Y \mid Z$? |
|------|----------------------|-----------------------------|
| 기온(Z) → 아이스크림(X), 익사(Y) | ✗ (겉보기 상관) | ✓ (교란 제거) |
| 천재(X), 인맥(Y) → 합격(Z) | ✓ | ✗ (explaining away) |

이 **비단조성**이야말로 조건부 독립을 "자명하지 않은" 개념으로 만든다.

### 세 가지 동치 서술

독립의 정의는 세 가지 방식으로 서술될 수 있다:

1. **대칭형**: $P(X, Y \mid Z) = P(X \mid Z) P(Y \mid Z)$
2. **예측형**: $P(X \mid Y, Z) = P(X \mid Z)$ (i.e. $Y$는 $X$ 예측에 쓸모없음)
3. **인수분해형**: $P(X, Y, Z) = g(X, Z) h(Y, Z)$ for some non-negative functions $g, h$

이 세 가지의 동치성이 정리 1.1의 핵심이다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 조건부 독립 (이산 변수)

확률변수 $X, Y, Z$가 공통 확률공간 위에 정의되어 있고, 모두 이산이라고 하자 (연속은 밀도로 치환). 가능한 값에서 $P(Z = z) > 0$이라 하자.

$X$와 $Y$가 $Z$ 주어졌을 때 **조건부 독립**(conditionally independent given $Z$)이라는 것을

$$X \perp\!\!\!\perp Y \mid Z$$

로 표기하고, 모든 $(x, y, z)$에서 $P(Z = z) > 0$일 때

$$P(X = x, Y = y \mid Z = z) = P(X = x \mid Z = z) \cdot P(Y = y \mid Z = z)$$

가 성립함을 의미한다.

### 정의 1.2 — 측도론적 조건부 독립

**일반적 정의**(연속·혼합 포함): $\sigma$-대수 $\mathcal{G} \subseteq \mathcal{F}$에 대해, $\sigma$-대수 $\mathcal{A}$와 $\mathcal{B}$가 $\mathcal{G}$ 하에서 조건부 독립이라는 것은, 임의의 bounded $\mathcal{A}$-measurable $f$와 $\mathcal{B}$-measurable $g$에 대해

$$\mathbb{E}[fg \mid \mathcal{G}] = \mathbb{E}[f \mid \mathcal{G}] \cdot \mathbb{E}[g \mid \mathcal{G}] \quad \text{a.s.}$$

확률변수 버전: $X \perp\!\!\!\perp Y \mid Z$는 $\sigma(X) \perp\!\!\!\perp \sigma(Y) \mid \sigma(Z)$.

### 정의 1.3 — Independence Model

$\mathcal{U}$를 변수들의 유한 집합이라 하자. $\mathcal{U}$ 위의 **독립 모델**(independence model)은 서로 소인 부분집합의 세 쌍 $(A, B, C)$의 집합 $\mathcal{I}$로, $(A, B, C) \in \mathcal{I}$는 "$A \perp\!\!\!\perp B \mid C$"로 해석된다.

$\mathcal{I}_P := \{(A, B, C) : X_A \perp\!\!\!\perp X_B \mid X_C \text{ under } P\}$를 확률분포 $P$의 **independence model**이라 부른다.

---

## 🔬 정리와 증명

### 정리 1.1 — 조건부 독립의 세 가지 동치

**명제**: 이산 확률변수 $X, Y, Z$에 대해 다음은 서로 동치이다 ($P(z) > 0$인 모든 $z$에서):

1. $P(x, y \mid z) = P(x \mid z) P(y \mid z)$
2. $P(x \mid y, z) = P(x \mid z)$  (단, $P(y, z) > 0$)
3. $P(x, y, z) = g(x, z) h(y, z)$ for some non-negative $g, h$

**증명**:

**(1) ⟹ (2)**: $P(y, z) > 0$이라 하자.
$$P(x \mid y, z) = \frac{P(x, y \mid z)}{P(y \mid z)} = \frac{P(x \mid z) P(y \mid z)}{P(y \mid z)} = P(x \mid z)$$

**(2) ⟹ (1)**:
$$P(x, y \mid z) = P(x \mid y, z) P(y \mid z) = P(x \mid z) P(y \mid z)$$

**(1) ⟹ (3)**: $g(x, z) := P(x \mid z) P(z)$, $h(y, z) := P(y \mid z)$로 놓으면
$$P(x, y, z) = P(x, y \mid z) P(z) = P(x \mid z) P(y \mid z) P(z) = g(x, z) h(y, z)$$

**(3) ⟹ (1)**: $P(x, y, z) = g(x, z) h(y, z)$라 하자. 주변화로
$$P(z) = \sum_{x, y} g(x, z) h(y, z) = G(z) H(z), \quad G(z) := \sum_x g(x, z), \; H(z) := \sum_y h(y, z)$$
$$P(x, z) = \sum_y g(x, z) h(y, z) = g(x, z) H(z)$$
$$P(y, z) = G(z) h(y, z)$$
따라서
$$P(x \mid z) P(y \mid z) = \frac{g(x, z) H(z)}{G(z) H(z)} \cdot \frac{G(z) h(y, z)}{G(z) H(z)} = \frac{g(x, z) h(y, z)}{G(z) H(z)} = \frac{P(x, y, z)}{P(z)} = P(x, y \mid z)$$
$\square$

### 정리 1.2 — Semi-Graphoid 공리

**명제**: 모든 확률분포 $P$의 independence model $\mathcal{I}_P$는 다음 네 공리를 만족한다:

**(S) Symmetry**: $X \perp\!\!\!\perp Y \mid Z \implies Y \perp\!\!\!\perp X \mid Z$

**(D) Decomposition**: $X \perp\!\!\!\perp (Y, W) \mid Z \implies X \perp\!\!\!\perp Y \mid Z$

**(WU) Weak Union**: $X \perp\!\!\!\perp (Y, W) \mid Z \implies X \perp\!\!\!\perp Y \mid (Z, W)$

**(C) Contraction**: $X \perp\!\!\!\perp Y \mid Z$ **and** $X \perp\!\!\!\perp W \mid (Y, Z) \implies X \perp\!\!\!\perp (Y, W) \mid Z$

**증명**:

**Symmetry (S)**: 정의 1.1의 대칭성. $P(x, y | z) = P(x|z)P(y|z) \iff P(y, x | z) = P(y|z)P(x|z)$. $\square$

**Decomposition (D)**: 가정으로 $P(x, y, w | z) = P(x|z) P(y, w | z)$. $w$에 대해 주변화:
$$P(x, y | z) = \sum_w P(x, y, w | z) = \sum_w P(x|z) P(y, w | z) = P(x|z) \sum_w P(y, w | z) = P(x|z) P(y|z)$$
$\square$

**Weak Union (WU)**: 가정 $P(x, y, w | z) = P(x|z) P(y, w | z)$와 Decomposition으로 $P(x, w | z) = P(x|z) P(w|z)$. 이제
$$P(x, y | z, w) = \frac{P(x, y, w | z)}{P(w | z)} = \frac{P(x|z) P(y, w | z)}{P(w | z)} = P(x|z) P(y | z, w)$$
한편
$$P(x | z, w) = \frac{P(x, w | z)}{P(w | z)} = \frac{P(x|z) P(w|z)}{P(w|z)} = P(x|z)$$
따라서 $P(x, y | z, w) = P(x | z, w) P(y | z, w)$, 즉 $X \perp\!\!\!\perp Y \mid (Z, W)$. $\square$

**Contraction (C)**: 두 가정:
- $P(x, y | z) = P(x|z) P(y|z)$
- $P(x, w | y, z) = P(x | y, z) P(w | y, z)$

첫 번째에서 $P(x | y, z) = \frac{P(x, y | z)}{P(y | z)} = P(x | z)$. 두 번째에 대입:
$$P(x, w | y, z) = P(x | z) P(w | y, z)$$

이제
$$P(x, y, w | z) = P(x, w | y, z) P(y | z) = P(x|z) P(w | y, z) P(y | z) = P(x|z) P(y, w | z)$$

즉 $X \perp\!\!\!\perp (Y, W) \mid Z$. $\square$

### 정리 1.3 — Intersection Property는 Positive Density에서만 성립

**명제**(Graphoid 공리; 추가 조건 필요): $P > 0$이면

**(I) Intersection**: $X \perp\!\!\!\perp Y \mid (Z, W)$ **and** $X \perp\!\!\!\perp W \mid (Z, Y) \implies X \perp\!\!\!\perp (Y, W) \mid Z$

그러나 $P$가 positive가 아니면 intersection은 성립하지 않는다.

**반례** (positive density 조건 없이):

$Y = W$ (같은 변수의 두 사본), $X$는 $Y$(=$W$)와 완전 상관인 분포를 생각하자. 구체적으로 $Y, W \in \{0, 1\}$, $X \in \{0, 1\}$, $P(X = Y = W = 0) = 1/2$, $P(X = Y = W = 1) = 1/2$, 나머지 모두 0.

- $X \perp\!\!\!\perp Y \mid (Z, W)$: $W$를 알면 $Y = W$가 결정되므로 자명히 성립
- $X \perp\!\!\!\perp W \mid (Z, Y)$: 같은 논리로 성립
- 하지만 $X \perp\!\!\!\perp (Y, W) \mid Z$는 **성립하지 않음** — $X$와 $(Y, W)$는 완전 상관

이 반례에서 $P(Y = 0, W = 1) = 0$ 같은 0 확률 원소가 문제를 일으킨다. Positive density $P > 0$이면 이런 퇴화가 없어서 intersection이 성립한다 (Pearl 1988, Dawid 1979). $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
from itertools import product

# 세 이산 변수 X, Y, Z 위의 결합분포 샘플 생성
# 시나리오 1: Chain X → Z → Y (X ⊥ Y | Z 이어야 함)
np.random.seed(42)

n_states = 3
n_samples = 100_000

# X ~ Uniform(0,1,2)
X = np.random.choice(n_states, n_samples)
# Z | X: 대각 강한 조건부분포
A_XZ = np.array([[0.7, 0.2, 0.1],
                 [0.2, 0.6, 0.2],
                 [0.1, 0.2, 0.7]])
Z = np.array([np.random.choice(n_states, p=A_XZ[x]) for x in X])
# Y | Z: 또 다른 대각 강한 조건부분포
A_ZY = np.array([[0.8, 0.1, 0.1],
                 [0.1, 0.6, 0.3],
                 [0.1, 0.3, 0.6]])
Y = np.array([np.random.choice(n_states, p=A_ZY[z]) for z in Z])

def joint_table(X, Y, Z, n=3):
    P = np.zeros((n, n, n))
    for x, y, z in zip(X, Y, Z):
        P[x, y, z] += 1
    return P / P.sum()

P = joint_table(X, Y, Z)

# 조건부 독립 검증: P(X,Y|Z) vs P(X|Z)P(Y|Z)
def check_CI(P, verbose=False):
    """X ⊥ Y | Z 인지 검증."""
    P_XYZ = P
    P_Z = P_XYZ.sum(axis=(0, 1))
    P_XZ = P_XYZ.sum(axis=1)
    P_YZ = P_XYZ.sum(axis=0)
    
    max_diff = 0
    for z in range(P.shape[2]):
        if P_Z[z] < 1e-6:
            continue
        P_XY_given_Z = P_XYZ[:, :, z] / P_Z[z]
        P_X_given_Z = P_XZ[:, z] / P_Z[z]
        P_Y_given_Z = P_YZ[:, z] / P_Z[z]
        product_independent = np.outer(P_X_given_Z, P_Y_given_Z)
        diff = np.abs(P_XY_given_Z - product_independent).max()
        max_diff = max(max_diff, diff)
        if verbose:
            print(f"  z={z}: max|P(X,Y|z) - P(X|z)P(Y|z)| = {diff:.4f}")
    return max_diff

print("=" * 60)
print("시나리오 1: Chain X → Z → Y")
print("=" * 60)
ci_violation = check_CI(P, verbose=True)
print(f"  X ⊥ Y | Z 위반 정도: {ci_violation:.4f}")
print(f"  → {'조건부 독립 성립' if ci_violation < 0.01 else '위반'}")

# 시나리오 2: Collider (v-structure) X → Z ← Y, X ⊥ Y (unconditionally) 이지만 X ⊥̸ Y | Z
X2 = np.random.choice(n_states, n_samples)
Y2 = np.random.choice(n_states, n_samples)  # X와 독립
# Z = (X + Y) mod 3 (결정론적 결합)
Z2 = (X2 + Y2) % n_states

P2 = joint_table(X2, Y2, Z2)

print("\n" + "=" * 60)
print("시나리오 2: Collider X → Z ← Y (explaining away)")
print("=" * 60)

# X ⊥ Y (주변 독립)
P2_X = P2.sum(axis=(1, 2))
P2_Y = P2.sum(axis=(0, 2))
P2_XY = P2.sum(axis=2)
marginal_diff = np.abs(P2_XY - np.outer(P2_X, P2_Y)).max()
print(f"  P(X,Y) vs P(X)P(Y) 최대 차이 = {marginal_diff:.4f}")
print(f"  → X ⊥ Y {'성립' if marginal_diff < 0.01 else '위반'} (unconditional)")

# X ⊥ Y | Z 검사
ci_violation2 = check_CI(P2)
print(f"  X ⊥ Y | Z 위반 정도: {ci_violation2:.4f}")
print(f"  → {'조건부 독립 성립' if ci_violation2 < 0.01 else '위반 — explaining away!'}")
```

**출력 예시**:
```
============================================================
시나리오 1: Chain X → Z → Y
============================================================
  z=0: max|P(X,Y|z) - P(X|z)P(Y|z)| = 0.0054
  z=1: max|P(X,Y|z) - P(X|z)P(Y|z)| = 0.0041
  z=2: max|P(X,Y|z) - P(X|z)P(Y|z)| = 0.0038
  X ⊥ Y | Z 위반 정도: 0.0054
  → 조건부 독립 성립

============================================================
시나리오 2: Collider X → Z ← Y (explaining away)
============================================================
  P(X,Y) vs P(X)P(Y) 최대 차이 = 0.0027
  → X ⊥ Y 성립 (unconditional)
  X ⊥ Y | Z 위반 정도: 0.3333
  → 위반 — explaining away!
```

즉 Chain에서는 $X \perp\!\!\!\perp Y \mid Z$가 성립하고, Collider에서는 $X \perp\!\!\!\perp Y$이지만 $Z$를 조건으로 걸면 위반된다 — 이것이 d-separation의 세 패턴 중 두 가지.

---

## 🔗 AI/ML 연결

### Naive Bayes와 조건부 독립 가정

Naive Bayes classifier는 class $C$가 주어졌을 때 feature들이 조건부 독립이라 가정:
$$P(X_1, \ldots, X_d \mid C) = \prod_{i=1}^d P(X_i \mid C)$$

이는 **매우 강한 가정**이다 — 실제로 feature들은 서로 상관이 있을 수 있다. 하지만 이 가정 덕분에 고차원에서도 파라미터 수가 $O(d)$로 억제되고, **sparse data regime**에서 잘 동작한다.

### Markov 가정 (HMM, RNN)

HMM은 미래 상태가 현재 상태가 주어졌을 때 과거와 독립이라 가정:
$$Z_{t+1} \perp\!\!\!\perp Z_{1:t-1} \mid Z_t$$

이는 chain $Z_{t-1} \to Z_t \to Z_{t+1}$의 조건부 독립. RNN의 hidden state도 이 가정의 학습된 버전.

### VAE의 Mean-Field Approximation

VAE는 posterior $q(z | x)$를 다차원 가우시안으로 근사하면서, 종종 **independent components** 가정을 추가:
$$q(z \mid x) = \prod_i q(z_i \mid x)$$

이는 "주어진 $x$에 대해 latent 차원들은 조건부 독립"이라는 **mean-field assumption**. Ch6-01에서 이 가정이 ELBO 최적화에 어떻게 들어가는지 자세히 다룸.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 이산 변수 가정 | 연속 변수는 밀도로 치환, 측도론적 정의(1.2) 필요 |
| $P(Z = z) > 0$ | 0 확률 사건 위 조건부 분포는 일반적으로 unique하지 않음 (Borel paradox) |
| Positive density | Intersection property가 없으면 d-separation의 completeness가 약해짐 |
| Semi-graphoid 공리 | 공리계는 완전하지 않음 (Studený 1989) — 모든 유효한 CI 관계를 도출하지 못하는 경우 존재 |

**주의**: Semi-graphoid 공리가 모든 CI 관계를 포착하지는 못한다 — **complete characterization은 open problem**이다 (Studený 1989, 2005). 그래프 모델은 이 대수적 제한을 **그래프 구조로 치환**하는 영리한 표현.

---

## 📌 핵심 정리

$$\boxed{X \perp\!\!\!\perp Y \mid Z \iff P(X, Y \mid Z) = P(X \mid Z) P(Y \mid Z)}$$

| 공리 | 서술 | 직관 |
|------|------|------|
| **Symmetry** | $X \perp\!\!\!\perp Y \mid Z \Rightarrow Y \perp\!\!\!\perp X \mid Z$ | 독립은 대칭 |
| **Decomposition** | $X \perp\!\!\!\perp (Y, W) \mid Z \Rightarrow X \perp\!\!\!\perp Y \mid Z$ | 합집합이 독립이면 부분도 |
| **Weak Union** | $X \perp\!\!\!\perp (Y, W) \mid Z \Rightarrow X \perp\!\!\!\perp Y \mid (Z, W)$ | 관측을 더해도 독립 유지 |
| **Contraction** | $X \perp\!\!\!\perp Y \mid Z, X \perp\!\!\!\perp W \mid (Y, Z) \Rightarrow X \perp\!\!\!\perp (Y, W) \mid Z$ | 두 단계 독립의 결합 |
| **Intersection** | (positive density 시) | 두 조건부 독립이 하나로 결합 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $X, Y, Z$가 모두 Bernoulli(0.5)이고 서로 독립이라 하자. $W := X \oplus Y$ (XOR)라 할 때,

(a) $W \perp\!\!\!\perp X$인가?
(b) $W \perp\!\!\!\perp X \mid Y$인가?

<details>
<summary>힌트 및 해설</summary>

(a) $W \perp\!\!\!\perp X$: **YES**. $P(W = 1) = 1/2$이고 $P(W = 1 | X = 0) = P(Y = 1) = 1/2$, $P(W = 1 | X = 1) = P(Y = 0) = 1/2$. 따라서 $X$는 $W$에 정보를 주지 않는다.

(b) $W \perp\!\!\!\perp X \mid Y$: **NO**. $Y$를 알면 $W = X \oplus Y$는 $X$의 결정론적 함수. $P(W = 0 | X = 0, Y = 0) = 1$, $P(W = 0 | X = 1, Y = 0) = 0$. 따라서 $Y$가 주어졌을 때 $W$는 $X$와 완전 상관.

이 예시는 **"주변 독립이 조건부 독립을 함의하지 않는다"**의 전형. XOR은 심지어 쌍별(pairwise) 독립이지만 세 변수 전체로는 완전 결정론적 관계.

</details>

**문제 2** (심화): 정리 1.1의 (3) $P(X, Y, Z) = g(X, Z) h(Y, Z)$에서, $g$와 $h$가 유일하지 않음을 보여라 (힌트: $g \to g \cdot c(z)$, $h \to h / c(z)$).

<details>
<summary>힌트 및 해설</summary>

$c(z) > 0$인 임의의 함수에 대해, $\tilde g(x, z) := g(x, z) \cdot c(z)$, $\tilde h(y, z) := h(y, z) / c(z)$로 놓으면
$$\tilde g(x, z) \tilde h(y, z) = g(x, z) h(y, z) \cdot \frac{c(z)}{c(z)} = g(x, z) h(y, z) = P(x, y, z)$$

따라서 인수분해는 **$z$-방향의 자유도**를 갖는다. 이는 MRF의 clique potential $\phi_C(x_C)$가 unique하지 않은 이유(**rescaling freedom**)와 같은 현상. Exponential family 형태로 canonical parameterization을 하면 이 자유도를 고정할 수 있다.

</details>

**문제 3** (AI 연결): Naive Bayes가 스팸 필터에서 조건부 독립 가정 $P(\text{word}_i, \text{word}_j \mid \text{spam}) = P(\text{word}_i \mid \text{spam}) P(\text{word}_j \mid \text{spam})$을 쓴다. 이 가정이 **틀린데도** Naive Bayes가 잘 동작하는 이유를 설명하라.

<details>
<summary>힌트 및 해설</summary>

조건부 독립 가정은 실제로 위반된다 ("click"과 "link"는 스팸에서 상관). 하지만 Naive Bayes가 잘 동작하는 이유는:

1. **분류 성능은 확률 추정의 정확성보다 decision boundary의 방향만 필요**: $P(y = \text{spam} | x) > P(y = \text{ham} | x)$라는 rank-ordering만 맞으면 된다. 조건부 독립 위반이 양쪽 사후확률에 **비슷한 방향**으로 영향을 주면 순서는 유지.

2. **Zhang (2004)의 최적성 분석**: 의존성 구조가 클래스 간 대칭이면 Naive Bayes는 여전히 Bayes-optimal.

3. **고차원 희소성**: feature 수가 많고 데이터가 적을 때, 완전한 joint $P(x | y)$ 학습은 파라미터 폭발로 실패. **biased but low-variance** estimator가 더 낫다.

4. **Calibration은 나쁨**: Naive Bayes의 확률 출력은 극단적(너무 0 또는 1에 가까움)이지만, **argmax**만 쓰는 분류에는 문제없다.

이 예는 "모든 모델은 틀렸지만 일부는 유용하다" (George Box)의 전형. 그래프 모델의 CI 가정을 의식적으로 **자유도 감소 장치**로 쓰는 철학의 시작.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 📚 README](../README.md) | | [02. Bayesian Network — DAG 기반 인수분해 ▶](./02-bayesian-network-factorization.md) |

</div>
