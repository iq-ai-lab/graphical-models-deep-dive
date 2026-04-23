# 01. CRF의 정의 — Logistic Regression의 구조화 확장

## 🎯 핵심 질문

- CRF의 정의 $p(y | x) = \frac{1}{Z(x)} \exp(\sum_k w_k f_k(y, x))$에서 feature function $f_k$는 어떤 역할을 하는가?
- CRF와 HMM의 본질적 차이 — **discriminative vs generative** — 는 어떻게 수학적으로 구분되는가?
- Partition function $Z(x)$가 $x$에만 의존하는 것이 왜 CRF의 강점인가?
- CRF는 logistic regression을 어떻게 구조화된 output에 확장하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**CRF**는 structured prediction의 표준. **NER, POS tagging, chunking**의 모든 SOTA system (pre-BERT)은 CRF + features. 이미지 semantic segmentation의 dense CRF post-processing, 생물정보학의 gene finding, OCR의 character recognition. BERT 이후에도 **BiLSTM-CRF, Transformer-CRF** 아키텍처가 sequence labeling에서 표준. CRF를 "HMM의 discriminative 쌍둥이"로 이해하면 **generative vs discriminative trade-off**와 **feature engineering의 자유도**를 정확히 파악할 수 있다.

---

## 📐 수학적 선행 조건

- [Ch1-04 Markov Random Field와 Hammersley–Clifford](../ch1-conditional-independence/04-markov-random-field.md): MRF의 log-linear form
- [Ch3-01 HMM의 정의](../ch3-hmm/01-hmm-definition.md): HMM과 비교 대상
- Logistic regression: $p(y | x) = \sigma(w^T x)$
- Exponential family

---

## 📖 직관적 이해

### Generative vs Discriminative

**Generative model** (HMM, Naive Bayes): $p(x, y)$ 전체를 모델링. Inference: $p(y | x) = p(x, y) / p(x)$.

**Discriminative model** (Logistic Regression, CRF): 바로 $p(y | x)$를 모델링. $p(x)$는 신경 쓰지 않음.

| | Generative | Discriminative |
|--|-----------|----------------|
| 모델 | $p(x, y)$ | $p(y \| x)$ |
| 학습 | Joint log-likelihood | Conditional log-likelihood |
| Feature | $x$의 분포 가정 필요 | Arbitrary feature of $x$ |
| 데이터 효율 | Low data에서 강함 | High data에서 강함 |
| 예측 정확도 | 일반적으로 낮음 | 일반적으로 높음 (Ng-Jordan 2002) |

### CRF의 핵심 통찰

HMM: $p(y, x) = \pi_{y_1} \prod p(y_t | y_{t-1}) \prod p(x_t | y_t)$.

CRF: $p(y | x) = \frac{1}{Z(x)} \exp(\sum_k w_k f_k(y, x))$.

**차이**:
- HMM은 $x$에 대한 **분포를 가정** — $p(x | y)$ 등
- CRF는 $x$를 **주어진 조건**으로만 사용 — $x$의 분포를 모델링하지 않음

**결과**:
- Feature function $f_k(y, x)$가 **$x$의 임의 함수** 가능 (미래 context, 현재 character의 shape, 외부 lexicon 등)
- HMM에서는 이런 overlapping features를 추가하면 $p(x | y)$의 정규화 어려워짐
- CRF의 normalization $Z(x) = \sum_y \exp(\sum w_k f_k(y, x))$는 **각 $x$마다 독립적으로** 계산

### Feature Function Examples (NER)

"Bill Clinton visited Seoul"의 NER에서:

- $f_1(y, x) = \mathbb{1}[y_t = \text{PERSON}, x_t \text{ 시작하는 대문자}]$
- $f_2(y, x) = \mathbb{1}[y_t = \text{LOCATION}, x_t \text{ gazetteer에 도시명으로 있음}]$
- $f_3(y_{t-1}, y_t) = \mathbb{1}[y_{t-1} = \text{PERSON}, y_t = \text{PERSON}]$ (transition)
- $f_4(y_t, x) = \mathbb{1}[y_t = \text{ORG}, x_{t+1} = \text{"Inc."}]$ (future context!)

이런 **overlapping, context-dependent** features가 HMM에서는 정규화 문제를 일으키지만, CRF에서는 자연스럽다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Conditional Random Field

**CRF**는 조건부 확률 분포 $p(y | x)$로, $x \in \mathcal{X}$, $y \in \mathcal{Y}$에 대해

$$p(y | x; w) = \frac{1}{Z(x; w)} \exp\left(\sum_k w_k f_k(y, x)\right)$$

여기서
- $f_k: \mathcal{Y} \times \mathcal{X} \to \mathbb{R}$: **feature function**
- $w_k \in \mathbb{R}$: **weight parameters**
- $Z(x; w) := \sum_y \exp(\sum_k w_k f_k(y, x))$: **conditional partition function**

### 정의 1.2 — Linear-Chain CRF

Sequence $y = (y_1, \ldots, y_T)$에 대해, feature가 **인접 pair**에만 의존:

$$p(y | x; w) = \frac{1}{Z(x)} \exp\left(\sum_{t=1}^T \sum_k w_k f_k(y_{t-1}, y_t, x, t)\right)$$

$y_0$는 "start symbol". 이것이 HMM의 structure에 가장 직접 대응되는 CRF 형태.

### 정의 1.3 — CRF as Conditional MRF

CRF는 **조건부 MRF** — $x$를 evidence로 고정한 후 $y$ 위에 MRF:

$$p(y | x) \propto \prod_C \phi_C(y_C, x)$$

각 clique potential $\phi_C(y_C, x) = \exp(\sum_k w_k f_k(y_C, x))$가 $y_C$와 $x$ 모두에 의존.

### 정의 1.4 — HMM의 CRF 표현

HMM $p(y, x) = \pi_{y_1} \prod A_{y_{t-1}, y_t} \prod B_{y_t, x_t}$을 다음 CRF로 쓸 수 있음:

$$p(y, x) = \frac{1}{Z} \exp\left(\log \pi_{y_1} + \sum_t \log A_{y_{t-1}, y_t} + \sum_t \log B_{y_t, x_t}\right)$$

Feature:
- $f^\pi_{y_1}(y_1) = \mathbb{1}[y_1 = \cdot]$, weight = $\log \pi_\cdot$
- $f^A_{i, j}(y_{t-1}, y_t) = \mathbb{1}[y_{t-1} = i, y_t = j]$, weight = $\log A_{i, j}$
- $f^B_{i, k}(y_t, x_t) = \mathbb{1}[y_t = i, x_t = k]$, weight = $\log B_{i, k}$

하지만 $Z$가 $y$와 $x$ 모두에 대한 합. $p(y | x) = p(y, x) / p(x)$로 바꾸면 CRF와 동치.

---

## 🔬 정리와 증명

### 정리 1.1 — Logistic Regression이 Singleton CRF

**명제**: $y \in \{1, \ldots, K\}$ (scalar categorical), $x \in \mathbb{R}^d$에 대해 Logistic Regression은 "CRF with singleton features":
$$p(y = k | x) = \frac{\exp(w_k^T x)}{\sum_{k'} \exp(w_{k'}^T x)}$$

**증명**: Feature $f_{k, d}(y, x) = \mathbb{1}[y = k] \cdot x_d$, weight $w_{k, d}$. 그러면
$$p(y | x) = \frac{\exp(\sum_{k, d} w_{k, d} \mathbb{1}[y = k] x_d)}{Z(x)} = \frac{\exp(\sum_d w_{y, d} x_d)}{Z(x)} = \frac{\exp(w_y^T x)}{\sum_{y'} \exp(w_{y'}^T x)}$$

이는 multinomial logistic regression의 softmax. $\square$

### 정리 1.2 — Partition Function의 $x$-Dependency

**명제**: CRF의 $Z(x) = \sum_y \exp(\sum_k w_k f_k(y, x))$는 각 $x$마다 **따로 계산**. 하지만 log-linear structure는 efficient inference를 가능하게 함 (treewidth에 따라).

**증명 sketch**: 

$y$의 conditional distribution의 partition function은 각 입력 $x$마다 다른 값. HMM의 $p(x) = \sum_y p(x, y)$와 달리 $y$에 대한 marginalization만 필요.

Linear-chain CRF에서 $Z(x) = \sum_y \exp(\sum_t \phi_t(y_{t-1}, y_t, x_t))$는 HMM의 forward algorithm과 **구조적으로 동일**한 recursion으로 $O(K^2 T)$ 계산:

$$\alpha_t(k) = \sum_{k'} \alpha_{t-1}(k') \exp(\phi_t(k', k, x_t))$$

$Z(x) = \sum_k \alpha_T(k)$. $\square$

### 정리 1.3 — CRF은 Exponential Family

**명제**: CRF는 $y$에 대한 exponential family 분포 (with $x$-dependent natural parameters).

**증명**: 
$$p(y | x; w) = \exp\left(\sum_k w_k f_k(y, x) - \log Z(x; w)\right)$$

Exponential family form $p(y | \eta) = h(y) \exp(\eta^T T(y) - A(\eta))$:
- $T(y, x) = (f_1(y, x), f_2(y, x), \ldots)$: sufficient statistics
- $\eta = w$: natural parameters
- $A(\eta; x) = \log Z(x; w)$: log-partition
- $h(y) = 1$ (no base measure)

**결과**: Exponential family 이론 적용 가능:
- Log-likelihood gradient = mean parameter와 empirical statistic의 차
- 유일한 MLE (log-likelihood가 concave)
- Fisher information = covariance of sufficient statistics

$\square$

### 정리 1.4 — CRF Log-Likelihood Concave

**명제**: CRF log-likelihood $\ell(w) = \sum_i \log p(y^{(i)} | x^{(i)}; w)$는 $w$에 대해 **concave**.

**증명** (standard for exponential family):
$$\log p(y | x; w) = \sum_k w_k f_k(y, x) - \log Z(x; w)$$

첫 항은 linear in $w$. Log-partition $\log Z(x; w) = \log \sum_y \exp(\sum_k w_k f_k(y, x))$은 **convex in $w$** (log-sum-exp is convex).

Linear $-$ convex $=$ concave. Sum over data points: 여전히 concave. $\square$

**함의**: Unique global maximum! Gradient ascent나 L-BFGS로 확실히 최적 찾음. HMM Baum-Welch의 local optima 문제 없음 (discriminative training에서).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_partition_function(weights, x, K, T, features_fn):
    """Brute force computation of Z(x) — only for small (K, T)."""
    from itertools import product
    total = 0
    for y in product(range(K), repeat=T):
        score = sum(w * f for w, f in zip(weights, features_fn(y, x)))
        total += np.exp(score)
    return total

def example_features(y, x):
    """Simple feature function: 
    - emission: 1[y_t = k, x_t = o]
    - transition: 1[y_{t-1} = i, y_t = j]
    """
    T = len(y)
    K = max(y) + 1  # max label
    feats = []
    # Emission features (K * O features, O = observation vocab)
    # 여기서는 간단한 버전: f_k(y, x) = sum_t 1[y_t = k, x_t = 0]
    for k in range(K):
        feats.append(sum(1 for t in range(T) if y[t] == k and x[t] == 0))
        feats.append(sum(1 for t in range(T) if y[t] == k and x[t] == 1))
    # Transition features (K^2)
    for i in range(K):
        for j in range(K):
            feats.append(sum(1 for t in range(1, T) if y[t-1] == i and y[t] == j))
    return feats

# 예시: 2 labels, T = 4
K = 2
T = 4
x = [0, 1, 0, 1]

# Random weights
np.random.seed(0)
n_features = 2 * K + K * K
weights = np.random.randn(n_features)

Z = compute_partition_function(weights, x, K, T, example_features)
print(f"Z(x) = {Z:.4f}")

# p(y | x)
from itertools import product
for y in product(range(K), repeat=T):
    feats = example_features(y, x)
    score = sum(w * f for w, f in zip(weights, feats))
    p = np.exp(score) / Z
    if p > 0.1:
        print(f"y = {y}: p(y|x) = {p:.4f}")

# HMM과 비교 — same structure
# HMM generative: p(x, y) = π·A·B
# CRF discriminative: p(y | x) ∝ exp(weighted features)
# 같은 features 사용 시 수학적으로 동일한 forward algorithm 사용 가능

# Log-likelihood 증가 — L-BFGS 학습 시뮬레이션
# (데모: 작은 synthetic data)

# Generate synthetic data with known parameters
def generate_crf_data(n_samples, weights_true, K, T, features_fn):
    """Sample (y, x) pairs from CRF."""
    rng = np.random.default_rng(42)
    data = []
    for _ in range(n_samples):
        # Simple: sample x uniformly, then y | x
        x = list(rng.integers(0, 2, T))
        # Compute p(y | x)
        probs = []
        ys = []
        for y in product(range(K), repeat=T):
            feats = features_fn(y, x)
            score = sum(w * f for w, f in zip(weights_true, feats))
            probs.append(np.exp(score))
            ys.append(y)
        probs = np.array(probs) / sum(probs)
        y_idx = rng.choice(len(ys), p=probs)
        data.append((list(ys[y_idx]), x))
    return data

weights_true = np.random.randn(n_features) * 0.5
data = generate_crf_data(100, weights_true, K, T, example_features)

def neg_log_likelihood(weights, data, K, T, features_fn):
    """-log p(y | x)."""
    nll = 0
    for y, x in data:
        Z = compute_partition_function(weights, x, K, T, features_fn)
        feats = features_fn(y, x)
        score = sum(w * f for w, f in zip(weights, feats))
        nll -= score - np.log(Z)
    return nll / len(data)

from scipy.optimize import minimize

w_init = np.zeros(n_features)
result = minimize(neg_log_likelihood, w_init, args=(data, K, T, example_features),
                  method='BFGS', options={'maxiter': 50})

print(f"\nInitial NLL: {neg_log_likelihood(w_init, data, K, T, example_features):.4f}")
print(f"Final NLL:   {result.fun:.4f}")
print(f"Converged: {result.success}")

# 시각화
# ... (생략)
```

**출력 예시**:
```
Z(x) = 23.8752
y = (0, 0, 0, 0): p(y|x) = 0.1823
y = (1, 0, 1, 0): p(y|x) = 0.2105
y = (1, 1, 1, 1): p(y|x) = 0.1534

Initial NLL: 2.7726
Final NLL:   1.4891
Converged: True
```

CRF는 log-likelihood concave이므로 BFGS 수렴 보장.

---

## 🔗 AI/ML 연결

### NER, POS Tagging의 표준

CoNLL-2003 NER shared task (2003)에서 CRF-based system이 top. 이후 **Stanford NER**, **MALLET**, **CRF++** 등이 표준 tool이 됨. Feature engineering이 차별화 포인트.

Pre-BERT era SOTA:
- **POS tagging**: 97.5% accuracy (Toutanova et al. 2003, CRF with rich features)
- **NER**: F1 ≈ 89 (Finkel et al. 2005, CRF with gazetteer features)

### BiLSTM-CRF (Ch4-04의 예고)

Huang-Xu-Yu 2015, Lample et al. 2016:
- **BiLSTM**: context-aware feature extraction
- **CRF layer**: output consistency (transition constraints)
- 결과: CoNLL-2003 F1 ≈ 91 (neural features + CRF)

왜 순수 neural softmax가 아니라 CRF를 쓰는가?
- Softmax는 각 time step **독립적** prediction → B-PER 다음에 I-LOC 같은 **inconsistent** sequence 가능
- CRF가 transition constraint를 학습 → consistent output

### Dense CRF for Image Segmentation

Krähenbühl-Koltun 2011:
- Dense CRF: 모든 pixel pair 사이에 edge (fully-connected)
- Mean-field approximation로 inference
- **DeepLab** (Chen et al. 2016): CNN + dense CRF post-processing → state-of-art semantic segmentation

### Structured Perceptron과의 관계

Collins 2002의 **structured perceptron**은 CRF의 variant:
- CRF: $p(y | x) \propto \exp(w \cdot f)$, log-likelihood 최적화
- Perceptron: $\hat y = \arg\max_y w \cdot f$, mistake-driven update

같은 feature-based scoring, 다른 학습 objective. Practical 성능 비슷.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Feature engineering | Neural feature extractors가 대부분의 경우 우월 (BiLSTM-CRF) |
| Linear scoring | Non-linear interactions 포착 제한 — 최신 neural CRF로 해결 |
| Exact inference | General CRF topology에서 intractable — linear chain에만 exact |
| $p(x)$ 모델링 X | $x$ 분포 자체에 관심있다면 generative가 유리 |

**주의**: **High-data + rich features**에서 CRF가 HMM을 outperform (Ng-Jordan 2002, Lafferty et al. 2001). **Low-data**에서는 generative HMM이 더 stable (parameter 수가 작고 regularization 자연스러움).

---

## 📌 핵심 정리

$$\boxed{p(y | x; w) = \frac{1}{Z(x; w)} \exp\left(\sum_k w_k f_k(y, x)\right)}$$

| 특성 | CRF | HMM |
|------|-----|-----|
| 모델링 | Discriminative $p(y \| x)$ | Generative $p(x, y)$ |
| Feature | Arbitrary of $x$ | Restricted to emissions |
| Normalization | Per-$x$ ($Z(x)$) | Global ($Z = 1$ for generative CPTs) |
| Log-likelihood | Concave (unique optimum) | Non-concave with EM (local) |
| Flexibility | High | Low (strong independence assumptions) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): CRF에서 "$x$의 overlapping features"가 왜 HMM에서는 문제가 되는가?

<details>
<summary>힌트 및 해설</summary>

HMM의 $p(x | y) = \prod_t p(x_t | y_t)$은 **각 $x_t$가 $y_t$에만 의존**하는 factorization. 만약 "현재 단어의 suffix"와 "현재 단어의 POS tag"를 모두 feature로 쓰고 싶다면:

- Feature 1: $f_1 = x_t$ (단어)
- Feature 2: $f_2 = \text{suffix}(x_t)$ (접미사)

HMM에서는 $p(x_t | y_t)$가 joint distribution — $x_t$와 $\text{suffix}(x_t)$의 joint. $\text{suffix}(x_t)$는 $x_t$의 함수 (결정론적)이므로 **중복 정보**. $p(x_t, \text{suffix}(x_t) | y_t) = p(x_t | y_t)$ (suffix는 $x_t$가 주어지면 자동 결정).

결론: HMM에서 feature 추가 = $p(x | y)$ 재정의, 정규화 재계산. **각 feature가 독립 emission처럼 취급되면 double counting**.

**CRF는 이 문제 없음**: $p(y | x)$만 모델링 → $x$의 features가 arbitrary overlap 가능. $Z(x)$는 각 $x$에 대해 독립 정규화. 이것이 CRF의 **feature engineering 자유도**의 수학적 근거.

</details>

**문제 2** (심화): Ng-Jordan (2002)의 "Generative vs Discriminative" 논문의 핵심 결과 — "discriminative가 일반적으로 우수하지만 generative가 low-data regime에서 먼저 수렴" — 를 직관적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Asymptotic regime (large data)**:
- Discriminative ($p(y | x)$): 직접 task-relevant distribution 학습 → **Bayes error에 빠르게 접근**
- Generative ($p(x, y)$): $p(x)$도 모델링 — **misspecified** $p(x)$에 의해 misleading 가능

**Low-data regime**:
- Discriminative: 파라미터 많음 → overfitting 위험
- Generative: strong modeling assumption ($p(x | y)$ factorization 같은) → inductive bias → 적은 데이터로도 generalize

**구체적**:
- Naive Bayes (gen): 파라미터 $O(K \cdot d)$, $d$ features. Assumption: $x$ features CI given $y$.
- Logistic Regression (disc): 파라미터 $O(K \cdot d)$, 같은 수지만 assumption 없음.

Ng-Jordan의 결과:
- LR의 샘플 복잡도 $O(d)$ (Bayes risk에 접근하려면)
- NB의 샘플 복잡도 $O(\log d)$ (assumption이 맞다면)
- NB는 asymptotic error가 nonzero (bias from wrong assumption), LR는 consistent

**실용**:
- Training data 충분할 때: discriminative (CRF, LR, neural nets) 선호
- Training data 적을 때: generative (NB, HMM) with informative prior
- **Semi-supervised**: generative + unlabeled data가 powerful

</details>

**문제 3** (AI 연결): BERT로 NER을 할 때 "BERT + softmax per token" vs "BERT + CRF layer"의 차이를 분석하라.

<details>
<summary>힌트 및 해설</summary>

**BERT + Softmax (token classification)**:
$$p(y_t | x) = \text{softmax}(W h_t + b)$$

각 token position에서 **독립적** label prediction. Token-level accuracy 기준으로 학습.

**BERT + CRF**:
$$p(y | x) \propto \exp\left(\sum_t [\text{emission}(y_t, h_t) + \text{transition}(y_{t-1}, y_t)]\right)$$

Emission = BERT output projected to label scores. Transition = learnable $K \times K$ matrix. **Sequence-level** log-likelihood 학습.

**Trade-off**:
- **장점 of CRF layer**:
  - 일관성 강제: "B-PER → I-LOC" 같은 invalid 방지
  - Sequence-level optimization: token보다 span accuracy 중요한 task에서 유리 (NER F1)
  
- **단점 of CRF layer**:
  - Training/inference가 느림 (Viterbi, forward-backward)
  - 많은 데이터에서는 BERT가 이미 충분히 context 포착 → CRF 추가 이득 작음
  - Gradient flow가 복잡

**경험적 결과** (최근 논문들, e.g., Devlin et al. 2019):
- Small / noisy data: CRF 추가 도움 (+0.5-2 F1)
- Large data: 비슷하거나 미미 (+0.1-0.5 F1)
- CoNLL-2003: BERT-large alone ≈ 92.8 F1, BERT-large + CRF ≈ 93.0 F1

**결론**: CRF layer는 "Structural regularization" — BERT의 표현력이 부족할 때 도움. BERT-large 이상의 모델에서는 marginal. 하지만 **low-resource settings (rare NER classes)**에서는 여전히 유용.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch3-05 Linear Dynamical System과 Kalman Filter](../ch3-hmm/05-kalman-filter.md) | [📚 README](../README.md) | [02. Linear-Chain CRF의 Inference와 Learning ▶](./02-linear-chain-crf.md) |

</div>
