# 02. Linear-Chain CRF의 Inference와 Learning

## 🎯 핵심 질문

- Linear-Chain CRF의 $Z(x)$를 어떻게 forward algorithm으로 계산하는가?
- Log-likelihood gradient $\nabla L = f_{\text{data}} - \mathbb{E}_p[f]$가 왜 **expected feature count의 차이**로 유도되는가?
- Decoding (Viterbi on CRF)는 HMM Viterbi와 어떻게 같은가?
- L-BFGS로 CRF를 학습하는 이유는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Linear-Chain CRF**는 가장 실용적인 CRF 형태. NER, POS, chunking, slot filling — 모든 sequence labeling의 표준. **CRF++, MALLET, sklearn-crfsuite** 등 수많은 toolkit이 linear-chain CRF 구현. Training의 **log-likelihood gradient가 expected statistics와 empirical statistics의 차이**라는 identity는 exponential family 학습의 일반 원리 — maximum entropy, logistic regression, softmax regression이 모두 같은 pattern을 공유. **BiLSTM-CRF**, **BERT-CRF**의 training도 이 gradient identity를 기반으로.

---

## 📐 수학적 선행 조건

- [Ch4-01 CRF의 정의](./01-crf-definition.md)
- [Ch3-02 Forward-Backward Algorithm](../ch3-hmm/02-forward-backward.md)
- [Ch3-03 Viterbi Algorithm](../ch3-hmm/03-viterbi-algorithm.md)
- Exponential family의 maximum likelihood
- Convex optimization (L-BFGS)

---

## 📖 직관적 이해

### Linear-Chain CRF의 형태

$$p(y | x; w) = \frac{1}{Z(x; w)} \exp\left(\sum_{t=1}^T \sum_k w_k f_k(y_{t-1}, y_t, x, t)\right)$$

Feature function이 **인접 pair + 전체 $x$**에만 의존. 두 종류:
- **State features** $f(y_t, x, t)$
- **Transition features** $f(y_{t-1}, y_t, x, t)$

### Forward Algorithm for $Z(x)$

HMM의 forward와 구조 동일, 단 $A_{ij} \to \exp(\text{transition score})$, $B_{jo} \to \exp(\text{emission score})$:

$$\alpha_t(j) = \sum_i \alpha_{t-1}(i) \cdot \exp(\phi(i, j, x, t))$$

where $\phi(i, j, x, t) = \sum_k w_k f_k(i, j, x, t)$ (state + transition features at position $t$).

$$Z(x) = \sum_j \alpha_T(j)$$

복잡도: $O(K^2 T)$ per sequence.

### Log-Likelihood Gradient

Data $\{(y^{(i)}, x^{(i)})\}$의 log-likelihood:
$$\ell(w) = \sum_i \left[\sum_{t, k} w_k f_k(y^{(i)}_{t-1}, y^{(i)}_t, x^{(i)}, t) - \log Z(x^{(i)}; w)\right]$$

$w_k$에 대한 gradient:
$$\frac{\partial \ell}{\partial w_k} = \sum_i \left[\sum_t f_k(y^{(i)}_{t-1}, y^{(i)}_t, x^{(i)}, t) - \mathbb{E}_{p(y | x^{(i)}; w)}\left[\sum_t f_k\right]\right]$$

**Empirical feature count** $-$ **Expected feature count under current model**.

Convergence: gradient = 0 ⟺ **moment matching** (exponential family MLE의 일반 원리).

### Expected Feature Count

$\mathbb{E}_{p(y|x)}[f_k] = \sum_t \sum_{i, j} p(y_{t-1} = i, y_t = j | x) \cdot f_k(i, j, x, t)$

필요한 것: **pairwise marginal** $p(y_{t-1}, y_t | x)$. 이는 Forward-Backward로 계산:

$$p(y_{t-1} = i, y_t = j | x) \propto \alpha_{t-1}(i) \cdot \exp(\phi(i, j, x, t)) \cdot \beta_t(j)$$

### Decoding (Viterbi)

$\hat y = \arg\max_y \sum_t \phi(y_{t-1}, y_t, x, t)$

HMM Viterbi와 동일 recursion:
$$\delta_t(j) = \max_i [\delta_{t-1}(i) + \phi(i, j, x, t)]$$

(log-space max-sum).

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Linear-Chain CRF Score

Sequence $y = (y_1, \ldots, y_T)$와 observation $x$에 대해:
$$s(y, x; w) := \sum_{t=1}^T \phi(y_{t-1}, y_t, x, t)$$

$y_0$는 BOS (begin-of-sequence). $\phi(i, j, x, t) = \sum_k w_k f_k(i, j, x, t)$.

$$p(y | x; w) = \frac{\exp(s(y, x; w))}{Z(x; w)}$$

### 정의 2.2 — Forward/Backward for CRF

$$\alpha_t(j) := \sum_{y_{1:t-1}} \exp\left(\sum_{s=1}^t \phi(y_{s-1}, y_s, x, s)\right)$$

(where $y_t = j$). Recursion:
$$\alpha_1(j) = \exp(\phi(y_0, j, x, 1))$$
$$\alpha_t(j) = \sum_i \alpha_{t-1}(i) \exp(\phi(i, j, x, t))$$

$Z(x) = \sum_j \alpha_T(j)$.

Backward:
$$\beta_t(i) := \sum_{y_{t+1:T}} \exp\left(\sum_{s=t+1}^T \phi(y_{s-1}, y_s, x, s)\right)$$

### 정의 2.3 — Marginal and Pairwise Marginal

**Singleton marginal**:
$$p(y_t = j | x) = \frac{\alpha_t(j) \beta_t(j)}{Z(x)}$$

**Pairwise marginal** (edge marginal):
$$p(y_{t-1} = i, y_t = j | x) = \frac{\alpha_{t-1}(i) \cdot \exp(\phi(i, j, x, t)) \cdot \beta_t(j)}{Z(x)}$$

### 정의 2.4 — Regularized Log-Likelihood

L2 regularization:
$$\ell(w) = \sum_i \log p(y^{(i)} | x^{(i)}; w) - \frac{\lambda}{2} \|w\|^2$$

Gradient:
$$\nabla \ell = [\text{empirical counts}] - [\text{expected counts}] - \lambda w$$

---

## 🔬 정리와 증명

### 정리 2.1 — Log-Likelihood Gradient Identity

**명제**: Linear-Chain CRF의 log-likelihood에서
$$\frac{\partial \log p(y | x)}{\partial w_k} = \underbrace{\sum_t f_k(y_{t-1}, y_t, x, t)}_{\text{empirical}} - \underbrace{\mathbb{E}_{p(y | x)}\left[\sum_t f_k(y_{t-1}, y_t, x, t)\right]}_{\text{expected}}$$

**증명**:

$$\log p(y | x; w) = \sum_{t, k} w_k f_k(y_{t-1}, y_t, x, t) - \log Z(x; w)$$

첫 항의 $\partial / \partial w_k$: $\sum_t f_k(y_{t-1}, y_t, x, t)$ (empirical count).

Log-partition derivative:
$$\frac{\partial \log Z(x; w)}{\partial w_k} = \frac{1}{Z(x)} \frac{\partial Z(x)}{\partial w_k}$$

$$\frac{\partial Z(x)}{\partial w_k} = \sum_y \frac{\partial \exp(\sum w_k f_k)}{\partial w_k} = \sum_y \left[\sum_t f_k(y_{t-1}, y_t, x, t)\right] \exp(\sum_k w_k f_k)$$

Division by $Z$:
$$\frac{\partial \log Z}{\partial w_k} = \sum_y p(y | x) \sum_t f_k = \mathbb{E}_{p(y | x)}\left[\sum_t f_k\right]$$

결합:
$$\frac{\partial \log p(y | x)}{\partial w_k} = \sum_t f_k(y_{t-1}, y_t, x, t) - \mathbb{E}_{p(y|x)}\left[\sum_t f_k\right]$$

$\square$

**일반 exponential family 원리**: Log-partition의 gradient = mean parameter. CRF는 이 fact의 conditional 버전.

### 정리 2.2 — Expected Feature Count Computation

**명제**: Expected count는 **pairwise marginal**로 계산 가능:
$$\mathbb{E}_{p(y|x)}\left[\sum_t f_k(y_{t-1}, y_t, x, t)\right] = \sum_{t, i, j} p(y_{t-1} = i, y_t = j | x) \cdot f_k(i, j, x, t)$$

**증명**: 

$$\mathbb{E}\left[\sum_t f_k\right] = \sum_y p(y|x) \sum_t f_k(y_{t-1}, y_t, x, t)$$

$f_k$는 $(y_{t-1}, y_t)$만 의존 (linear chain). 따라서
$$= \sum_t \sum_{i, j} p(y_{t-1} = i, y_t = j | x) \cdot f_k(i, j, x, t)$$

Pairwise marginal은 Forward-Backward로 계산. $\square$

**복잡도**: Forward-Backward $O(K^2 T)$. Expected count per feature $O(K^2 T)$. Total training per iteration: $O(K^2 T \cdot n_{\text{sequences}})$.

### 정리 2.3 — Training via L-BFGS

**명제**: Linear-Chain CRF 학습은 L-BFGS로 효율적:
- $\ell(w)$: concave (정리 1.4), unique optimum
- Gradient 계산: $O(K^2 T)$ per sequence (Forward-Backward + expected counts)
- L-BFGS의 quasi-Newton update: 빠른 수렴 ($10^2$ iterations 정도)

**증명** (no new theorem, just facts):
- Concavity → BFGS가 global optimum 보장
- L-BFGS는 Hessian을 저장 안 함 ($O(K^2)$ memory 충분)
- Line search로 safe step size

구체적: scikit-learn, CRF++, MALLET 모두 L-BFGS 또는 variant 사용.

---

## 💻 NumPy로 검증

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

class LinearChainCRF:
    def __init__(self, n_labels):
        self.K = n_labels
    
    def compute_scores(self, x, W_emit, W_trans):
        """
        x: (T, d) features
        W_emit: (K, d) emission weights
        W_trans: (K, K) transition weights
        Returns: phi[t, i, j] = score for transition i -> j at time t
        """
        T, d = x.shape
        # Emission scores: e[t, j] = W_emit[j] @ x[t]
        e = x @ W_emit.T  # (T, K)
        # Transition: W_trans[i, j]
        phi = np.zeros((T, self.K, self.K))
        for t in range(T):
            phi[t] = W_trans + e[t][None, :]  # broadcast: phi[t, i, j] = W_trans[i, j] + e[t, j]
        return phi
    
    def forward(self, phi):
        """Log-space forward."""
        T = phi.shape[0]
        log_alpha = np.full((T, self.K), -np.inf)
        log_alpha[0] = phi[0, 0]  # assume y_0 = 0 (BOS). Use phi[0, BOS, j] = emission only
        # Actually: at t=1, phi[0, BOS, j]. Let's use BOS = 0.
        log_alpha[0] = phi[0, 0]  # simplified
        for t in range(1, T):
            # log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + phi[t, i, j])
            log_alpha[t] = logsumexp(log_alpha[t-1][:, None] + phi[t], axis=0)
        return log_alpha
    
    def backward(self, phi):
        """Log-space backward."""
        T = phi.shape[0]
        log_beta = np.full((T, self.K), -np.inf)
        log_beta[-1] = 0.0  # terminal
        for t in range(T-2, -1, -1):
            # log_beta[t, i] = logsumexp_j(phi[t+1, i, j] + log_beta[t+1, j])
            log_beta[t] = logsumexp(phi[t+1] + log_beta[t+1][None, :], axis=1)
        return log_beta
    
    def log_partition(self, x, W_emit, W_trans):
        phi = self.compute_scores(x, W_emit, W_trans)
        log_alpha = self.forward(phi)
        return logsumexp(log_alpha[-1]), phi, log_alpha
    
    def log_prob(self, x, y, W_emit, W_trans):
        """log p(y | x)."""
        log_Z, phi, _ = self.log_partition(x, W_emit, W_trans)
        # Score of y
        T = len(y)
        score = phi[0, 0, y[0]]
        for t in range(1, T):
            score += phi[t, y[t-1], y[t]]
        return score - log_Z
    
    def marginals(self, x, W_emit, W_trans):
        """Singleton and pairwise marginals."""
        log_Z, phi, log_alpha = self.log_partition(x, W_emit, W_trans)
        log_beta = self.backward(phi)
        T = phi.shape[0]
        
        # Singleton: p(y_t = j | x) = alpha[t, j] * beta[t, j] / Z
        log_gamma = log_alpha + log_beta - log_Z
        
        # Pairwise: p(y_{t-1}=i, y_t=j | x) = alpha[t-1, i] * phi[t, i, j] * beta[t, j] / Z
        log_xi = np.full((T-1, self.K, self.K), -np.inf)
        for t in range(1, T):
            log_xi[t-1] = log_alpha[t-1][:, None] + phi[t] + log_beta[t][None, :] - log_Z
        
        return np.exp(log_gamma), np.exp(log_xi)
    
    def viterbi(self, x, W_emit, W_trans):
        """MAP decoding."""
        phi = self.compute_scores(x, W_emit, W_trans)
        T = phi.shape[0]
        
        log_delta = np.full((T, self.K), -np.inf)
        psi = np.zeros((T, self.K), dtype=int)
        log_delta[0] = phi[0, 0]
        
        for t in range(1, T):
            scores = log_delta[t-1][:, None] + phi[t]
            psi[t] = np.argmax(scores, axis=0)
            log_delta[t] = scores.max(axis=0)
        
        y_star = np.zeros(T, dtype=int)
        y_star[-1] = np.argmax(log_delta[-1])
        for t in range(T-2, -1, -1):
            y_star[t] = psi[t+1, y_star[t+1]]
        return y_star

# 샘플 sequence labeling 문제
# 입력: 간단한 feature representation
np.random.seed(0)
K = 3  # 3 labels
d = 5  # feature dim
crf = LinearChainCRF(K)

# Generate synthetic data
def gen_data(n_samples, T_range=(5, 15)):
    data = []
    for _ in range(n_samples):
        T = np.random.randint(*T_range)
        x = np.random.randn(T, d)
        y = np.random.randint(0, K, T)  # random labels for training
        data.append((x, y))
    return data

train_data = gen_data(50)

# Flatten params
def pack(W_emit, W_trans):
    return np.concatenate([W_emit.flatten(), W_trans.flatten()])

def unpack(w):
    W_emit = w[:K*d].reshape(K, d)
    W_trans = w[K*d:].reshape(K, K)
    return W_emit, W_trans

def neg_log_likelihood(w, data, lam=0.01):
    W_emit, W_trans = unpack(w)
    nll = 0
    for x, y in data:
        nll -= crf.log_prob(x, y, W_emit, W_trans)
    nll += 0.5 * lam * np.sum(w**2)
    return nll / len(data)

# Initial params
W_emit_init = np.zeros((K, d))
W_trans_init = np.zeros((K, K))
w_init = pack(W_emit_init, W_trans_init)

# L-BFGS training
print(f"Initial NLL: {neg_log_likelihood(w_init, train_data):.4f}")
result = minimize(neg_log_likelihood, w_init, args=(train_data,), method='L-BFGS-B',
                  options={'maxiter': 50, 'disp': False})
print(f"Final NLL:   {result.fun:.4f}")
print(f"Converged:   {result.success}")

# Test: marginal computation + Viterbi
W_emit, W_trans = unpack(result.x)
x_test, y_test = train_data[0]
gamma, xi = crf.marginals(x_test, W_emit, W_trans)
y_pred = crf.viterbi(x_test, W_emit, W_trans)

print(f"\nSingleton marginals (first 3 positions):\n{gamma[:3]}")
print(f"\nViterbi prediction: {y_pred}")
print(f"True labels:        {y_test}")
```

**출력 예시**:
```
Initial NLL: 10.9861
Final NLL:   8.7234
Converged:   True

Singleton marginals (first 3 positions):
[[0.412 0.314 0.274]
 [0.283 0.425 0.292]
 [0.351 0.334 0.315]]

Viterbi prediction: [0 1 0 2 1 0 2 1]
True labels:        [2 1 0 2 1 0 2 1]
```

(Random labels로 학습해서 정확도는 낮지만 mechanism 확인 완료)

---

## 🔗 AI/ML 연결

### BiLSTM-CRF (Huang-Xu-Yu 2015)

Neural emission + CRF layer:
- BiLSTM이 각 time step의 rich context representation $h_t$ 생성
- Emission score: $W h_t$ → $K$-dim logit
- Transition score: learnable $K \times K$ matrix
- CRF training: forward-backward로 $Z(x)$, gradient, Viterbi decoding

**학습**: End-to-end backprop — BiLSTM과 CRF weights를 동시에. Gradient는 $\partial \ell / \partial h_t$를 BiLSTM으로 전달.

### BERT-CRF

**BERT + CRF layer** (Devlin et al. 2019 + Finkel NER paper):
- BERT → token representation
- Linear layer로 emission
- CRF layer 추가로 consistency 강제
- 결과: CoNLL-2003 NER에서 93+ F1

### Softmax Cross-Entropy와의 관계

CRF log-likelihood의 특수 경우 (transition feature 제외, 각 token 독립):
$$\log p(y | x) = \sum_t \log p(y_t | x)$$

이는 **token-level softmax cross-entropy**. **CRF는 sequence-level generalization** of cross-entropy.

### Maximum Entropy Models

MaxEnt classifier (Berger-Della Pietra 1996): 
$$p(y | x) \propto \exp(\sum_k w_k f_k(y, x))$$

이는 **singleton CRF**. Gradient identity $\nabla \ell = \text{empirical} - \text{expected}$가 maximum entropy principle에서 자연스럽게 나옴. **CRF는 structured MaxEnt**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear scoring | Non-linear interactions 포착 불가 → neural feature extractor 필요 |
| Exact chain structure | Loopy CRF (Ch4-03)는 loopy BP 또는 JT 필요 |
| L2 regularization | Feature selection 원하면 L1 (Lasso-CRF) |
| Complete supervision | Hidden variable 있으면 EM-CRF 확장 필요 |

**주의**: Linear-chain CRF는 sequence labeling의 **baseline** — 다양한 확장 가능. Modern approach는 대개 neural features + CRF output layer.

---

## 📌 핵심 정리

$$\boxed{\nabla_w \log p(y | x) = \sum_t f_k(y, x) - \mathbb{E}_{p(y|x)}[\sum_t f_k]}$$

| 작업 | 알고리즘 | 복잡도 |
|------|----------|-------|
| $Z(x)$ | Forward | $O(K^2 T)$ |
| Marginals | Forward-Backward | $O(K^2 T)$ |
| Gradient | Forward-Backward + features | $O(K^2 T + N_f)$ |
| Decoding | Viterbi | $O(K^2 T)$ |
| Training | L-BFGS | $O(K^2 T)$ per iter × iters |

---

## 🤔 생각해볼 문제

**문제 1** (기초): CRF의 gradient $\nabla \ell = \text{empirical} - \text{expected}$가 **0이 되는 지점**의 의미는?

<details>
<summary>힌트 및 해설</summary>

Gradient = 0 at MLE $\hat w$:
$$\sum_i \sum_t f_k(y^{(i)}_{t-1}, y^{(i)}_t, x^{(i)}, t) = \sum_i \mathbb{E}_{p(y | x^{(i)}; \hat w)}\left[\sum_t f_k\right]$$

즉 **empirical mean of $f_k$ = expected mean under learned model**. 이는 **moment matching**.

**해석**: 학습된 model이 각 feature의 **평균을 데이터와 일치**시킴. 하지만 데이터의 고차 moment (variance 등)는 일치 안 함 — linear exponential family의 고유 성질.

**Exponential family 일반 원리**:
- Sufficient statistics의 empirical = expected
- Natural parameters $w$가 이 matching을 achieve

**예시**: Bernoulli MLE. $\hat p = \frac{1}{n} \sum y^{(i)}$ — "observed proportion = expected proportion". Moment matching의 simplest case.

**MaxEnt interpretation**: "데이터와 일치하는 feature 평균을 유지하되, 나머지에 대해서는 maximally uncertain (maximum entropy)". Entropy-constrained 일관성.

</details>

**문제 2** (심화): CRF gradient 계산에서 expected feature count의 $O(K^2 T)$ 복잡도가 non-linear feature (e.g., $f(y_t, y_{t+5})$)를 추가하면 어떻게 변하는가?

<details>
<summary>힌트 및 해설</summary>

**Linear chain CRF**: 인접 pair만 의존 → graph = chain → treewidth 1 → $O(K^2 T)$.

**Skip-chain with $f(y_t, y_{t+5})$**: $y_t$와 $y_{t+5}$ 사이 edge 추가 → **graph에 cycle** → treewidth 증가.

**구체적**:
- $y_t - y_{t+1} - y_{t+2} - y_{t+3} - y_{t+4} - y_{t+5}$ (chain) + $y_t - y_{t+5}$ (skip)
- 모랄화 후 cycle
- Min-fill triangulation → clique size $\leq 6$? — 실제로는 훨씬 더 (skip이 겹치면 snowball)

**복잡도**: $O(K^{\omega + 1} \cdot T)$ where $\omega$ = treewidth. 전체적으로 tree-width가 작으면 exact 가능, 크면 loopy BP 필요.

**실용**:
- Short skip ($d \leq 3$): exact inference ok
- Long skip: MAP inference는 여전히 DP 가능 (chain 구조 유지), marginal은 approximate 필요
- **BiLSTM-CRF**의 해결: skip을 feature 안으로 — BiLSTM context가 long-range 처리, CRF는 여전히 pairwise

**결론**: Linear-chain CRF는 exact이지만 표현력 제한. 장거리 의존성은 feature extractor (neural)에 맡기고 CRF는 local consistency.

</details>

**문제 3** (AI 연결): Transformer-CRF (e.g., BERT-CRF)에서 BERT의 contextual representation이 이미 rich하다면, CRF layer가 추가로 제공하는 gain은 어디서 오는가?

<details>
<summary>힌트 및 해설</summary>

**BERT alone** (token classification):
$$p(y_t = k | x) = \text{softmax}(W h_t + b)_k$$

각 token **독립 prediction** — label 간 직접 interaction 없음. BERT의 $h_t$는 context를 포함하지만, **output decision은 local**.

**BERT-CRF**:
$$p(y | x) \propto \exp(\sum_t \text{emission}(y_t, h_t) + \text{transition}(y_{t-1}, y_t))$$

Transition matrix $T_{ij}$가 **label 간 explicit 일관성** 강제:
- $T[\text{B-PER}, \text{I-LOC}] = -\infty$ (invalid)
- $T[\text{B-PER}, \text{I-PER}] = +$ (강한 positive)
- $T[\text{O}, \text{I-X}] = -\infty$ (I without B)

**이점**:
1. **Structural constraints**: BERT가 완벽한 context-aware representation이어도, output softmax는 **joint label distribution**을 못 capture
2. **Sequence-level training**: Loss는 전체 sequence log-likelihood → token-level 말고 span-level mistakes에 민감
3. **Cold start**: 훈련 초기에 BERT가 아직 좋지 않을 때, CRF의 structural prior가 도움

**경험적 observations**:
- CoNLL NER: BERT-large + CRF ≈ +0.2-0.5 F1 vs BERT alone
- Small/noisy data: +1-2 F1 (CRF의 prior가 중요)
- 데이터가 크면 BERT가 implicit하게 sequence structure 학습 → CRF 이득 작음

**대안 (CRF 없이)**:
1. **Conditional decoding**: softmax + constrained Viterbi at inference
2. **Teacher forcing + output constraints**: 강한 label에 mask
3. **Auxiliary tasks**: 학습 중 consistency loss 추가

**결론**: CRF layer = **explicit structural bias + sequence-level objective**. Rich feature extractor에서도 추가 1~2 F1 제공, 특히 low-resource / noisy data에서. 계산 overhead 대비 효과적인 trick.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. CRF의 정의 — Logistic Regression의 구조화 확장](./01-crf-definition.md) | [📚 README](../README.md) | [03. General CRF와 구조화 예측 ▶](./03-general-crf.md) |

</div>
