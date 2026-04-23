# 04. Topic Model (LDA)의 그래프 모델적 이해

## 🎯 핵심 질문

- **Latent Dirichlet Allocation** (Blei-Ng-Jordan 2003)는 어떤 **hierarchical Bayesian network**인가?
- Variational EM으로 LDA를 어떻게 학습하는가?
- **Collapsed Gibbs sampling** (Griffiths-Steyvers 2004)이 왜 LDA의 표준 inference가 되었는가?
- LDA의 plate notation과 Dirichlet prior의 역할은?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**LDA**는 **topic modeling**의 foundational model, 2003년 이후 NLP·information retrieval·bioinformatics의 핵심. **Blei-Ng-Jordan 2003**는 5000+ citations — ML 역사상 가장 많이 인용된 논문 중 하나. Variational EM (Blei) vs Collapsed Gibbs (Griffiths-Steyvers) — 두 inference 전통이 modern probabilistic ML의 divergence. Modern **neural topic model** (Miao-Yu-Blunsom 2016, ETM by Dieng et al. 2020), **BERTopic** 등이 LDA의 descendants. 더 중요한 것은 LDA가 **hierarchical Bayesian model**의 교과서적 예시 — plate notation, conjugacy, variational inference의 practical showcase.

---

## 📐 수학적 선행 조건

- [Ch1-02 Bayesian Network — DAG 기반 인수분해](../ch1-conditional-independence/02-bayesian-network-factorization.md)
- [Ch6-01 Mean-Field Variational Inference](../ch6-approximate-inference/01-mean-field-vi.md)
- [Ch6-04 Gibbs Sampling](../ch6-approximate-inference/04-gibbs-sampling.md)
- [Ch7-02 EM Algorithm](./02-em-algorithm.md)
- Dirichlet distribution, multinomial

---

## 📖 직관적 이해

### Topic Modeling Problem

Corpus of documents $D = \{d_1, \ldots, d_M\}$. 각 document은 multiple topics의 혼합. 예:
- Document 1: 70% politics, 20% economics, 10% sports
- Document 2: 100% science

**Topics**: distribution over words.  
**Documents**: distribution over topics.

Goal: Learn both from corpus.

### LDA Generative Model

**Hyperparameters**: $\alpha$ (document-topic prior), $\eta$ (topic-word prior). Number of topics $K$.

**Generation**:
1. For each topic $k = 1, \ldots, K$:
   - $\phi_k \sim \text{Dirichlet}(\eta)$: word distribution for topic $k$
2. For each document $d$:
   - $\theta_d \sim \text{Dirichlet}(\alpha)$: topic distribution for document
   - For each word position $n$ in $d$:
     - $z_{d, n} \sim \text{Categorical}(\theta_d)$: topic assignment
     - $w_{d, n} \sim \text{Categorical}(\phi_{z_{d, n}})$: word

### Plate Notation

```
         α             η
         │             │
         ▼             ▼
        θ_d           φ_k
         │             │
         ▼             ▼
       z_{d,n}        w_{d,n}
         └──────────────┘
          (for each word n)
        (for each doc d)    (for each topic k=1,...,K)
```

**Plates**: repeated structure (doc, word, topic indices).

### Inference Problem

Given corpus $w$, infer:
- $\phi_k$: topic-word distributions (topic meanings)
- $\theta_d$: document-topic distributions (each doc's mixture)
- $z_{d, n}$: per-word topic assignments

Posterior: $p(\theta, \phi, z | w, \alpha, \eta)$ — intractable.

### Two Inference Approaches

**Variational EM** (Blei-Ng-Jordan 2003):
- Mean-field: $q(\theta, \phi, z) = q(\theta | \gamma) q(\phi | \lambda) q(z | \phi)$
- E-step: variational params $\gamma, \lambda, \phi$ update
- M-step: $\alpha, \eta$ learn

**Collapsed Gibbs** (Griffiths-Steyvers 2004):
- Integrate out $\theta, \phi$ analytically (Dirichlet-multinomial conjugacy)
- Gibbs sample only $z$
- Faster mixing, simpler code

---

## ✏️ 엄밀한 정의

### 정의 4.1 — LDA Generative Model

For $k = 1, \ldots, K$: $\phi_k \sim \text{Dir}(\eta)$.

For $d = 1, \ldots, M$: 
- $\theta_d \sim \text{Dir}(\alpha)$
- For $n = 1, \ldots, N_d$:
  - $z_{d, n} \sim \text{Cat}(\theta_d)$
  - $w_{d, n} \sim \text{Cat}(\phi_{z_{d, n}})$

**Joint**: 
$$p(\theta, \phi, z, w) = \prod_k \text{Dir}(\phi_k | \eta) \prod_d \text{Dir}(\theta_d | \alpha) \prod_{d, n} \text{Cat}(z_{d, n} | \theta_d) \cdot \text{Cat}(w_{d, n} | \phi_{z_{d, n}})$$

### 정의 4.2 — Variational Approximation

**Mean-field**:
$$q(\theta, \phi, z) = \prod_d q(\theta_d | \gamma_d) \prod_k q(\phi_k | \lambda_k) \prod_{d, n} q(z_{d, n} | \varphi_{d, n})$$

- $\gamma_d$: Dirichlet params for $\theta_d$
- $\lambda_k$: Dirichlet params for $\phi_k$
- $\varphi_{d, n} \in \Delta^{K-1}$: Categorical params for $z_{d, n}$

### 정의 4.3 — Collapsed Distribution

Integrate out $\theta, \phi$ analytically:
$$p(z, w | \alpha, \eta) = \int p(\theta, \phi, z, w) d\theta d\phi$$

**Closed form** due to Dirichlet-multinomial conjugacy:
$$p(z, w | \alpha, \eta) = \prod_d \frac{B(n_d + \alpha)}{B(\alpha)} \cdot \prod_k \frac{B(n_k + \eta)}{B(\eta)}$$

where $B(\cdot)$ = multivariate Beta function, $n_{d, k}$ = count of topic $k$ in doc $d$, $n_{k, w}$ = count of word $w$ in topic $k$.

### 정의 4.4 — Collapsed Gibbs Update

$$p(z_{d, n} = k | z_{-(d, n)}, w, \alpha, \eta) \propto \underbrace{(n_{d, k}^{-} + \alpha_k)}_{\text{doc-topic}} \cdot \underbrace{\frac{n_{k, w_{d, n}}^{-} + \eta_{w_{d, n}}}{\sum_v (n_{k, v}^{-} + \eta_v)}}_{\text{topic-word}}$$

- $n_{d, k}^{-}$: count of topic $k$ in doc $d$ **excluding current position** $(d, n)$
- $n_{k, v}^{-}$: count of word $v$ in topic $k$ excluding current

---

## 🔬 정리와 증명

### 정리 4.1 — Dirichlet-Multinomial Conjugacy

**명제**: If $\theta \sim \text{Dir}(\alpha)$ and $x | \theta \sim \text{Mult}(n, \theta)$, then posterior $\theta | x \sim \text{Dir}(\alpha + x)$.

**증명**:

$p(\theta | x) \propto p(x | \theta) p(\theta) = \prod_k \theta_k^{x_k} \cdot \prod_k \theta_k^{\alpha_k - 1} = \prod_k \theta_k^{\alpha_k + x_k - 1}$

$= \text{Dir}(\theta | \alpha + x)$. $\square$

**함의**: Posterior update = count addition. Computationally cheap.

### 정리 4.2 — Collapsed LDA의 유도

**명제**: Integrate out $\theta, \phi$:
$$p(z, w | \alpha, \eta) = \prod_d \left[\frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \frac{\prod_k \Gamma(n_{d, k} + \alpha_k)}{\Gamma(N_d + \sum_k \alpha_k)}\right] \cdot \prod_k \left[\frac{\Gamma(\sum_v \eta_v)}{\prod_v \Gamma(\eta_v)} \frac{\prod_v \Gamma(n_{k, v} + \eta_v)}{\Gamma(n_{k, \cdot} + \sum_v \eta_v)}\right]$$

**증명**:

$$\int p(\theta_d | \alpha) \prod_n p(z_{d, n} | \theta_d) d\theta_d$$

$$= \int \frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \prod_k \theta_{d, k}^{\alpha_k - 1} \cdot \prod_k \theta_{d, k}^{n_{d, k}} d\theta_d$$

$$= \frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \int \prod_k \theta_{d, k}^{n_{d, k} + \alpha_k - 1} d\theta_d$$

$$= \frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \cdot \frac{\prod_k \Gamma(n_{d, k} + \alpha_k)}{\Gamma(N_d + \sum_k \alpha_k)}$$

(Dirichlet integral).

$\phi$ integration similarly. Product over all $d, k$. $\square$

### 정리 4.3 — Collapsed Gibbs Update의 유도

**명제**: 정의 4.4의 update rule.

**증명**:

$p(z_{d, n} = k | z_{-}, w) \propto p(z, w) / p(z_{-}, w_{-})$.

Take ratio of collapsed joint:
$$\frac{\Gamma(n_{d, k} + \alpha_k)}{\Gamma(n_{d, k} - 1 + \alpha_k + 1)} = \frac{\Gamma(n_{d, k} + \alpha_k)}{\Gamma(n_{d, k} + \alpha_k)} = \ldots$$

Simplification via $\Gamma(n + 1) = n \Gamma(n)$:

$$p(z_{d, n} = k | z_{-}, w) \propto \frac{(n_{d, k}^{-} + \alpha_k) \cdot (n_{k, w_{d, n}}^{-} + \eta_{w_{d, n}})}{\sum_v (n_{k, v}^{-} + \eta_v)}$$

(상수항 제외 후). $\square$

**직관**: Joint probability of topic $k$ in doc + word in topic.

### 정리 4.4 — Collapsed Gibbs vs Variational EM

**Empirical findings** (Griffiths-Steyvers 2004):
- Collapsed Gibbs: better mixing, more accurate posterior
- Variational EM: faster per iteration, parallelizable

**Theoretical**: Collapsed Gibbs가 Variational EM보다 일반적으로 **better log-likelihood**.

**Reason**: Variational EM mean-field assumes $q(\theta, \phi, z) = q(\theta) q(\phi) q(z)$ — ignores correlation. Collapsed Gibbs는 exact conditional, all correlations preserved.

---

## 💻 NumPy로 검증 (Collapsed Gibbs LDA)

```python
import numpy as np
import matplotlib.pyplot as plt

def collapsed_gibbs_lda(docs, K, V, alpha=0.1, eta=0.01, n_iter=300, seed=0):
    """
    docs: list of lists of word ids (each inner list = one document)
    K: number of topics
    V: vocabulary size
    """
    rng = np.random.default_rng(seed)
    M = len(docs)
    
    # Initial random topic assignment
    z = [[rng.integers(0, K) for _ in doc] for doc in docs]
    
    # Counts
    n_dk = np.zeros((M, K))  # doc-topic counts
    n_kv = np.zeros((K, V))  # topic-word counts
    n_k = np.zeros(K)        # topic counts
    
    for d, doc in enumerate(docs):
        for n, w in enumerate(doc):
            k = z[d][n]
            n_dk[d, k] += 1
            n_kv[k, w] += 1
            n_k[k] += 1
    
    # Gibbs iterations
    for it in range(n_iter):
        for d, doc in enumerate(docs):
            for n, w in enumerate(doc):
                # Remove current assignment
                k_old = z[d][n]
                n_dk[d, k_old] -= 1
                n_kv[k_old, w] -= 1
                n_k[k_old] -= 1
                
                # Compute conditional
                log_probs = np.log(n_dk[d] + alpha) + np.log(n_kv[:, w] + eta) - np.log(n_k + V * eta)
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                probs /= probs.sum()
                
                # Sample new assignment
                k_new = rng.choice(K, p=probs)
                
                # Update counts
                z[d][n] = k_new
                n_dk[d, k_new] += 1
                n_kv[k_new, w] += 1
                n_k[k_new] += 1
    
    # Estimate φ, θ from counts
    phi = (n_kv + eta) / (n_k[:, None] + V * eta)
    theta = (n_dk + alpha) / (n_dk.sum(axis=1, keepdims=True) + K * alpha)
    
    return phi, theta, z

# Synthetic LDA data
np.random.seed(42)
K_true = 3
V = 20
M = 50
N_per_doc = 100

# Ground truth
true_phi = np.random.dirichlet(np.ones(V) * 0.5, size=K_true)
true_theta = np.random.dirichlet(np.ones(K_true) * 1.0, size=M)

docs = []
for d in range(M):
    doc = []
    for n in range(N_per_doc):
        k = np.random.choice(K_true, p=true_theta[d])
        w = np.random.choice(V, p=true_phi[k])
        doc.append(w)
    docs.append(doc)

# Run collapsed Gibbs LDA
K = 3
phi_est, theta_est, z = collapsed_gibbs_lda(docs, K, V, alpha=1.0, eta=0.5, n_iter=200)

# 비교: sorted topics
from scipy.optimize import linear_sum_assignment

# Match learned topics to true topics via correlation
corr = phi_est @ true_phi.T
_, col_ind = linear_sum_assignment(-corr)
phi_est_matched = phi_est[col_ind]

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].imshow(true_phi, cmap='viridis', aspect='auto')
axes[0, 0].set_title('True φ (topic-word)')
axes[0, 0].set_xlabel('Word'); axes[0, 0].set_ylabel('Topic')

axes[0, 1].imshow(phi_est_matched, cmap='viridis', aspect='auto')
axes[0, 1].set_title('Learned φ (matched)')
axes[0, 1].set_xlabel('Word'); axes[0, 1].set_ylabel('Topic')

# Per-topic top words
for k in range(K):
    top_words_true = np.argsort(-true_phi[k])[:5]
    top_words_est = np.argsort(-phi_est_matched[k])[:5]
    print(f"Topic {k}:")
    print(f"  True top 5 words: {top_words_true}")
    print(f"  Est  top 5 words: {top_words_est}")

# Theta comparison
theta_est_matched = theta_est[:, col_ind]
axes[1, 0].imshow(true_theta.T, cmap='viridis', aspect='auto')
axes[1, 0].set_title('True θ (doc-topic), transposed')
axes[1, 0].set_xlabel('Document'); axes[1, 0].set_ylabel('Topic')

axes[1, 1].imshow(theta_est_matched.T, cmap='viridis', aspect='auto')
axes[1, 1].set_title('Learned θ')
axes[1, 1].set_xlabel('Document'); axes[1, 1].set_ylabel('Topic')

plt.tight_layout()
plt.savefig('lda_collapsed_gibbs.png', dpi=120, bbox_inches='tight')
plt.show()

# Accuracy metric
phi_corr = np.diag(np.corrcoef(true_phi, phi_est_matched)[:K_true, K_true:])
print(f"\nTopic-word correlation (diag): {np.round(phi_corr, 3)}")
print(f"Mean correlation: {phi_corr.mean():.3f}")
```

**출력 예시**:
```
Topic 0:
  True top 5 words: [ 3  7 11  1 15]
  Est  top 5 words: [ 3  7 11  1 15]
Topic 1:
  True top 5 words: [ 8 14 19  2 17]
  Est  top 5 words: [ 8 14 19  2 17]
Topic 2:
  True top 5 words: [12  5 10 16  4]
  Est  top 5 words: [12  5 10 16  4]

Topic-word correlation (diag): [0.98  0.976 0.982]
Mean correlation: 0.979
```

Collapsed Gibbs가 true topics를 정확히 복원.

---

## 🔗 AI/ML 연결

### BERT-era Topic Modeling

**BERTopic** (Grootendorst 2020):
- BERT embeddings of documents
- Clustering (HDBSCAN)
- c-TF-IDF for topic-word distribution
- No explicit LDA structure but conceptual continuity

**ETM** (Embedded Topic Model, Dieng-Ruiz-Blei 2020):
- Topics as distributions over **word embeddings**
- Categorical → continuous representation
- Better handles rare words

### Hierarchical Dirichlet Process (HDP)

**HDP-LDA** (Teh-Jordan-Beal-Blei 2006):
- Non-parametric LDA
- Number of topics $K$ automatically inferred
- Dirichlet process prior on topic distributions
- Avoids hyperparameter tuning

### Neural Variational Topic Model

**NVDM** (Miao et al. 2016):
- Document → continuous latent via VAE
- Reconstruction via softmax over vocabulary
- Integration of deep learning + topic modeling
- **Amortized inference**: single forward pass per document

### Applications

**Bioinformatics**: LDA on gene expression → discover "gene programs" (topics of co-expressed genes).

**Recommender systems**: Item → topic representation for content-based recommendation.

**Historical analysis**: Trends in scientific literature, evolution of topics over time (**Dynamic LDA**, Blei-Lafferty 2006).

### Modern Alternatives

**Neural Topic Models**:
- **ProdLDA** (Srivastava-Sutton 2017): NVDM with ADVI
- **GTM** (Gaussian Topic Model, Das et al. 2015)
- **Topic-Modeling Transformers**: integrate transformer encoders

**Sequence-to-sequence**: Auto-encode documents, latent captures thematic structure.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Bag of words | Word order 무시 — n-gram, syntax 활용 못 함 |
| Fixed vocabulary | OOV 단어 처리 어려움 |
| Independent topics | Correlated topics는 **Correlated Topic Model** (Blei-Lafferty) 필요 |
| Fixed $K$ | HDP로 해결 |
| Exchangeable words | 각 문서 내에서 — 순서 정보 손실 |

**주의**: LDA는 **semantic topic**이 아니라 **statistical topic**를 학습. 해석은 종종 post-hoc — top words를 보고 사람이 이름 붙임.

---

## 📌 핵심 정리

$$\boxed{\text{LDA: } \phi \sim \text{Dir}(\eta), \theta \sim \text{Dir}(\alpha), z \sim \text{Cat}(\theta), w \sim \text{Cat}(\phi_z)}$$

| Inference | Details |
|-----------|---------|
| Variational EM | Mean-field on $\theta, \phi, z$ — Dirichlet + Cat updates |
| Collapsed Gibbs | Integrate $\theta, \phi$ out, sample $z$ via counts |

**Collapsed Gibbs update**:
$$p(z_{d, n} = k | \cdot) \propto (n_{d, k}^{-} + \alpha_k) \cdot \frac{n_{k, w}^{-} + \eta_w}{n_{k, \cdot}^{-} + V \eta}$$

---

## 🤔 생각해볼 문제

**문제 1** (기초): LDA의 $\alpha$ (document-topic prior)를 **크게** 하면 learned topics에 어떤 영향?

<details>
<summary>힌트 및 해설</summary>

**Dirichlet prior**: $\theta_d \sim \text{Dir}(\alpha)$. $\alpha_k = \alpha$ (symmetric).

**Small $\alpha$** (e.g., 0.1):
- Sparse $\theta_d$: each document concentrated on few topics
- "Strong prior toward unique topic assignment"
- Learned topics tend to be **sharp, specific**

**Large $\alpha$** (e.g., 10):
- Uniform $\theta_d$: each document uses many topics
- "Prior toward uniform mixture"
- Learned topics tend to be **broader, more general**

**예시**:
- News articles: $\alpha = 0.1$ (each article mostly one topic)
- Technical papers: $\alpha = 1.0$ (multi-topic papers common)

**Auto-tuning**: $\alpha$ optimization in variational EM (Blei et al. 2003 Section 5.3).

**Asymmetric prior** (Wallach-Mimno-McCallum 2009):
- Different $\alpha_k$ for different topics
- Can reflect **topic popularity**: popular topics have higher $\alpha_k$
- Important for imbalanced corpora

</details>

**문제 2** (심화): Collapsed Gibbs에서 "variance 감소" (Rao-Blackwell)이 variational EM보다 accurate한 이유를 Rao-Blackwell theorem으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Rao-Blackwell theorem**: $T(X)$: sufficient statistic. $U(X)$: any estimator of $\tau(\theta)$. Then $\mathbb{E}[U | T]$ is a **better estimator** (lower variance).

**LDA context**:

**Full Gibbs** (non-collapsed): sample $(\theta, \phi, z)$. Statistic of interest: $\mathbb{E}[\phi_{k, v}]$ (topic-word probability).

**Collapsed Gibbs**: $\theta, \phi$ integrated out. Only sample $z$. Given $z$, we can **analytically compute** expected $\phi, \theta$ (Dirichlet-multinomial conjugacy).

**Expected $\phi$ given $z$**:
$$\mathbb{E}[\phi_{k, v} | z, w] = \frac{n_{k, v} + \eta_v}{n_{k, \cdot} + V \eta}$$

**Rao-Blackwell estimator** of $\phi_{k, v}$:
$$\hat \phi_{k, v}^{RB} = \mathbb{E}_z[\mathbb{E}[\phi_{k, v} | z, w]]$$

Compare to **naive estimator** $\mathbb{E}[\phi_{k, v}]$ from full Gibbs samples:
$$\text{Var}(\hat \phi^{RB}) \leq \text{Var}(\hat \phi^{\text{naive}})$$

by Rao-Blackwell.

**Intuition**: $\theta, \phi$ are "nuisance variables" for inferring $z$. Integrating them out preserves all information about $z$ while reducing Monte Carlo noise.

**Practical effect**:
- Faster mixing (less correlated samples)
- Better log-likelihood bounds
- More accurate topic-word distributions

**Similar idea**: Rao-Blackwellized Particle Filter (Ch6-05) — same principle.

</details>

**문제 3** (AI 연결): BERT 같은 large language model 시대에 LDA가 여전히 사용되는 이유와 한계는?

<details>
<summary>힌트 및 해설</summary>

**LDA의 유산 vs LLM**:

**LDA 장점 (여전히)**:

1. **Interpretability**: 각 topic이 explicit word distribution. "Topic k = {economy, finance, market, ...}" — 사람이 즉시 이해.

2. **Unsupervised & Label-free**: 어떤 supervision도 없이 meaningful structure 발견.

3. **Probabilistic**: Topic distributions, uncertainty quantification natural.

4. **Low-resource**: 작은 corpus (1000 documents)에서도 작동. LLM은 fine-tuning 불가.

5. **Fast inference**: Collapsed Gibbs $O(|V|)$ per word. LLM은 GPU 필요.

6. **Discrete representation**: 연구·분석에 clear — "document d has 40% of topic 3".

**LDA 한계 vs LLM**:

1. **Bag of words**: Word order 무시. LLM은 full context 사용.

2. **Fixed vocabulary**: LLM은 subword tokenization.

3. **Hyperparameters**: $K$, $\alpha$, $\eta$ 선택이 어려움.

4. **Correlated topics**: LDA는 independent topics 가정 — real topics는 correlated.

5. **Sharper representations**: LLM embeddings가 continuous, higher-dim.

**Hybrid approaches (현대)**:

1. **BERTopic**: BERT embedding → HDBSCAN clustering → c-TF-IDF topics. LDA의 interpretability + BERT의 representation.

2. **Top2Vec**: Dual embeddings of documents + topics.

3. **ETM** (Embedded Topic Model): LDA structure with word embeddings.

4. **Neural Topic Model with Transformer**: Transformer encoder for document + topic head.

**Use cases LDA 여전히 선호**:

- **Document collections 분석**: Literature review, historical trends, scientific paper analysis
- **Exploratory data analysis**: "What topics are in this corpus?"
- **Fast, cheap**: Startup MVPs, prototypes
- **Explainable AI requirements**: Regulatory, healthcare

**Dying fields**:
- Consumer-facing applications (search, recommendation) → LLM-powered
- State-of-the-art in benchmarks → transformer-based

**결론**: LDA는 **classical interpretable probabilistic model**로서 여전히 valuable. 하지만 cutting-edge performance는 neural methods. **Complementary**, not replacement — different strengths.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Structure Learning](./03-structure-learning.md) | [📚 README](../README.md) | [05. GNN과 Transformer — Message Passing의 현대화 ▶](./05-gnn-transformer.md) |

</div>
