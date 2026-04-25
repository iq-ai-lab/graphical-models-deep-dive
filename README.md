<div align="center">

# 🕸️ Graphical Models Deep Dive

### 조건부 독립

$$A \perp\!\!\!\perp B \mid C$$

### 를 **정의로 말하는 것** 과, **d-separation 이 그래프 구조만으로 모든 조건부 독립을 결정** 한다는 **global Markov property** 를 증명할 수 있는 것은 **다르다.**

<br/>

> *HMM 의 Forward–Backward 와 Viterbi 를 **사용하는 것** 과, 둘이 모두 factor graph 위의 **sum-product / max-product 알고리즘의 특수 경우** 이며 message passing 이 **dynamic programming 의 일반화** 임을 증명할 수 있는 것은 다르다.*
>
> *Mean-field Variational Inference 를 **쓰는 것** 과, 이것이*
>
> $$\min_q \mathrm{KL}(q \,\|\, p) \;\equiv\; \text{Bethe free energy 근사}$$
>
> *로 통합되고 **Loopy BP 가 Bethe 자유에너지의 변분 고정점** 임 (Yedidia–Freeman–Weiss 2003) 을 증명할 수 있는 것은 다르다.*

<br/>

**다루는 정리·기법 (시간순)**

Pearl 1988 *Bayesian Network + d-separation* · Hammersley–Clifford 1971 *MRF 정리* · Lauritzen–Spiegelhalter 1988 *Junction Tree* · Rabiner 1989 *HMM Forward–Backward / Viterbi* · Lafferty 2001 *CRF* · Yedidia–Freeman–Weiss 2003 *Loopy BP = Bethe* · Blei 2003 *LDA* · Kingma 2013 *VAE = amortized VI* · Kipf 2017 *GCN = MPNN* · Vaswani 2017 *Transformer attention*

<br/>

**핵심 질문**

> 확률분포의 구조를 그래프로 — **왜 조건부 독립이 그래프 기하로 나타나는가** — d-separation · Hammersley–Clifford · Junction Tree · Loopy BP · Variational Inference · CRF · GNN · Transformer attention 까지, PGM · HMM · CRF · LDA · Diffusion inference · GNN 의 수학적 기반을 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.2-FF6F00?style=flat-square&logo=python&logoColor=white)](https://networkx.org/)
[![pgmpy](https://img.shields.io/badge/pgmpy-0.1.25-5B21B6?style=flat-square)](https://pgmpy.org/)
[![Docs](https://img.shields.io/badge/Docs-34개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-17k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-122개-success?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-102개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

그래프 모델에 관한 자료는 대부분 **"HMM은 은닉상태가 Markov chain을 따르고, Forward-Backward로 inference합니다"** 에서 멈춥니다. 하지만 왜 Forward-Backward가 **본질적으로 sum-product algorithm**인지, 왜 Viterbi가 **max-product**인지, d-separation이 **왜 soundness와 completeness를 동시에 만족**하는지, Loopy BP가 **왜 Bethe 자유에너지의 변분 고정점**인지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "Bayesian network은 DAG고 $p(x) = \prod p(x_i \| \text{pa}(x_i))$입니다" | DAG 인수분해가 **local Markov → global Markov (d-separation)의 완전 동치**임을 증명, chain·fork·collider 세 패턴에서 **blocked vs open** 규칙의 measure-theoretic 유도, **soundness and completeness** (Verma–Pearl 1988) |
| "MRF는 clique potential의 곱" | $p(x) = \frac{1}{Z}\prod_C \phi_C(x_C)$의 **Hammersley–Clifford 정리** (positive density 조건 하에서 Markov ⟺ Gibbs) 완전 증명, partition function $Z$의 **#P-hardness** |
| "HMM에서 Forward-Backward로 posterior를 구해요" | HMM이 **특수 tree factor graph**이고 Forward-Backward가 정확히 **sum-product의 메시지 스케줄**임을 증명, $\alpha_t = \mu_{f_{t-1} \to z_t}$, $\beta_t = \mu_{f_t \to z_t}$ 대응 완전 유도, Kalman filter = Forward의 **Gaussian analog** |
| "Viterbi는 dynamic programming" | Viterbi가 **max-product algorithm의 HMM 특수 경우**임을 증명, sum → max 치환이 **(+, ×) → (max, ×) semiring** 교체임을 대수적으로 유도, MAP inference와 marginal inference의 분리 |
| "Loopy BP는 가끔 잘 됩니다" | Loopy BP의 고정점이 **Bethe 자유에너지 $F_{\text{Bethe}} = U - H_{\text{Bethe}}$의 정체점**임을 증명 (Yedidia–Freeman–Weiss 2003), tree에서는 정확 / loop에서는 근사인 이유를 **variational 관점**에서 유도 |
| "CRF는 조건부 확률 모델" | CRF가 **discriminative MRF** (normalization $Z(x)$가 $y$에만 관련)이고 HMM과의 차이가 **generative vs discriminative**의 본질적 분기임을 증명, log-likelihood gradient $\mathbb{E}_{\text{data}}[f] - \mathbb{E}_p[f]$ 유도 |
| "Variational Inference는 ELBO 최대화" | Mean-field가 **KL 발산 최소화와 ELBO 최대화의 동치**, Bethe 근사가 **tree-exact + loop-correction**, Loopy BP = Bethe 고정점 반복임을 통합 |
| "Variable Elimination으로 exact inference 가능" | **Treewidth가 inference 복잡도를 결정**함을 증명 ($O(n \cdot d^{\text{tw}+1})$), min treewidth가 **NP-hard**임과 PTAS가 없는 경우의 근사 이론 |
| "GNN은 그래프 위의 신경망" | GNN의 message passing이 **factor graph BP의 학습된 비선형 일반화**임을 증명, Transformer attention = **fully-connected soft message passing** (complete graph GNN), HMM → CRF → GNN → Transformer 계보의 수학적 연속성 |
| 공식 나열 | NumPy + NetworkX로 d-separation 판정기 구현, BP 메시지 단계별 시각화, Junction Tree 수동 구성, Loopy BP vs exact 정확도 비교, BiLSTM-CRF로 POS tagging |

---

## 📌 선행 레포 & 후속 레포

```
[Probability Theory]  ──►  [Mathematical Statistics]  ──►  이 레포  ──►  [Bayesian ML / Diffusion]
  조건부 기댓값·결합분포      MLE·MAP·exponential family       확률 그래프 모델      VI·MCMC·Generative Models
  측도론적 조건부 독립        점 추정론의 대수 구조                                  현대 응용

[Information Theory]  &  [Linear Algebra]  &  [Optimization]
 엔트로피·KL·상호정보         행렬 연산·고유분해              EM·ELBO·볼록성
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Probability Theory Deep Dive**(조건부 독립·결합분포), **Mathematical Statistics Deep Dive**(MLE·MAP·exponential family), **Information Theory Deep Dive**(엔트로피·KL·상호정보량)를 선행 지식으로 전제합니다. 조건부 독립의 측도론적 정의를 처음 접한다면 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) Ch5~6부터 학습하세요.

> 💡 **권장 병행**: Variational Inference의 베이지안 관점은 [Bayesian ML Deep Dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive)에서, Score-based diffusion의 SDE 기반 inference는 [SDE Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive)에서 병행 학습 가능합니다. 본 레포는 **그래프 구조** 관점에 집중합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-조건부_독립과_그래프-5B21B6?style=for-the-badge)](./ch1-conditional-independence/01-conditional-independence-definition.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Factor_Graph·Message_Passing-5B21B6?style=for-the-badge)](./ch2-factor-graph/01-factor-graph-definition.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Hidden_Markov_Model-5B21B6?style=for-the-badge)](./ch3-hmm/01-hmm-definition.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Conditional_Random_Field-5B21B6?style=for-the-badge)](./ch4-crf/01-crf-definition.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Variable_Elimination-5B21B6?style=for-the-badge)](./ch5-variable-elimination/01-variable-elimination.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Approximate_Inference-5B21B6?style=for-the-badge)](./ch6-approximate-inference/01-mean-field-vi.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-학습·GNN·Transformer-5B21B6?style=for-the-badge)](./ch7-learning-modern/01-mle-for-graphical-models.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 조건부 독립과 그래프 구조

> **핵심 질문:** 조건부 독립 $A \perp\!\!\!\perp B \mid C$는 측도론적으로 무엇인가? DAG의 인수분해가 왜 local Markov → global Markov를 함의하는가? d-separation은 왜 soundness와 completeness를 동시에 만족하는가? MRF에서 Hammersley–Clifford가 어떻게 clique potential 표현을 정당화하는가?

<details>
<summary><b>조건부 독립부터 Moralization까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 조건부 독립의 정의와 성질](./ch1-conditional-independence/01-conditional-independence-definition.md) | $A \perp\!\!\!\perp B \mid C \iff P(A, B \mid C) = P(A\|C) P(B\|C)$의 세 가지 동치 서술, **semi-graphoid 공리** (symmetry, decomposition, weak union, contraction) 증명, intersection property가 positive density에서만 성립함을 반례로 확인 |
| [02. Bayesian Network — DAG 기반 인수분해](./ch1-conditional-independence/02-bayesian-network-factorization.md) | $p(x_1, \ldots, x_n) = \prod_i p(x_i \mid \text{pa}(x_i))$로의 인수분해, **chain rule + 조건부 독립 가정**으로 유도, topological order의 역할, **local Markov property** 정의와 chain rule로부터의 증명 |
| [03. d-separation — 세 경로 패턴의 대수](./ch1-conditional-independence/03-d-separation.md) | Chain, Fork, **Collider (v-structure)** 세 패턴에서 blocked vs open 규칙, **d-separation ⟺ 조건부 독립** (soundness: Verma–Pearl 1988, completeness: Meek 1995) 증명, **explaining away** 현상의 대수적 필연성 |
| [04. Markov Random Field와 Hammersley–Clifford](./ch1-conditional-independence/04-markov-random-field.md) | $p(x) = \frac{1}{Z}\prod_C \phi_C(x_C)$의 정의, **Hammersley–Clifford 정리** — positive density 하에서 **Markov property ⟺ Gibbs distribution** 완전 증명 (Möbius inversion), partition function $Z$의 #P-hardness |
| [05. Moralization — DAG ↔ MRF 변환](./ch1-conditional-independence/05-moralization.md) | DAG의 parents를 **moralize** (공통 자식을 가진 부모들을 연결 후 방향 제거), 이 변환이 **conditional independence를 보존하지는 않음**을 반례로 확인, I-map과 P-map 개념, **minimal I-map** 유일성 정리 |

</details>

<br/>

### 🔹 Chapter 2: Factor Graph와 Message Passing

> **핵심 질문:** Factor graph가 어떻게 Bayesian network과 MRF를 통합하는가? Sum-product algorithm은 왜 tree에서 정확한 marginal을 제공하는가? Junction tree로 어떻게 loop가 있는 그래프에서도 exact inference를 할 수 있는가? Loopy BP의 수렴성은 어떻게 분석되는가?

<details>
<summary><b>Factor Graph부터 Loopy BP까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Factor Graph의 정의와 통합 표현](./ch2-factor-graph/01-factor-graph-definition.md) | Bipartite graph (variable node + factor node)의 정의, $p(x) = \frac{1}{Z}\prod_f \phi_f(x_{N(f)})$, **BN과 MRF를 factor graph로 동일하게 표현**하는 변환 규칙, directed / undirected 세계의 다리 |
| [02. Sum-Product Algorithm (Belief Propagation)](./ch2-factor-graph/02-sum-product-algorithm.md) | 변수→factor 메시지 $\mu_{x \to f}(x) = \prod_{f' \in N(x) \setminus f} \mu_{f' \to x}(x)$, factor→변수 메시지 $\mu_{f \to x}(x) = \sum_{x' \setminus x} f \prod \mu$, **tree에서 정확한 marginal** $p(x_i) \propto \prod_{f \in N(x_i)} \mu_{f \to x_i}(x_i)$ 증명 (귀납법) |
| [03. Max-Product Algorithm과 MAP Inference](./ch2-factor-graph/03-max-product-algorithm.md) | sum → max 치환이 $(+, \times) \to (\max, \times)$ **semiring 교체**임을 대수적으로 설명, Viterbi가 HMM에서의 max-product임을 유도, **tie-breaking과 uniqueness** 이슈, max-sum (log space) 변형 |
| [04. Junction Tree Algorithm](./ch2-factor-graph/04-junction-tree.md) | **Chordal triangulation** (fill-in edge 추가), junction tree 구성 + **running intersection property** 증명, cluster 간 message passing, 복잡도 $O(n \cdot d^{\text{tw}+1})$ 유도, min-fill / min-weight 휴리스틱 |
| [05. Loopy BP와 Bethe 자유에너지](./ch2-factor-graph/05-loopy-bp-bethe.md) | Loopy BP의 수렴/발산 예시, **Yedidia–Freeman–Weiss 2003**: Loopy BP의 고정점 ⟺ Bethe 자유에너지 $F_{\text{Bethe}} = U - H_{\text{Bethe}}$의 stationary point 증명, Ising model에서의 damping과 tree-reweighted BP |

</details>

<br/>

### 🔹 Chapter 3: Hidden Markov Model과 그 가족

> **핵심 질문:** HMM의 세 가지 문제 (Evaluation, Decoding, Learning)는 무엇이고 각각 어떤 알고리즘이 푸는가? Forward-Backward와 Viterbi가 왜 factor graph의 sum-product/max-product인가? Kalman filter가 어떻게 HMM의 Gaussian analog인가?

<details>
<summary><b>HMM의 정의부터 Kalman Filter까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. HMM의 정의와 세 가지 문제](./ch3-hmm/01-hmm-definition.md) | 은닉상태 $z_t$ + 관측 $x_t$, 전이 $p(z_t \| z_{t-1})$, 방출 $p(x_t \| z_t)$, HMM의 **factor graph 표현**, 세 가지 표준 문제 정의 (Evaluation·Decoding·Learning), 시계열 POS tagging 예제 |
| [02. Forward-Backward Algorithm = Sum-Product](./ch3-hmm/02-forward-backward.md) | Forward $\alpha_t(z) = p(x_{1:t}, z_t = z)$, Backward $\beta_t(z) = p(x_{t+1:T} \| z_t = z)$, posterior $p(z_t \| x_{1:T}) \propto \alpha_t \beta_t$, **$\alpha$/$\beta$가 정확히 factor graph sum-product 메시지임을 대응**으로 증명 |
| [03. Viterbi Algorithm = Max-Product](./ch3-hmm/03-viterbi-algorithm.md) | $\delta_t(z) = \max_{z_{1:t-1}} p(x_{1:t}, z_{1:t-1}, z_t = z)$의 recursion, backtracking pointer $\psi_t$, **Viterbi가 factor graph max-product의 HMM 특수경우**임을 유도, log space numerical stability |
| [04. Baum-Welch — EM for HMM](./ch3-hmm/04-baum-welch.md) | E-step: Forward-Backward로 $p(z_t \| x, \theta^{\text{old}})$와 $p(z_t, z_{t+1} \| x, \theta^{\text{old}})$ 계산, M-step: 전이·방출 파라미터 closed-form 업데이트, **EM의 HMM 특수경우**로서 ELBO 단조증가 |
| [05. Linear Dynamical System과 Kalman Filter](./ch3-hmm/05-kalman-filter.md) | 연속 상태 HMM, Gaussian transitions/emissions, **Kalman filter = Forward algorithm의 Gaussian 버전** 증명 (Gaussian의 conjugacy), **RTS smoother = Backward**, particle filter로의 비선형 일반화 |

</details>

<br/>

### 🔹 Chapter 4: Conditional Random Field와 구조화 예측

> **핵심 질문:** CRF가 HMM과 어떻게 다른가 (discriminative vs generative)? Linear-chain CRF의 inference와 learning은 어떤 구조인가? BiLSTM-CRF가 왜 NER의 표준이 되었는가?

<details>
<summary><b>CRF의 정의부터 Neural CRF까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. CRF의 정의 — Logistic Regression의 구조화 확장](./ch4-crf/01-crf-definition.md) | $p(y \| x) = \frac{1}{Z(x)}\exp(\sum_k w_k f_k(y, x))$, HMM vs CRF의 **생성/판별** 본질적 차이, $Z(x)$가 $x$에만 의존하므로 **feature engineering의 자유도** 증가, conditional log-linear model로서의 위치 |
| [02. Linear-Chain CRF의 Inference와 Learning](./ch4-crf/02-linear-chain-crf.md) | Forward-Backward 유사 알고리즘으로 $Z(x)$와 marginal 계산, log-likelihood gradient **$\nabla L = \sum_k f_k(y, x) - \mathbb{E}_{p(y \| x)}[f_k]$** (expected feature와 empirical feature의 차이) 유도, L-BFGS 학습 |
| [03. General CRF와 구조화 예측](./ch4-crf/03-general-crf.md) | **Skip-chain CRF** (long-range coreference), **Tree CRF** (parse tree), **Grid CRF** (이미지 분할, $\alpha$-expansion), **Structured SVM** (margin-based 학습)과의 비교, MAP inference로서의 decoding |
| [04. Neural CRF와 딥러닝 통합](./ch4-crf/04-neural-crf.md) | **BiLSTM-CRF** 아키텍처 (Huang et al. 2015), emission score = LSTM 출력 · transition score = CRF 파라미터, **Transformer + CRF** (NER SOTA), end-to-end 미분가능 학습, Viterbi가 여전히 test-time inference |

</details>

<br/>

### 🔹 Chapter 5: Variable Elimination과 Exact Inference

> **핵심 질문:** Variable Elimination에서 ordering이 왜 중요한가? Treewidth가 어떻게 inference 복잡도를 결정하는가? MAP inference는 왜 NP-hard이고 marginal inference는 왜 #P-hard인가?

<details>
<summary><b>Variable Elimination부터 복잡도 이론까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Variable Elimination Algorithm](./ch5-variable-elimination/01-variable-elimination.md) | Marginalization을 factor 분배법칙으로 국소화 $\sum_x f_1(x, y) f_2(x, z) = (\sum_x f_1 f_2)(y, z)$, **elimination ordering**이 intermediate factor 크기를 결정, 간단한 BN 예제로 단계별 trace |
| [02. Treewidth와 Inference Complexity](./ch5-variable-elimination/02-treewidth.md) | Elimination ordering에서 **최대 clique 크기 − 1 = treewidth** 정의, 복잡도 $O(n \cdot d^{\text{tw}+1})$, **min treewidth 문제의 NP-hardness** (Arnborg–Corneil–Proskurowski 1987), min-fill 휴리스틱의 성능 |
| [03. Clique Tree와 Junction Tree](./ch5-variable-elimination/03-clique-tree.md) | Triangulation으로 chordal graph 만들기, clique 간 **running intersection property**, belief propagation on junction tree, 두 가지 protocol (**Hugin, Shafer-Shenoy**)의 동치성 |
| [04. Inference의 복잡도 이론](./ch5-variable-elimination/04-inference-complexity.md) | **MAP inference는 NP-hard in general**, **marginal inference는 #P-hard** (Roth 1996), polytree에서는 linear, bounded treewidth에서는 polynomial, **PTAS가 있는 특수 경우** (planar graphs with bounded degree) |

</details>

<br/>

### 🔹 Chapter 6: Approximate Inference

> **핵심 질문:** Mean-field VI는 왜 ELBO를 최대화하는가? Bethe 자유에너지와 Loopy BP는 어떻게 연결되는가? Gibbs sampling에서 Markov blanket이 왜 local computation을 가능하게 하는가? Particle filter로 어떻게 비선형 state-space model을 다룰 수 있는가?

<details>
<summary><b>Variational Inference부터 RJMCMC까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Mean-Field Variational Inference](./ch6-approximate-inference/01-mean-field-vi.md) | Factorized approximation $q(x) = \prod_i q_i(x_i)$, **ELBO = $\mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q]$**, coordinate ascent 업데이트 $\log q_i \propto \mathbb{E}_{q_{-i}}[\log p]$, **KL 최소화 ⟺ ELBO 최대화** 등가 증명 |
| [02. Bethe 자유에너지와 Loopy BP의 변분 해석](./ch6-approximate-inference/02-bethe-loopy-bp.md) | Bethe 근사 $F_{\text{Bethe}} = U - H_{\text{Bethe}}$가 **tree에서 정확, loop에서는 근사**임을 증명, **Yedidia–Freeman–Weiss 2003**: Loopy BP 고정점 ⟺ Bethe 자유에너지 정체점 완전 유도 |
| [03. Expectation Propagation (EP)](./ch6-approximate-inference/03-expectation-propagation.md) | **Assumed density filtering + iterative refinement**, 각 factor를 exponential family로 근사하여 tractable posterior 유지, Gaussian EP (Minka 2001), **GP classification에서의 standard**, KL-proj의 moment matching 해석 |
| [04. Gibbs Sampling on Graphical Models](./ch6-approximate-inference/04-gibbs-sampling.md) | MRF/BN에서 Gibbs = 조건부 $p(x_i \| x_{-i})$ 샘플링, **Markov blanket 정리**로 local computation 증명 (MRF: neighbors, BN: parents + children + co-parents), **Ising model**·LDA의 표준 |
| [05. Particle Filter와 Sequential Monte Carlo](./ch6-approximate-inference/05-particle-filter.md) | **비선형·비Gaussian state-space model**에서 posterior sampling, **Sequential importance sampling + resampling**, degeneracy problem과 ESS 기반 적응적 resampling, **auxiliary PF**·resample-move PF 등 현대 variants |
| [06. Reversible Jump MCMC (RJMCMC)](./ch6-approximate-inference/06-reversible-jump-mcmc.md) | **모델 구조 자체가 불확실할 때** 차원이 변하는 MCMC (Green 1995), dimension-matching proposal의 **detailed balance** 증명, 변화점 검출·mixture model 수 추정에의 응용 |

</details>

<br/>

### 🔹 Chapter 7: 학습과 현대 응용

> **핵심 질문:** MRF의 파라미터 학습에서 partition function이 왜 intractable한가? EM이 왜 ELBO의 lower bound 반복으로 이해되는가? Structure Learning이 왜 NP-hard이고 Chow-Liu tree는 왜 가능한가? GNN과 Transformer attention이 어떻게 message passing의 현대적 확장인가?

<details>
<summary><b>MLE·EM·Structure Learning·LDA·GNN까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Maximum Likelihood for Graphical Models](./ch7-learning-modern/01-mle-for-graphical-models.md) | BN + complete data: **count-based MLE** 해석적 해, MRF: gradient $\nabla \log L = \mathbb{E}_{\text{data}}[f] - \mathbb{E}_{\text{model}}[f]$ 유도, partition function 때문에 **intractable**, **Contrastive Divergence** (Hinton 2002)·Pseudo-likelihood·Score Matching으로의 대체 |
| [02. EM Algorithm — 불완전 데이터](./ch7-learning-modern/02-em-algorithm.md) | Latent variable 모델에서 $Q(\theta \| \theta^{\text{old}}) = \mathbb{E}_{p(z \| x, \theta^{\text{old}})}[\log p(x, z \| \theta)]$ 최대화, **ELBO의 lower bound 해석**으로 monotonic improvement 증명, GMM·HMM의 EM이 모두 특수경우, generalized EM (GEM) |
| [03. Structure Learning](./ch7-learning-modern/03-structure-learning.md) | **Score-based** (BIC, BDeu, AIC) vs **Constraint-based** (PC algorithm, IC), **DAG learning의 NP-hardness** (Chickering 1996), greedy hill-climbing, **Chow-Liu 1968 알고리즘**으로 tree-restricted MLE를 **maximum spanning tree**로 해결 |
| [04. Topic Model (LDA)의 그래프 모델적 이해](./ch7-learning-modern/04-lda-topic-model.md) | **Latent Dirichlet Allocation** (Blei–Ng–Jordan 2003)의 hierarchical Bayesian network 표현, **Variational EM** 유도 (mean-field on $\phi, \theta, z$), **Collapsed Gibbs sampling** (Griffiths–Steyvers 2004)의 $\phi, \theta$ 주변화 |
| [05. GNN과 Transformer — Message Passing의 현대화](./ch7-learning-modern/05-gnn-transformer.md) | **GNN** (Gilmer et al. 2017): $h_v^{(t+1)} = \text{UPDATE}(h_v^{(t)}, \text{AGG}(\{\text{MSG}(h_u^{(t)}) : u \in N(v)\}))$이 BP의 **학습된 비선형 일반화**임을 증명, **Transformer attention = complete graph soft message passing**, HMM → CRF → GNN → Transformer 계보의 수학적 연속성 |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명**을 제공하는 대표 정리 모음입니다. 각 챕터의 문서에서 $\square$로 종결되는 엄밀한 증명을 확인할 수 있습니다. (전체 122개 정리 중 핵심만 발췌)

| 정리 | 서술 | 출처 문서 |
|------|------|----------|
| **Semi-graphoid 공리** | 조건부 독립이 symmetry·decomposition·weak union·contraction을 만족 — graph 표현의 대수적 기반 | [Ch1-01](./ch1-conditional-independence/01-conditional-independence-definition.md) |
| **DAG 인수분해 ⟺ local Markov** | $p = \prod p(x_i \| \text{pa}(x_i)) \iff X_i \perp\!\!\!\perp \text{nondesc} \mid \text{pa}$ | [Ch1-02](./ch1-conditional-independence/02-bayesian-network-factorization.md) |
| **d-separation의 soundness·completeness** | d-sep $\Rightarrow$ 조건부 독립 (Verma–Pearl 1988), 그리고 거의 모든 분포에서 완전 (Meek 1995) | [Ch1-03](./ch1-conditional-independence/03-d-separation.md) |
| **Hammersley–Clifford 정리** | Positive density 하에서 Markov ⟺ Gibbs $p \propto \prod_C \phi_C$ — Möbius inversion으로 증명 | [Ch1-04](./ch1-conditional-independence/04-markov-random-field.md) |
| **Sum-product algorithm의 정확성 (tree)** | Tree factor graph에서 BP는 정확한 marginal $p(x_i) \propto \prod_{f \in N(x_i)} \mu_{f \to x_i}$ | [Ch2-02](./ch2-factor-graph/02-sum-product-algorithm.md) |
| **Semiring 관점의 sum/max-product 통일** | $(+, \times)$ → marginal, $(\max, \times)$ → MAP — 동일 구조의 서로 다른 semiring 연산 | [Ch2-03](./ch2-factor-graph/03-max-product-algorithm.md) |
| **Junction Tree의 correctness** | Running intersection property를 만족하는 clique tree에서 cluster-level BP가 정확 | [Ch2-04](./ch2-factor-graph/04-junction-tree.md) |
| **Loopy BP ⟺ Bethe 자유에너지** | Yedidia–Freeman–Weiss 2003 — Loopy BP 고정점과 Bethe free energy stationary point의 동치 | [Ch2-05](./ch2-factor-graph/05-loopy-bp-bethe.md) |
| **Forward-Backward = Sum-product (HMM)** | $\alpha_t, \beta_t$가 HMM factor graph의 왼→오, 오→왼 sum-product 메시지와 일치 | [Ch3-02](./ch3-hmm/02-forward-backward.md) |
| **Viterbi = Max-product (HMM)** | Viterbi recursion이 HMM factor graph의 max-product와 일치, backtracking = max-product의 argmax | [Ch3-03](./ch3-hmm/03-viterbi-algorithm.md) |
| **Baum-Welch = EM의 HMM 특수경우** | E-step = Forward-Backward, M-step = 전이·방출 closed-form, ELBO 단조증가 | [Ch3-04](./ch3-hmm/04-baum-welch.md) |
| **Kalman Filter = Forward의 Gaussian analog** | Gaussian의 conjugacy로 $\alpha_t$가 Gaussian으로 유지, 평균·공분산의 recursion | [Ch3-05](./ch3-hmm/05-kalman-filter.md) |
| **CRF log-likelihood gradient** | $\nabla L = \sum_k f_k(y, x) - \mathbb{E}_{p(y \| x)}[f_k]$ — empirical vs expected feature 차이 | [Ch4-02](./ch4-crf/02-linear-chain-crf.md) |
| **Treewidth = Inference 복잡도** | Exact inference 복잡도 $O(n \cdot d^{\text{tw}+1})$, min treewidth 문제의 NP-hardness | [Ch5-02](./ch5-variable-elimination/02-treewidth.md) |
| **Marginal inference #P-hardness** | Roth 1996 — general graph에서 marginal inference는 #P-complete | [Ch5-04](./ch5-variable-elimination/04-inference-complexity.md) |
| **KL 최소화 ⟺ ELBO 최대화** | VI의 기본 항등식 $\log p(x) = \text{ELBO}(q) + \text{KL}(q \| p(\cdot \| x))$ | [Ch6-01](./ch6-approximate-inference/01-mean-field-vi.md) |
| **Markov Blanket 정리 (MRF)** | $p(x_i \| x_{-i}) = p(x_i \| x_{N(i)})$ — Gibbs sampling local computation의 근거 | [Ch6-04](./ch6-approximate-inference/04-gibbs-sampling.md) |
| **EM의 monotonic improvement** | $\log p(x \| \theta^{(t+1)}) \geq \log p(x \| \theta^{(t)})$ — ELBO의 lower bound 반복 증명 | [Ch7-02](./ch7-learning-modern/02-em-algorithm.md) |
| **Chow-Liu 정리 (1968)** | Tree-restricted MLE = 상호정보량을 edge weight로 하는 maximum spanning tree | [Ch7-03](./ch7-learning-modern/03-structure-learning.md) |
| **GNN = 학습된 BP, Attention = complete graph GNN** | Message passing의 비선형 일반화로서의 GNN, softmax attention = fully-connected soft BP | [Ch7-05](./ch7-learning-modern/05-gnn-transformer.md) |

> 💡 **챕터별 총 정리 수**: Ch1(16) · Ch2(18) · Ch3(18) · Ch4(14) · Ch5(15) · Ch6(21) · Ch7(20) — 합계 **122개 정리 + 증명**, 약 **17,000 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
networkx==3.2.0       # 그래프 구조 시각화 (d-sep, factor graph, junction tree)
pgmpy==0.1.25         # PGM 라이브러리 (검증용 exact/approximate inference)
hmmlearn==0.3.0       # HMM 레퍼런스 비교
scikit-learn==1.3.0   # CRF 데이터셋·feature extraction
torch==2.1.0          # BiLSTM-CRF, GNN 실험 (Ch4·Ch7에서 최소한)
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            networkx==3.2.0 pgmpy==0.1.25 hmmlearn==0.3.0 \
            scikit-learn==1.3.0 torch==2.1.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — HMM Forward-Backward / Viterbi를 factor graph sum/max-product로 재확인
import numpy as np
import matplotlib.pyplot as plt

# 작은 HMM
n_states, n_obs = 3, 2
pi = np.array([0.5, 0.3, 0.2])       # 초기분포
A  = np.array([[0.7, 0.2, 0.1],      # 전이
               [0.1, 0.6, 0.3],
               [0.2, 0.2, 0.6]])
B  = np.array([[0.9, 0.1],           # 방출
               [0.5, 0.5],
               [0.2, 0.8]])

obs = [0, 1, 0, 0, 1, 1, 0]
T = len(obs)

# ─────────────────────────────────────────────
# 1. Forward (= sum-product 왼→오 메시지)
# ─────────────────────────────────────────────
alpha = np.zeros((T, n_states))
alpha[0] = pi * B[:, obs[0]]
for t in range(1, T):
    alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
likelihood = alpha[-1].sum()
print(f'P(obs) = {likelihood:.6f}')

# 2. Backward (= sum-product 오→왼 메시지)
beta = np.zeros((T, n_states))
beta[-1] = 1.0
for t in range(T-2, -1, -1):
    beta[t] = A @ (B[:, obs[t+1]] * beta[t+1])

# 3. Posterior marginal γ_t(i) = p(z_t = i | obs)  — sum-product의 결과
gamma = alpha * beta
gamma = gamma / gamma.sum(axis=1, keepdims=True)

# 4. Viterbi (= max-product)
delta = np.zeros((T, n_states))
psi   = np.zeros((T, n_states), dtype=int)
delta[0] = pi * B[:, obs[0]]
for t in range(1, T):
    for j in range(n_states):
        scores = delta[t-1] * A[:, j]
        psi[t, j]   = np.argmax(scores)
        delta[t, j] = scores.max() * B[j, obs[t]]

# Backtrack
z_star = np.zeros(T, dtype=int)
z_star[-1] = np.argmax(delta[-1])
for t in range(T-2, -1, -1):
    z_star[t] = psi[t+1, z_star[t+1]]

print(f'MAP states (Viterbi): {z_star}')

# 시각화: posterior marginal (sum-product) vs MAP path (max-product)
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(n_states):
    ax.plot(gamma[:, i], marker='o', label=f'P(z_t={i}|x)  [sum-product]')
ax.step(range(T), z_star, 'k-', linewidth=3, label='Viterbi MAP  [max-product]')
ax.set_xlabel('t'); ax.legend(); ax.set_title('Forward-Backward vs Viterbi — same factor graph, different semiring')
plt.tight_layout(); plt.show()

# 정리: Forward-Backward = sum-product, Viterbi = max-product. 같은 factor graph 위의
# 서로 다른 semiring (+,×) vs (max,×) 연산. 둘 다 BP의 특수경우.
#
# ⚠️ 주의: 위 코드는 개념 설명용 unscaled 구현. T가 크면 α_T가 underflow할 수 있음.
#         실전에서는 scaled Forward-Backward (각 step에서 정규화) 또는 log-space
#         구현 필수. 구체적 구현은 Ch3-02 문서 참조.
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 그래프 모델이 ML에서 중요한가** | HMM/CRF/GNN/Transformer/Diffusion과의 연결점 |
| 3 | 📐 **수학적 선행 조건** | Probability·Statistics·Information Theory 레포의 어떤 정리를 전제로 하는지 |
| 4 | 📖 **직관적 이해** | 그래프 구조로 조건부 독립을 "읽는" 연습 |
| 5 | ✏️ **엄밀한 정의** | d-separation·Markov property·factor graph의 엄밀한 정의 |
| 6 | 🔬 **정리와 증명** | d-sep soundness·Hammersley–Clifford·BP on tree·Bethe correspondence — "자명하다" 없이 |
| 7 | 💻 **NumPy 구현 검증** | NumPy로 BP/HMM/CRF 바닥 구현 + **NetworkX로 그래프 시각화** + pgmpy로 결과 검증 |
| 8 | 🔗 **AI/ML 연결** | BiLSTM-CRF, GNN, Transformer attention, Topic model, Diffusion inference |
| 9 | ⚖️ **가정과 한계** | Treewidth 폭발 / Loopy BP 발산 / partition function intractability |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·d-sep 판정·구현 문제 |

> 📚 **연습문제 총 102개**: 34문서 × 문서당 3문제(기초/심화/AI 연결), 모든 문제에 `<details>` 펼침 해설 포함. d-sep 판정 문제부터 Viterbi 재구현, Loopy BP 수렴 실험, BiLSTM-CRF 학습 연결까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 자동으로 다음 챕터 첫 문서로 연결되므로 순차 학습이 끊기지 않습니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 500줄(증명·코드·연습문제 포함) 기준 **약 1~1.5시간**. 전체 34문서는 약 **42~52시간** 상당.

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "HMM을 쓰지만 Forward-Backward가 왜 sum-product인지 모른다" — HMM·BP 집중 (5일, 약 12~15시간)</b></summary>

<br/>

```
Day 1  Ch1-01~03  조건부 독립 / DAG 인수분해 / d-separation
Day 2  Ch2-01~02  Factor graph와 Sum-Product algorithm
Day 3  Ch2-03     Max-Product과 MAP inference
Day 4  Ch3-01~02  HMM 정의와 Forward-Backward
       → 직접 sum-product 메시지로 재유도
Day 5  Ch3-03~04  Viterbi (= max-product) 와 Baum-Welch (= EM)
```

</details>

<details>
<summary><b>🟡 "CRF를 NER에 쓰지만 왜 discriminative인지 엄밀히 모른다" — CRF 집중 (1주, 약 14~18시간)</b></summary>

<br/>

```
Day 1  Ch1-01~04  조건부 독립 / BN / d-sep / MRF (Hammersley–Clifford)
Day 2  Ch2-01~02  Factor graph와 Sum-Product
Day 3  Ch3-01~02  HMM 정의 + Forward-Backward (CRF와 비교 대상)
Day 4  Ch4-01     CRF의 정의와 HMM과의 discriminative/generative 분기
Day 5  Ch4-02     Linear-Chain CRF inference + log-likelihood gradient
Day 6  Ch4-03     General CRF — skip-chain, tree, grid CRF
Day 7  Ch4-04     BiLSTM-CRF / Transformer-CRF 아키텍처
```

</details>

<details>
<summary><b>🔴 "그래프 모델과 inference를 완전 정복한다 + GNN/Transformer 계보 이해" — 전체 정복 (8주, 약 42~52시간)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 조건부 독립과 그래프 구조
        → d-separation 3 패턴 직접 적용 연습 (20+ 예제)
        → Hammersley–Clifford 증명 재구성

2주차  Chapter 2 전체 — Factor Graph와 Message Passing
        → NumPy로 sum-product 바닥 구현
        → Tree BP의 정확성 직접 확인 + Junction tree 수동 구성

3주차  Chapter 3 전체 — HMM 가족
        → Forward-Backward = sum-product 대응 완전 숙지
        → Kalman filter = Gaussian HMM Forward 재유도

4주차  Chapter 4 전체 — CRF와 구조화 예측
        → BiLSTM-CRF로 POS tagging 실전 구현
        → HMM vs CRF 성능 비교 실험

5주차  Chapter 5 전체 — Variable Elimination
        → Treewidth와 elimination ordering 직접 최적화
        → Min-fill 휴리스틱의 NumPy 구현

6주차  Chapter 6 (1~3) — Variational Inference와 Loopy BP
        → Mean-field VI 바닥 구현
        → Loopy BP ⟺ Bethe 증명 재구성
        → EP를 Gaussian toy example로 재현

7주차  Chapter 6 (4~6) — Sampling 기반 근사
        → Ising model에 Gibbs sampling 적용
        → Particle filter로 비선형 tracking
        → RJMCMC로 모델 차원 추정

8주차  Chapter 7 전체 — 학습과 현대 응용
        → Chow-Liu tree 학습 구현
        → LDA를 collapsed Gibbs로 학습
        → GNN / Transformer attention = BP의 신경망적 일반화 정리
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 측도, 조건부 기댓값, 조건부 독립의 측도론적 정의 | Ch1 전체(조건부 독립·Markov property), Ch3~6 전반 |
| [mathematical-statistics-deep-dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive) | MLE, MAP, exponential family | Ch4(CRF as log-linear model), Ch7-01(MLE), Ch7-02(EM) |
| [information-theory-deep-dive](https://github.com/iq-ai-lab/information-theory-deep-dive) | 엔트로피, KL divergence, 상호정보량 | Ch6-01(ELBO as KL), Ch6-02(Bethe entropy), Ch7-03(Chow-Liu) |
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 행렬 연산, 고유분해 | Ch3-05(Kalman filter의 행렬 recursion), Ch2-04(Junction tree) |
| [bayesian-ml-deep-dive](https://github.com/iq-ai-lab/bayesian-ml-deep-dive) | Variational Inference, MCMC | Ch6 전체(병렬 학습 권장) — 본 레포는 **그래프 구조** 강조 |
| [sde-deep-dive](https://github.com/iq-ai-lab/sde-deep-dive) | 이토 적분, Fokker-Planck, Score-SDE | Ch6(Langevin/SGLD와의 연결), Ch7-05(Diffusion inference) |
| [deep-learning-deep-dive](https://github.com/iq-ai-lab/deep-learning-deep-dive) | NN 아키텍처, 학습 동역학 | Ch4-04(BiLSTM-CRF), Ch7-05(GNN·Transformer) |

> 💡 이 레포는 **확률분포의 그래프 구조와 그 위의 inference**에 집중합니다. Probability에서 조건부 독립의 측도론적 정의를, Information Theory에서 KL 발산을 학습한 후 오면 Ch1·Ch6이 훨씬 자연스럽습니다. Ch4(CRF)·Ch7-05(GNN/Transformer)는 딥러닝 실전 경험이 있을 때 최대의 효과를 냅니다.

---

## 📖 Reference

### 🏛️ PGM 바이블·표준 교재
- **Probabilistic Graphical Models: Principles and Techniques** (Koller & Friedman, 2009) — **PGM 바이블**, 이 레포의 뼈대
- **Pattern Recognition and Machine Learning** (Bishop, 2006) — Chapter 8 Graphical Models, Chapter 13 HMM, Chapter 10 Variational Inference
- **Machine Learning: A Probabilistic Perspective** (Murphy, 2012) — 현대 ML 관점의 PGM
- **Information Theory, Inference, and Learning Algorithms** (MacKay, 2003) — BP·MRF의 직관적 해설
- **Graphical Models, Exponential Families, and Variational Inference** (Wainwright & Jordan, 2008) — **변분 관점의 표준 리뷰 논문** (FnT)

### 🔄 Message Passing · Belief Propagation · Junction Tree
- **Factor Graphs and the Sum-Product Algorithm** (Kschischang, Frey & Loeliger, 2001) — **Factor graph 원전**
- **Local Computations with Probabilities on Graphical Structures and Their Application to Expert Systems** (Lauritzen & Spiegelhalter, 1988) — **Junction Tree / Hugin algorithm 원전**
- **Probabilistic Networks and Expert Systems** (Cowell, Dawid, Lauritzen & Spiegelhalter, 1999) — JT 및 evidence propagation의 표준 교재
- **Constructing Free-Energy Approximations and Generalized Belief Propagation** (Yedidia, Freeman & Weiss, 2003) — **Bethe / Loopy BP 원전**
- **Understanding Belief Propagation and Its Generalizations** (Yedidia, Freeman & Weiss, 2001) — BP 튜토리얼
- **Loopy Belief Propagation for Approximate Inference: An Empirical Study** (Murphy, Weiss & Jordan, 1999) — Loopy BP 실증 연구
- **Complexity of Finding Embeddings in a k-Tree** (Arnborg, Corneil & Proskurowski, 1987) — **Treewidth NP-hardness 원전**
- **On the Hardness of Approximate Reasoning** (Roth, 1996) — **Marginal inference #P-hardness 원전**

### 🎯 Bayesian Network · Causal · d-Separation
- **Probabilistic Reasoning in Intelligent Systems** (Pearl, 1988) — **BN·d-separation 원전**
- **Causality: Models, Reasoning, and Inference** (Pearl, 2009) — 인과 추론
- **Equivalence and Synthesis of Causal Models** (Verma & Pearl, 1990) — d-sep soundness·completeness

### 📈 HMM · State-Space Model · Kalman
- **A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition** (Rabiner, 1989) — **HMM 표준 튜토리얼**
- **A New Approach to Linear Filtering and Prediction Problems** (Kalman, 1960) — **Kalman filter 원전**
- **Maximum Likelihood from Incomplete Data via the EM Algorithm** (Dempster, Laird & Rubin, 1977) — **EM 원전**

### 🎨 CRF · Structured Prediction
- **Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data** (Lafferty, McCallum & Pereira, 2001) — **CRF 원전**
- **An Introduction to Conditional Random Fields** (Sutton & McCallum, 2012) — CRF 튜토리얼 (FnT)
- **Bidirectional LSTM-CRF Models for Sequence Tagging** (Huang, Xu & Yu, 2015) — **BiLSTM-CRF 원전**
- **Max-Margin Markov Networks** (Taskar, Guestrin & Koller, 2003) — Structured SVM

### 🎲 Approximate Inference · Sampling
- **A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants** (Neal & Hinton, 1998) — EM as ELBO
- **Expectation Propagation for Approximate Bayesian Inference** (Minka, 2001) — **EP 원전**
- **Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images** (Geman & Geman, 1984) — Gibbs sampling의 PGM 응용
- **Reversible Jump Markov Chain Monte Carlo** (Green, 1995) — **RJMCMC 원전**
- **Sequential Monte Carlo Methods in Practice** (Doucet, de Freitas & Gordon, 2001) — Particle filter 종합

### 📚 Topic Model · LDA
- **Latent Dirichlet Allocation** (Blei, Ng & Jordan, 2003) — **LDA 원전**
- **Finding Scientific Topics** (Griffiths & Steyvers, 2004) — Collapsed Gibbs sampling for LDA

### 🕸️ Structure Learning
- **Approximating Discrete Probability Distributions with Dependence Trees** (Chow & Liu, 1968) — **Chow-Liu 원전**
- **Learning Bayesian Networks: The Combination of Knowledge and Statistical Data** (Heckerman, Geiger & Chickering, 1995) — Score-based
- **Causation, Prediction, and Search** (Spirtes, Glymour & Scheines, 2000) — PC algorithm

### 🧠 Graph Neural Network · Transformer as Message Passing
- **Neural Message Passing for Quantum Chemistry** (Gilmer et al., 2017) — **GNN의 message passing formulation**
- **Semi-Supervised Classification with Graph Convolutional Networks** (Kipf & Welling, 2017) — GCN
- **Graph Attention Networks** (Veličković et al., 2018) — GAT
- **Attention Is All You Need** (Vaswani et al., 2017) — Transformer (complete graph GNN)
- **Relational Inductive Biases, Deep Learning, and Graph Networks** (Battaglia et al., 2018) — GNN·PGM의 통합 관점

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"조건부 독립을 그래프로 그릴 수 있는 것과, d-separation이 global Markov를 결정하고 Forward-Backward가 sum-product의 HMM 특수경우이며 Loopy BP가 Bethe 자유에너지의 변분 고정점임을 증명할 수 있는 것은 다르다"*

</div>
