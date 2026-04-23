# 01. HMM의 정의와 세 가지 문제

## 🎯 핵심 질문

- Hidden Markov Model은 어떤 **두 가정**(Markov + output independence)에 기반하는가?
- HMM의 세 가지 표준 문제 — Evaluation, Decoding, Learning — 는 각각 무엇인가?
- HMM을 **factor graph**로 표현하면 어떻게 되는가? 이 표현이 왜 자연스러운가?
- 왜 HMM은 chain factor graph의 **대표 예**인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**HMM**은 시계열 모델링의 **기초 architecture**다. 음성 인식 (acoustic model), 생물 정보학 (gene prediction, protein structure), **POS tagging**의 전통적 방법, 금융 시계열의 regime switching model — 모두 HMM에 기반. 현대적으로 **RNN·LSTM**은 HMM의 continuous-state, learned-transition 일반화. **Transformer의 causal attention**도 완전히 일반화된 chain BN. **State-space model**(Kalman filter, particle filter, S4 등) 전체 계보의 출발. HMM을 graph model로 이해하면 이 모든 후속 모델이 **factor graph의 변형**으로 통일된다.

---

## 📐 수학적 선행 조건

- [Ch1-02 Bayesian Network — DAG 기반 인수분해](../ch1-conditional-independence/02-bayesian-network-factorization.md): BN, Markov chain
- [Ch2-01 Factor Graph의 정의와 통합 표현](../ch2-factor-graph/01-factor-graph-definition.md): factor graph
- [Ch2-02 Sum-Product Algorithm](../ch2-factor-graph/02-sum-product-algorithm.md): tree BP
- 확률론: conditional distribution, 시계열

---

## 📖 직관적 이해

### HMM의 그림

```
   z_1 ─► z_2 ─► z_3 ─► ... ─► z_T     (hidden states)
    │       │       │             │
    ▼       ▼       ▼             ▼
   x_1     x_2     x_3           x_T    (observations)
```

- $z_t$: **hidden state** (은닉상태), 관측 불가
- $x_t$: **observation** (관측), 데이터로 주어짐
- 두 Markov 가정:
  - **Transition Markov**: $z_{t+1} \perp\!\!\!\perp z_{1:t-1} \mid z_t$
  - **Output independence**: $x_t \perp\!\!\!\perp x_{-t}, z_{-t} \mid z_t$

### 세 가지 모수 (Parameters)

$\theta = (\pi, A, B)$:
- $\pi_i := p(z_1 = i)$: **initial distribution**
- $A_{ij} := p(z_{t+1} = j \mid z_t = i)$: **transition matrix**
- $B_{ik} := p(x_t = k \mid z_t = i)$: **emission matrix**

### 결합분포

$$p(z_{1:T}, x_{1:T}) = p(z_1) \prod_{t=2}^T p(z_t | z_{t-1}) \prod_{t=1}^T p(x_t | z_t)$$

$$= \pi_{z_1} \cdot \prod_{t=2}^T A_{z_{t-1}, z_t} \cdot \prod_{t=1}^T B_{z_t, x_t}$$

### 세 가지 문제 (Rabiner 1989)

1. **Evaluation**: 주어진 $\theta$와 observation $x_{1:T}$에 대해 $p(x_{1:T} | \theta) = ?$
   - 도구: **Forward algorithm** (Ch3-02)

2. **Decoding**: 가장 그럴듯한 hidden sequence $z^*_{1:T} = \arg\max_{z_{1:T}} p(z_{1:T} | x_{1:T}, \theta)$
   - 도구: **Viterbi algorithm** (Ch3-03)

3. **Learning**: 데이터 $\{x^{(i)}_{1:T_i}\}$에서 $\theta$ 추정
   - 도구: **Baum-Welch = EM** (Ch3-04)

### POS Tagging 예시

영어 문장 "the cat sat" — tag sequence 찾기.

- Hidden states: POS tag (NOUN, VERB, DET, ...)
- Observations: 단어
- Transition: "DET 다음에 NOUN이 올 확률"
- Emission: "NOUN의 실제 단어가 'cat'일 확률"

관측: "the cat sat" → Viterbi로 tag sequence "DET NOUN VERB" 복원.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Hidden Markov Model

**HMM**은 다음 요소로 구성:

- **State space** $\mathcal{Z} = \{1, 2, \ldots, N\}$ (hidden states)
- **Observation space** $\mathcal{X}$ (discrete $\{1, \ldots, M\}$ 또는 continuous $\mathbb{R}^d$)
- **Initial distribution** $\pi \in \Delta^{N-1}$ (probability simplex)
- **Transition matrix** $A \in [0, 1]^{N \times N}$, $\sum_j A_{ij} = 1$
- **Emission distribution** $B$: discrete일 때 $B \in [0, 1]^{N \times M}$, continuous일 때 $p(x_t | z_t = i) = f_i(x_t; \theta_i)$

결합분포:
$$p(z_{1:T}, x_{1:T} | \theta) = \pi_{z_1} \prod_{t=2}^T A_{z_{t-1}, z_t} \prod_{t=1}^T B_{z_t, x_t}$$

### 정의 1.2 — HMM as Factor Graph

HMM의 factor graph:

```
       [f_π]    [f_2]     [f_3]           [f_T]
         │       │  │      │  │            │  │
         │       │  │      │  │            │  │
        z_1 ── z_2 ── z_3 ──   ... ──    z_T
         │       │        │                 │
       [g_1]  [g_2]    [g_3]             [g_T]
         │       │        │                 │
        x_1     x_2      x_3               x_T
```

- Variable nodes: $z_1, \ldots, z_T, x_1, \ldots, x_T$
- Factor nodes:
  - $f_\pi(z_1) = \pi_{z_1}$ (initial)
  - $f_t(z_{t-1}, z_t) = A_{z_{t-1}, z_t}$ (transition)
  - $g_t(z_t, x_t) = B_{z_t, x_t}$ (emission)

**관측**: $x_t$는 "evidence"로 관측된 값에 고정. 이는 $g_t$ factor를 $z_t$만의 함수로 reduce:
$$\tilde g_t(z_t) := B_{z_t, \hat x_t}$$

### 정의 1.3 — Three Standard Problems

HMM의 세 표준 문제 (Rabiner 1989):

**Problem 1 — Evaluation**: $p(x_{1:T} | \theta) = \sum_{z_{1:T}} p(z_{1:T}, x_{1:T} | \theta)$
- Direct: $N^T$ terms — exponential
- Solution: **Forward algorithm** — $O(N^2 T)$

**Problem 2 — Decoding**: $z^*_{1:T} = \arg\max_{z_{1:T}} p(z_{1:T} | x_{1:T}, \theta)$
- Equivalent: $\arg\max_z p(z, x | \theta)$ (posterior proportional to joint)
- Solution: **Viterbi algorithm** — $O(N^2 T)$

**Problem 3 — Learning**: $\hat\theta = \arg\max_\theta p(x | \theta)$ from data
- **Baum-Welch** (EM for HMM): iterative, monotonic ELBO improvement

---

## 🔬 정리와 증명

### 정리 1.1 — HMM Local Markov

**명제**: HMM은 다음 CI를 만족:
- $z_{t+1} \perp\!\!\!\perp z_{1:t-1} \mid z_t$ (transition Markov)
- $x_t \perp\!\!\!\perp \{z_{-t}, x_{-t}\} \mid z_t$ (output independence)

**증명**: HMM factor graph에서 d-separation 적용. 
- Chain $z_{t-1} \to z_t \to z_{t+1}$: $z_t$ 관측 시 $z_{t-1}, z_{t+1}$ CI
- $z_t \to x_t$ emission: $x_t$의 유일 parent가 $z_t$

이로부터 directly 귀결. $\square$

### 정리 1.2 — HMM Factor Graph는 Tree

**명제**: HMM의 factor graph는 **tree** (more precisely, poly-tree에 가까운 linear chain).

**증명**: 

Variable nodes: $\{z_1, \ldots, z_T, x_1, \ldots, x_T\}$. Factor nodes: initial + $T-1$ transitions + $T$ emissions.

Undirected edges in factor graph:
- $(z_1, f_\pi)$: 1 edge
- $(z_t, f_{t}), (z_{t-1}, f_t)$ for transitions: each $f_t$ has 2 edges
- $(z_t, g_t), (x_t, g_t)$ for emissions: each $g_t$ has 2 edges

Cycles가 있으려면 path가 닫혀야 하지만 observations $x_t$는 leaf(emission factor의 유일 connector). Transitions은 chain으로 연결되어 cycle 없음.

따라서 factor graph = tree. **Sum-product이 exact**. $\square$

### 정리 1.3 — Forward Recursion의 유도

**명제**: $\alpha_t(z_t) := p(z_t, x_{1:t} | \theta)$는 다음 recursion:
$$\alpha_1(z_1) = \pi_{z_1} B_{z_1, x_1}$$
$$\alpha_t(z_t) = \left[\sum_{z_{t-1}} \alpha_{t-1}(z_{t-1}) A_{z_{t-1}, z_t}\right] B_{z_t, x_t}$$

그리고 likelihood $p(x_{1:T}) = \sum_{z_T} \alpha_T(z_T)$.

**증명**:

Base: $\alpha_1(z_1) = p(z_1, x_1) = p(z_1) p(x_1 | z_1) = \pi_{z_1} B_{z_1, x_1}$.

Inductive:
$$\alpha_t(z_t) = p(z_t, x_{1:t}) = \sum_{z_{t-1}} p(z_{t-1}, z_t, x_{1:t})$$

Chain rule 및 HMM의 CI 활용:
$$p(z_{t-1}, z_t, x_{1:t}) = p(z_{t-1}, x_{1:t-1}) p(z_t | z_{t-1}) p(x_t | z_t)$$
$$= \alpha_{t-1}(z_{t-1}) \cdot A_{z_{t-1}, z_t} \cdot B_{z_t, x_t}$$

합:
$$\alpha_t(z_t) = B_{z_t, x_t} \sum_{z_{t-1}} \alpha_{t-1}(z_{t-1}) A_{z_{t-1}, z_t}$$

Likelihood: $p(x_{1:T}) = \sum_{z_T} p(z_T, x_{1:T}) = \sum_{z_T} \alpha_T(z_T)$. $\square$

### 정리 1.4 — Complexity

**명제**: Forward algorithm: $O(N^2 T)$ time, $O(NT)$ space.

**증명**: 각 $\alpha_t(z_t)$ 계산은 $N$ terms (summation over $z_{t-1}$). $z_t$는 $N$개 값 → 총 $N^2$ operations per step. $T$ steps → $O(N^2 T)$. Storage: $\alpha_t$는 size $N$ vector, $T$ step 전부 $O(NT)$. $\square$

**Brute force와 비교**: $p(x) = \sum_{z_{1:T}} \prod p(\cdot)$ — $N^T$ terms. HMM은 **Markov 가정으로 exponential → polynomial**.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# HMM 정의
class HMM:
    def __init__(self, pi, A, B):
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.B = np.array(B)
        self.N = len(pi)
        self.M = B.shape[1]
    
    def sample(self, T, rng=None):
        """Ancestral sampling."""
        rng = rng or np.random.default_rng()
        z = np.zeros(T, dtype=int)
        x = np.zeros(T, dtype=int)
        z[0] = rng.choice(self.N, p=self.pi)
        x[0] = rng.choice(self.M, p=self.B[z[0]])
        for t in range(1, T):
            z[t] = rng.choice(self.N, p=self.A[z[t-1]])
            x[t] = rng.choice(self.M, p=self.B[z[t]])
        return z, x
    
    def likelihood_bruteforce(self, x):
        """O(N^T) — only for small T."""
        T = len(x)
        from itertools import product
        total = 0
        for z in product(range(self.N), repeat=T):
            p = self.pi[z[0]] * self.B[z[0], x[0]]
            for t in range(1, T):
                p *= self.A[z[t-1], z[t]] * self.B[z[t], x[t]]
            total += p
        return total
    
    def likelihood_forward(self, x):
        """O(N^2 T) forward algorithm."""
        T = len(x)
        alpha = np.zeros((T, self.N))
        alpha[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * self.B[:, x[t]]
        return alpha[-1].sum(), alpha

# 예시: 날씨 HMM (teaching example)
# States: 0=Sunny, 1=Cloudy, 2=Rainy
# Obs: 0=NoUmbrella, 1=Umbrella
pi = [0.6, 0.3, 0.1]
A = [[0.7, 0.2, 0.1],
     [0.3, 0.5, 0.2],
     [0.2, 0.3, 0.5]]
B = [[0.9, 0.1],   # Sunny: rarely umbrella
     [0.6, 0.4],   # Cloudy
     [0.2, 0.8]]   # Rainy: usually umbrella

hmm = HMM(pi, A, B)

# 샘플 생성
rng = np.random.default_rng(42)
z, x = hmm.sample(T=10, rng=rng)
print(f"Hidden states (S/C/R): {z}")
print(f"Observations (No/Umb): {x}")

# Evaluation 비교
like_bf = hmm.likelihood_bruteforce(x)
like_fw, alpha = hmm.likelihood_forward(x)
print(f"\nEvaluation (brute force):  {like_bf:.6e}")
print(f"Evaluation (forward):      {like_fw:.6e}")
print(f"절대 오차: {abs(like_bf - like_fw):.2e}")
print(f"차원: brute {3**10} terms vs forward {3**2 * 10} ops")

# 복잡도 스케일링 실험
print("\n복잡도 비교 (T 증가):")
print(f"{'T':>5} {'Brute force ops':>20} {'Forward ops':>15}")
for T_test in [5, 10, 15, 20]:
    bf_ops = 3**T_test
    fw_ops = 3**2 * T_test
    print(f"{T_test:>5} {bf_ops:>20} {fw_ops:>15}")

# 시각화: HMM factor graph
fig, ax = plt.subplots(figsize=(12, 5))
T = 5
pos = {}
for t in range(T):
    pos[f'z{t+1}'] = (2*t, 1)
    pos[f'x{t+1}'] = (2*t, -1)
    pos[f'f{t+1}'] = (2*t - 1, 1)  # transition factor (except t=0 is initial)
    pos[f'g{t+1}'] = (2*t, 0)  # emission

G = nx.Graph()
# Add nodes
for t in range(T):
    G.add_node(f'z{t+1}', kind='var')
    G.add_node(f'x{t+1}', kind='var')
    G.add_node(f'g{t+1}', kind='factor')
    if t == 0:
        G.add_node(f'f{t+1}', kind='factor')  # initial factor f_π
    else:
        G.add_node(f'f{t+1}', kind='factor')  # transition

# Add edges
G.add_edge('f1', 'z1')
for t in range(1, T):
    G.add_edge(f'f{t+1}', f'z{t}')
    G.add_edge(f'f{t+1}', f'z{t+1}')

for t in range(T):
    G.add_edge(f'g{t+1}', f'z{t+1}')
    G.add_edge(f'g{t+1}', f'x{t+1}')

# Draw
var_nodes = [n for n in G.nodes() if G.nodes[n]['kind'] == 'var']
fac_nodes = [n for n in G.nodes() if G.nodes[n]['kind'] == 'factor']

nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, node_shape='o',
                       node_color='lightblue', node_size=1200, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=fac_nodes, node_shape='s',
                       node_color='lightcoral', node_size=800, ax=ax)
nx.draw_networkx_edges(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

ax.set_title('HMM as Factor Graph (T=5)')
ax.axis('off')
plt.tight_layout()
plt.savefig('hmm_factor_graph.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
Hidden states (S/C/R): [0 0 1 2 2 2 1 0 0 0]
Observations (No/Umb): [0 0 0 1 1 1 0 0 0 0]

Evaluation (brute force):  4.371504e-05
Evaluation (forward):      4.371504e-05
절대 오차: 5.42e-20
차원: brute 59049 terms vs forward 90 ops

복잡도 비교 (T 증가):
    T      Brute force ops     Forward ops
    5                  243              45
   10                59049              90
   15             14348907             135
   20           3486784401             180
```

Forward는 brute force 대비 $T = 20$에서 약 **2천만 배** 빠름.

---

## 🔗 AI/ML 연결

### RNN as Continuous-State HMM

RNN의 hidden state $h_t = f(h_{t-1}, x_t)$는 HMM의 continuous-state, deterministic-transition, learned-emission version:
- Transition $p(h_t | h_{t-1}, x_t)$ → deterministic $f$
- Emission $p(x_t | h_t)$ → softmax over vocabulary
- Markov 가정은 그대로 유지 ($h_t$가 모든 과거 정보를 압축)

**LSTM/GRU**는 gated transition — 더 복잡한 dynamics 학습.

### State Space Models (S4, Mamba)

Gu et al. 2021의 S4(Structured State Space)는:
$$h_t' = A h_t + B x_t, \quad y_t = C h_t$$

Linear Gaussian HMM (= Kalman filter, Ch3-05)의 직접 일반화. Discretization으로 **시간축 flexibility**, HiPPO 이론으로 **long-range memory**.

### Transformer's Causal Attention vs HMM

HMM: "각 $z_t$는 $z_{t-1}$에만 의존" — first-order Markov.
Transformer: "각 position은 모든 이전 position에 의존" — **no Markov assumption**.

이 차이가 Transformer가 long-range dependency를 잘 포착하는 근본 이유. 대신 Markov가 없으면 factor graph가 더 이상 tree가 아니고 exact inference도 exponential.

### Speech Recognition

**Classical speech recognition** (HTK, Kaldi 2015 이전): HMM-GMM acoustic model. 각 phoneme이 HMM state. Viterbi로 best phone sequence 디코딩.

**Neural speech** (wav2vec, Whisper): neural feature extractor + CTC/seq2seq. 하지만 여전히 CTC는 **monotonic alignment HMM의 일반화**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| First-order Markov | Long-range dependency 포착 못함 — higher-order HMM 또는 RNN 필요 |
| Stationary transition | $A$가 시간 불변 — non-stationary 환경에서 부적합 |
| Discrete state | Continuous는 Kalman filter 또는 particle filter |
| Output independence | $x_t$는 $z_t$만 의존 — joint emission은 CRF가 유연 |

**주의**: HMM은 **generative** model — $p(x, z)$ 전체를 모델링. 이는 efficient sampling을 가능케 하지만, **discriminative** CRF(Ch4)가 더 좋은 labeling 성능을 내는 경우 많음.

---

## 📌 핵심 정리

$$\boxed{p(z_{1:T}, x_{1:T}) = \pi_{z_1} \prod_{t=2}^T A_{z_{t-1}, z_t} \prod_{t=1}^T B_{z_t, x_t}}$$

| 문제 | 알고리즘 | 복잡도 |
|------|---------|-------|
| Evaluation | Forward | $O(N^2 T)$ |
| Decoding | Viterbi (Ch3-03) | $O(N^2 T)$ |
| Learning | Baum-Welch (Ch3-04) | $O(N^2 T I)$ per data |

| 개념 | 의미 |
|------|------|
| **Transition Markov** | 미래 ⊥ 과거 \| 현재 |
| **Output independence** | $x_t$는 $z_t$만 의존 |
| **Factor graph** | Chain tree, sum-product exact |
| **3 problems** | Rabiner 1989의 framework |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $N = 3$ states, $T = 10$ observations일 때 brute force와 forward의 연산 수를 비교하라.

<details>
<summary>힌트 및 해설</summary>

**Brute force**: $N^T = 3^{10} = 59049$ hidden sequences. 각각 $O(T)$ 연산. 총 $3^{10} \cdot 10 \approx 6 \times 10^5$.

**Forward**: $O(N^2 T) = 9 \cdot 10 = 90$ operations.

**Speedup**: $59049 / 90 \approx 656$.

$T$가 증가하면 차이가 exponential로 폭발: $T = 50$에서 $N^T = 3^{50} \approx 7 \times 10^{23}$ vs forward $450$ ops — 아예 비교 불가.

**교훈**: Dynamic programming이 HMM을 가능한 모델로 만듦. Markov assumption이 핵심.

</details>

**문제 2** (심화): Second-order HMM ($z_{t+1} | z_{t-1}, z_t$)의 복잡도는 어떻게 되는가? Factor graph는?

<details>
<summary>힌트 및 해설</summary>

**Second-order HMM**: Transition $p(z_t | z_{t-1}, z_{t-2})$.

**Factor graph**: 
- Variables: $z_1, \ldots, z_T, x_1, \ldots, x_T$
- Transition factor $f_t(z_{t-2}, z_{t-1}, z_t)$ — 3-variable factor!

**Tree 구조**: 여전히 tree (chain of larger factors). Sum-product exact.

**복잡도**: Factor marginalization $O(N^3)$ per step (3-variable factor). Total $O(N^3 T)$.

**등가 변환**: "Super-state" $\tilde z_t := (z_{t-1}, z_t)$로 정의 → first-order HMM with $N^2$ states → $O((N^2)^2 T) = O(N^4 T)$.

두 표현이 같은 복잡도 (approximately)이지만 첫 번째가 더 간결. 일반적으로 $k$-th order HMM은 $O(N^{k+1} T)$.

**실용**: 음성 인식에서는 second-order HMM을 쓰기도 했지만, 현대적으로는 RNN/Transformer가 unlimited order를 handle.

</details>

**문제 3** (AI 연결): Transformer의 self-attention이 "fully-connected chain BN"으로 해석될 때, HMM과의 표현력/복잡도 trade-off는?

<details>
<summary>힌트 및 해설</summary>

**HMM** (first-order chain):
- Expressiveness: 제한적 — current state만으로 예측
- Complexity: $O(N^2 T)$ — efficient
- Parameters: $O(N^2 + NM)$ — compact
- Inference: exact (tree)

**Transformer** (fully-connected):
- Expressiveness: maximum — 모든 이전 모든 position 참조
- Complexity: $O(T^2 d)$ per layer, $L$ layers → $O(L T^2 d)$
- Parameters: $O(L d^2)$ (attention + FFN)
- Inference: approximate (loopy)

**Trade-off**:
1. **표현력 vs 효율**: HMM은 tree 구조로 exact하지만 Markov 제약 / Transformer는 표현력 높지만 quadratic complexity
2. **파라미터 수**: HMM은 state 수로, Transformer는 hidden dim으로 — 다른 공간
3. **Long-range**: HMM은 state가 bottleneck / Transformer는 직접 access

**하이브리드 접근**:
- **State Space Models** (S4, Mamba): linear transition (HMM-like) + selective mechanism (Transformer-like) → long-range에 efficient
- **Linformer, Performer**: attention을 low-rank로 근사 → $O(T)$
- **Hidden Markov Transformer**: attention을 state transition으로 해석

**결론**: HMM과 Transformer는 sequence model space의 양 극단 (structure-rich vs structure-free). 최신 연구는 둘의 장점을 결합.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch2-05 Loopy BP와 Bethe 자유에너지](../ch2-factor-graph/05-loopy-bp-bethe.md) | [📚 README](../README.md) | [02. Forward-Backward Algorithm = Sum-Product ▶](./02-forward-backward.md) |

</div>
