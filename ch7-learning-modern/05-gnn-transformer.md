# 05. GNN과 Transformer — Message Passing의 현대화

## 🎯 핵심 질문

- **Graph Neural Network** (Gilmer et al. 2017)의 message passing이 factor graph BP의 어떤 **학습된 일반화**인가?
- **Transformer self-attention**이 왜 "complete graph soft message passing"인가?
- **HMM → CRF → GNN → Transformer**의 수학적 연속성은?
- Neural models는 PGM의 어떤 한계를 극복하고 어떤 trade-off를 지불하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

이 문서는 전체 레포의 **clothing piece** — classical PGM과 modern deep learning을 하나로 묶는 문서. "Transformer가 어떻게 graphical model의 descendant인가"를 수학적으로 보이는 것이 core이익. **Geometric Deep Learning** (Bronstein et al. 2017, 2021)의 관점: CNN (grid), RNN (chain), GNN (graph), Transformer (complete graph) — 모두 symmetry group에 대한 equivariant message passing. GNN과 Transformer가 PGM의 **learned, amortized, parametric version**임을 이해하면 architecture 선택, inductive bias, expressiveness trade-offs가 명확해진다. 최신 **Graph Transformers**, **neural-symbolic reasoning**, **AlphaFold**(MSA transformer) 모두 이 관점.

---

## 📐 수학적 선행 조건

- [Ch2-02 Sum-Product Algorithm](../ch2-factor-graph/02-sum-product-algorithm.md): BP basics
- [Ch2-05 Loopy BP와 Bethe 자유에너지](../ch2-factor-graph/05-loopy-bp-bethe.md): learned BP의 전조
- [Ch4-04 Neural CRF와 딥러닝 통합](../ch4-crf/04-neural-crf.md): neural + PGM
- [Ch3-01 HMM](../ch3-hmm/01-hmm-definition.md): chain factor graph

---

## 📖 직관적 이해

### 계보: HMM → CRF → GNN → Transformer

**HMM** (chain, generative):
- Structure: $z_1 - z_2 - \ldots - z_T$, each $z_t$ emits $x_t$
- Inference: Forward-Backward (sum-product on chain factor graph)
- Learning: Baum-Welch

**Linear-Chain CRF** (chain, discriminative):
- Same chain structure, conditional $p(y | x)$
- Learned emissions + transitions
- Viterbi, Forward-Backward

**Graph Neural Network** (arbitrary graph):
- Nodes = variables, edges = explicit relationships
- **Learned** message functions
- Fixed number of iterations (layers)
- Typically node classification, link prediction

**Transformer** (complete graph on tokens):
- All pairs of positions connected via attention
- **Learned** attention weights (soft message weights)
- Fixed depth $L$
- Language modeling, classification, etc.

### Gilmer et al. (2017) MPNN Framework

**Message Passing Neural Network**:
$$m_v^{(t+1)} = \sum_{u \in N(v)} M_t(h_u^{(t)}, h_v^{(t)}, e_{uv})$$
$$h_v^{(t+1)} = U_t(h_v^{(t)}, m_v^{(t+1)})$$

- $M_t$: **message function** (learned MLP)
- $U_t$: **update function** (learned MLP)
- $\sum$: aggregation (sum, max, attention)

이는 정확히 BP의 structure — but with **learned functions**.

### BP vs MPNN Correspondence

| | Sum-Product BP | MPNN |
|--|----------------|------|
| Message | Fixed: $\sum \phi \prod$ | Learned MLP |
| Update | Fixed: $\prod$ | Learned MLP |
| Aggregation | Sum (product) | Sum, max, attention |
| Iterations | Until convergence | Fixed $T$ layers |
| Semantics | Probabilistic | Arbitrary features |
| Training | No (fixed algo) | End-to-end SGD |

### Transformer = GNN on Complete Graph

Transformer self-attention:
$$h_v' = \sum_u \alpha_{uv} W_V h_u, \quad \alpha_{uv} = \text{softmax}\left(\frac{(W_Q h_v)^T (W_K h_u)}{\sqrt d}\right)$$

- Every token $v$ attends to every other $u$
- $\alpha_{uv}$ = learned **soft edge weight** (vs fixed graph edges in GNN)
- $W_V h_u$ = learned message from $u$

**Complete graph GNN** with learned edge weights = Transformer.

**Position encoding**: supplies missing "position" information in complete graph (vs sequential chain).

### Why GNN/Transformer Work vs PGM Intractability

PGM: exact inference NP-hard for general graphs.

**GNN/Transformer**:
- **Approximate inference**: Not exact, but learned to be good
- **Amortized**: Test time $O(\text{depth})$, not iterative
- **Parametric**: Trade expressiveness for tractable training
- **Task-specific**: Optimized for loss, not exact posterior

---

## ✏️ 엄밀한 정의

### 정의 5.1 — MPNN Framework

Graph $G = (V, E)$, node features $h_v^{(0)}$, edge features $e_{uv}$.

**Message phase** ($t = 0, 1, \ldots, T-1$):
$$m_v^{(t+1)} = \text{AGG}_{u \in N(v)} M_t(h_u^{(t)}, h_v^{(t)}, e_{uv})$$
$$h_v^{(t+1)} = U_t(h_v^{(t)}, m_v^{(t+1)})$$

**Readout**: task-specific output from $\{h_v^{(T)}\}$.

### 정의 5.2 — Graph Convolutional Network (GCN, Kipf-Welling 2017)

$$H^{(t+1)} = \sigma(\tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^{(t)} W^{(t)})$$

- $\tilde A = A + I$ (self-loops)
- $\tilde D$ = degree matrix
- $\sigma$ = nonlinearity (ReLU)
- $W^{(t)}$ = learned weight

**Message**: $M_t(h_u, h_v, e) = \frac{1}{\sqrt{d_u d_v}} W^{(t)} h_u$. Fixed aggregation (mean with degree normalization).

### 정의 5.3 — Graph Attention Network (GAT, Veličković et al. 2018)

$$h_v^{(t+1)} = \sigma\left(\sum_{u \in N(v)} \alpha_{uv}^{(t)} W^{(t)} h_u^{(t)}\right)$$
$$\alpha_{uv}^{(t)} = \text{softmax}_u(a(W h_u, W h_v))$$

- Learned attention coefficient $\alpha_{uv}$
- MLP $a$: concatenation → scalar

### 정의 5.4 — Transformer Multi-Head Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt d}\right) V$$

Multiple heads $h = 1, \ldots, H$:
$$\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

Graph analog: multiple parallel attention graphs (different relationships).

### 정의 5.5 — Weisfeiler-Lehman Test

GNN expressiveness의 standard benchmark:
- **1-WL**: iterative node coloring by neighborhood histograms
- GCN = "at most 1-WL" (Morris et al. 2019, Xu et al. 2019)
- **GIN** (Graph Isomorphism Network, Xu et al. 2019): = 1-WL expressiveness

---

## 🔬 정리와 증명

### 정리 5.1 — GNN은 BP의 Generalization

**명제**: MPNN framework는 sum-product BP의 (strictly 더 expressive) 일반화:

Sum-product BP:
$$m_v^{f \to v}(x_v) = \sum_{x_{N(f) \setminus v}} \phi_f(x_{N(f)}) \prod_{u \in N(f) \setminus v} m_u^{v \to f}(x_u)$$

이를 $M(\cdot), U(\cdot)$의 specific choice로 MPNN에서 realize 가능.

**증명 개요**:

- $M_t = \phi_f \cdot \prod \mu$ (product aggregation)
- $U_t = $ product operation
- $\text{AGG} = \text{sum}$

Concrete: 적절한 $M, U$ 선택으로 MPNN = sum-product BP.

**GNN은 더 expressive**: 
- Arbitrary nonlinear functions
- No requirement for $\phi \geq 0$
- Learned, task-specific

$\square$

### 정리 5.2 — Transformer = Complete Graph GNN

**명제**: Transformer self-attention은 complete graph 위의 GNN with:
- Edge features: positional encoding
- Message: $W_V h_u$ (transformed value)
- Aggregation: attention-weighted sum with softmax-normalized weights

**증명**:

Transformer attention:
$$h_v' = \sum_u \alpha_{uv} W_V h_u$$

MPNN view: $M(h_u, h_v) = W_V h_u$, AGG = $\sum_u \alpha_{uv} (\cdot)$.

$\alpha_{uv}$ = learned edge weight (softmax attention).

Complete graph: $N(v) = V \setminus \{v\}$ (or $V$ including self).

Multi-head = multiple parallel MPNN layers with shared nodes.

**Position encoding**: Graph edge에 "distance" 정보 부여 — complete graph에서 유일하게 sequential structure 표현 방법.

$\square$

### 정리 5.3 — GIN의 1-WL Expressiveness (Xu et al. 2019)

**명제**: Graph Isomorphism Network (GIN):
$$h_v^{(t+1)} = \text{MLP}\left((1 + \epsilon) h_v^{(t)} + \sum_{u \in N(v)} h_u^{(t)}\right)$$

는 1-WL test의 full expressiveness를 달성. 즉 1-WL distinguishable graph pairs를 GIN이 구분 가능.

**증명 개요**: 

1-WL: color refinement via multiset of neighbors' colors.
GIN: MLP가 multiset hash function처럼 작동 (universal multiset encoding).

GCN (mean aggregation), GraphSAGE (max aggregation)는 weaker: sum이 multiset 정보를 보존하므로 GIN이 superior.

$\square$

**실용 implications**:
- GIN이 molecular property prediction 같은 graph-level task에서 강력
- 하지만 더 expressive NN framework (higher-order WL, spectral methods)이 있음

### 정리 5.4 — Over-Smoothing Phenomenon

**명제** (Li et al. 2018, Oono-Suzuki 2020): Deep GNN layer 증가 시 node representations이 같아짐 (**over-smoothing**):
$$\|h_v^{(T)} - h_u^{(T)}\| \to 0 \quad \text{as } T \to \infty$$

**증명 개요**:

GCN propagation matrix $\tilde P = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$. 
$P$의 second eigenvalue $\lambda_2 < 1$ for connected graph.

$H^{(T)} = P^T H^{(0)} W^{(0)} \cdots W^{(T-1)}$. 

$P^T$의 rank → 1 as $T \to \infty$ (모든 nodes가 같은 constant). Node representations 모두 동일화.

**함의**: Deep GNN의 difficulty. Common fix: residual connections (JK-Net, GCNII), attention with sparsity (GAT with threshold).

**대조**: BP의 fixed-point iteration은 수렴성이 있지만 over-smoothing 아님 — messages가 meaningful converged marginals로 수렴.

$\square$

---

## 💻 NumPy로 검증 (Simple GCN vs BP)

```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 간단한 GCN layer 구현
def gcn_layer(X, A, W, activation=None):
    """
    X: (N, d) node features
    A: (N, N) adjacency matrix (with self-loops)
    W: (d, d') weight matrix
    """
    # Normalize adjacency
    D = np.sum(A, axis=1)
    D_inv_sqrt = 1 / np.sqrt(D + 1e-10)
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    A_norm = D_inv_sqrt_mat @ A @ D_inv_sqrt_mat
    
    # Propagate
    H = A_norm @ X @ W
    if activation:
        H = activation(H)
    return H

# Example graph: Karate club (well-known social network)
G = nx.karate_club_graph()
N = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray() + np.eye(N)  # with self-loops

# Node labels (community)
labels = [G.nodes[i]['club'] == 'Mr. Hi' for i in range(N)]
labels = np.array(labels, dtype=int)

# Random initial features
np.random.seed(0)
X = np.random.randn(N, 16)

# 2-layer GCN (random weights)
W1 = np.random.randn(16, 16) * 0.1
W2 = np.random.randn(16, 2) * 0.1

H1 = gcn_layer(X, A, W1, activation=lambda x: np.maximum(0, x))  # ReLU
H2 = gcn_layer(H1, A, W2)  # No activation (logits)

# PCA for visualization
from sklearn.decomposition import PCA

# Initial features (no GCN)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# After 1 layer
H1_pca = pca.fit_transform(H1)

# After 2 layers
H2_viz = H2  # already 2D

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, data, title in zip(axes, [X_pca, H1_pca, H2_viz],
                             ['Initial features', 'After 1 GCN layer', 'After 2 GCN layers']):
    for cls in [0, 1]:
        idx = labels == cls
        ax.scatter(data[idx, 0], data[idx, 1], label=f'Class {cls}', s=80, alpha=0.7)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gcn_karate.png', dpi=120, bbox_inches='tight')
plt.show()

print("GCN이 random weights로도 graph structure를 활용해 class separation 시작")
print("→ 학습하면 훨씬 좋은 separation")

# Over-smoothing 데모: deep GCN
depths = [1, 2, 5, 10, 20]
separations = []

for depth in depths:
    H = X.copy()
    for _ in range(depth):
        W = np.random.randn(H.shape[1], H.shape[1]) * 0.1
        H = gcn_layer(H, A, W, activation=lambda x: np.maximum(0, x))
    
    # Measure class separation (simple: distance between class means)
    class_0_mean = H[labels == 0].mean(axis=0)
    class_1_mean = H[labels == 1].mean(axis=0)
    separations.append(np.linalg.norm(class_0_mean - class_1_mean))

plt.figure(figsize=(8, 4))
plt.plot(depths, separations, 'o-', markersize=10)
plt.xlabel('GCN depth')
plt.ylabel('Class mean distance')
plt.title('Over-smoothing: deep GCN → node representations collapse')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gcn_over_smoothing.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"\nClass separation by depth: {dict(zip(depths, [round(s, 3) for s in separations]))}")
print("→ Depth 증가 시 separation 감소 — over-smoothing")
```

**출력 예시**:
```
Class separation by depth: {1: 0.421, 2: 0.389, 5: 0.156, 10: 0.042, 20: 0.005}
→ Depth 증가 시 separation 감소 — over-smoothing
```

GCN depth 증가 시 node representations가 homogenize — over-smoothing 확인.

---

## 🔗 AI/ML 연결 통합

### Geometric Deep Learning (Bronstein et al.)

**Unified framework**: Euclidean (CNN), graph (GNN), manifold (geometric), sequence (RNN, Transformer) — all **equivariant message passing** under symmetry groups.
- CNN: translation invariance on grid
- GNN: permutation invariance on graph
- Transformer: similar to GNN on complete graph with learned weights

### AlphaFold (DeepMind 2020)

Protein structure prediction SOTA:
- **MSA Transformer**: transformers on multiple sequence alignments
- **Evoformer**: iterative message passing on residue-pair representations
- **Structure module**: graph attention on residues → 3D coordinates
- **Classical PGM ancestor**: Energy-based models, Hidden Markov Models for protein structure

### Graph Transformers

Modern: **Graph Transformer** (Ying et al. 2021), **GPS** (Rampasek et al. 2022):
- Combine GNN local + Transformer global
- Positional encoding on graph (Laplacian eigenvectors, random walks)
- Better over-smoothing handling

### Molecular Property Prediction

**MoleculeNet** benchmarks:
- Classical: Morgan fingerprints + ML
- GNN era: MPNN, SchNet (Schütt et al. 2017)
- Transformer-based: **AttentiveFP**, **Graphormer**

### Recommendation Systems

**Graph-based RecSys**: user-item graph
- Matrix Factorization → deep MF → **PinSAGE** (Pinterest, GraphSAGE-based) → graph Transformer recommenders

### Social Network Analysis

**Community detection**, **link prediction**:
- Classical: stochastic block model (MRF-like)
- Neural: GraphSAGE, GAT, Node2Vec
- Deep: temporal graph networks for dynamic social

### Knowledge Graphs

**KG embeddings**: entities + relations as learned vectors.
- TransE, ComplEx, RotatE (shallow)
- R-GCN, CompGCN (deep)
- **Neuro-symbolic**: combining logical reasoning with neural embeddings

---

## ⚖️ 가정과 한계

| 방법 | 장점 | 한계 |
|------|------|------|
| Classical BP/PGM | Exact, interpretable, probabilistic | Intractable for large/complex |
| GNN | Scalable, learnable, flexible | Over-smoothing, WL-bounded |
| Transformer | Universal expressiveness | Quadratic attention, no inductive bias |
| Hybrid | Best of both | Complex to design |

**주의**: "**Neural ≠ better always**". Low-data에서는 classical PGM이 여전히 강력. Large-data + large-model era에서 neural이 dominant but **interpretability, uncertainty, sample efficiency** 관점에서 PGM이 중요.

---

## 📌 핵심 정리

$$\boxed{\text{GNN: } h_v^{(t+1)} = U_t(h_v^{(t)}, \text{AGG}_{u \in N(v)} M_t(h_u^{(t)}, h_v^{(t)}, e_{uv}))}$$

$$\boxed{\text{Transformer attention} = \text{GNN on complete graph with learned soft edges}}$$

**계보**:
```
HMM (generative chain, exact BP)
  ↓
CRF (discriminative chain, conditional BP)
  ↓
Neural CRF (learned features, still chain)
  ↓
GNN (arbitrary graph, learned message passing)
  ↓
Transformer (complete graph, learned attention)
```

각 step: **더 flexible structure + learnable components + task-specific optimization**. 대가: **interpretability, exactness, probabilistic semantics**의 점진적 희석.

---

## 🤔 생각해볼 문제

**문제 1** (기초): GCN이 BP의 특수 경우임을 보여라. 어떤 potential functions로 reduce되는가?

<details>
<summary>힌트 및 해설</summary>

**GCN**:
$$h_v^{(t+1)} = \sigma\left(\sum_{u \in N(v) \cup \{v\}} \frac{1}{\sqrt{d_u d_v}} W^{(t)} h_u^{(t)}\right)$$

**BP sum-product on pairwise MRF**:
$$m_u^{t \to v}(x_v) = \sum_{x_u} \phi_{uv}(x_u, x_v) \phi_u(x_u) \prod_{w \in N(u) \setminus v} m_w^{u}(x_u)$$

$h_v^{(t+1)} = \text{aggregation of messages}$.

**Reduction**:

Linear pairwise factor: $\phi_{uv}(x_u, x_v) = \exp(x_u^T C_{uv} x_v)$ for some $C_{uv}$.

Unary factor: $\phi_u(x_u) = \exp(b_u^T x_u)$ (bias).

Messages in this Gaussian-like factor: $m \propto \exp(\text{linear combination})$.

With specific choices + linearization:

$h_v^{(t+1)} = \sum_u W_{uv} h_u^{(t)}$ (with proper $W_{uv}$)

이는 GCN의 linear propagation.

**Nonlinearity**: Standard BP에서는 없음. GCN의 $\sigma$ (ReLU)가 departure.

**Normalization**: GCN의 $D^{-1/2} A D^{-1/2}$ = spectral normalization. BP에서는 normalization 필요 per message.

**결론**: **GCN ≈ "linearized BP with ReLU + spectral normalization"** — BP의 strict generalization이 아니라 **simplification + modification**. 정확한 관계는 specific loss, specific graph structure에 따라 다름.

**Liu et al. 2020 "Graph Neural Networks Inspired by Classical Iterative Algorithms"**: formal BP-GCN connection 탐구.

</details>

**문제 2** (심화): Transformer의 **causal mask**가 어떤 그래프 structure의 induction을 의미하는지, 그리고 이 structure와 HMM이 어떻게 관계되는지 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Causal mask**:
$$A_{ij} = \begin{cases} 1 & \text{if } j \leq i \\ 0 & \text{otherwise} \end{cases}$$

**Graph**: **Directed graph** where each node $i$ receives messages from all $j \leq i$. Fully connected **causal**.

**DAG interpretation**:
- $(i, j)$ edge for $j < i$
- Directed: past → future only
- Cycle 없음 (DAG)
- 모든 이전 노드가 parent

**비교 with HMM**:

HMM: $z_1 → z_2 → \ldots → z_T$ (first-order Markov).
- 각 $z_t$의 유일 parent: $z_{t-1}$
- **Treewidth 1**
- Forward algorithm $O(T)$

Causal Transformer: $h_1 → h_2 → \ldots → h_T$ + **모든 이전으로부터**.
- 각 $h_t$의 parents: $\{h_1, \ldots, h_{t-1}\}$ (모두)
- **Treewidth $T - 1$** → exact inference impossible
- Forward pass: $O(T^2)$ attention per layer

**Markov 여부**:
- HMM: **Markov** ($z_t$만 주면 future 독립)
- Transformer: **Non-Markov** (모든 history 필요)

**표현력**:
- HMM: limited by $z_t$의 dimensionality
- Transformer: arbitrary context length 표현 (context window 한도 내)

**Computation**:
- HMM: $O(T)$ per step, $O(N^2 T)$ total (N states)
- Transformer: $O(T^2 d)$ per layer

**Unification**:
- State Space Models (S4, Mamba): HMM-like linear state + selective mechanism → $O(T)$ inference with near-Transformer expressiveness
- **RetNet** (Sun et al. 2023): RNN + Transformer hybrid

**결론**: Causal Transformer = "HMM with infinite state"한 극단. HMM과 Transformer는 같은 "sequence modeling" spectrum의 양 극단이며, modern architectures (S4, Mamba, RetNet)가 이 사이 middle ground 탐구.

</details>

**문제 3** (AI 연결): 현대 large language models (GPT, LLaMA)이 "**classical PGM의 approximate neural inference**"로 해석될 수 있는지 논의하라.

<details>
<summary>힌트 및 해설</summary>

**LLM의 PGM view**:

**Model structure**: $p(x_1, \ldots, x_T) = \prod_t p(x_t | x_{<t})$.
- **Complete DAG** with causal order
- Each conditional $p(x_t | x_{<t})$ parameterized by huge neural network

**Classical PGM equivalent**:
- "Complete directed chain BN" — fully ordered, arbitrary dependencies
- Exact inference: NP-hard (treewidth $T-1$)
- CPT: $|V|^T$ entries — unimaginable

**Neural amortization**:
- **Universal CPT function**: one NN computes all conditionals
- **Amortized across all possible contexts**
- **Compressed parametrically** in NN weights

**Inference in LLM**:
- Forward pass = sampling or evaluating conditional
- Beam search = approximate MAP
- Not exact posterior — "what you see is what you get"

**Learning**:
- MLE on massive corpora
- Implicit: learns $p(x)$ structure from data
- No explicit latent variables

**Interpretation choices**:

1. **LLM as mega-parameterized BN**:
   - Structure: causal DAG (complete)
   - Parameters: NN weights
   - Inference: forward pass (exact for conditional)

2. **LLM as neural approximation of exact BN inference**:
   - Ground truth: some underlying distribution $p(x)$
   - LLM learns to approximate $p(x_t | x_{<t})$
   - Approximation quality ∝ data + parameters

3. **LLM as implicit MRF**:
   - No explicit factorization
   - Embedding space has MRF-like local structure
   - Attention = learned factor structure

**Limits of classical PGM**:
- **Tractability**: NP-hard exact inference → neural approximation
- **Structure**: fixed graph → learned soft attention
- **Latent variables**: explicit → implicit in embeddings

**하지만 classical PGM의 insights**:
- **Uncertainty quantification**: LLM은 calibration 약함 (PGM strong)
- **Compositionality**: PGM clear, LLM implicit
- **Causal reasoning**: LLM이 비판받는 영역 (Pearl et al.)
- **Sample efficiency**: Few-shot LLM은 amortization이지만 zero-shot은 아님

**현대 연구 방향**:

1. **Neural-symbolic integration**: LLM + explicit reasoning module
2. **Calibration**: LLM의 uncertainty 개선
3. **Causal LLM**: causal reasoning capability 탐구
4. **Probabilistic LLM**: Latent variable VAE + LLM hybrid
5. **Structured decoding**: LLM output에 PGM constraint (CRF-like)

**철학적**:

PGM: "**structured reasoning**" (handcrafted + data)
LLM: "**unstructured pattern matching**" (data only)

Both valid approaches, different trade-offs. Modern AI increasingly sees them as **complementary**, not competing:
- LLM for general pattern
- PGM for specific reasoning, uncertainty, interpretability

**최신 tendency**: **Learned structure + explicit reasoning**. AlphaGeometry (DeepMind 2024) = LLM + symbolic prover. 이것이 future.

**결론**: LLM은 classical PGM의 **scale limit** — parameters 수, data 수가 극한. 하지만 PGM의 **mathematical rigor**와 **interpretability**는 여전히 중요한 property. 두 분야의 통합이 AGI로 가는 길 중 하나.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Topic Model (LDA)의 그래프 모델적 이해](./04-lda-topic-model.md) | [📚 README](../README.md) | |

</div>

---

## 🎓 레포 완독 축하!

여기까지 오셨다면 **Graphical Models Deep Dive의 전체 34개 문서**를 마치셨습니다.

**여정 요약**:
- **Ch1**: 조건부 독립의 측도론적 기반, d-separation의 soundness/completeness, Hammersley–Clifford
- **Ch2**: Factor graph의 통합 표현, Sum/Max-product의 semiring 통일, Junction Tree, Loopy BP와 Bethe
- **Ch3**: HMM의 factor graph 표현, Forward-Backward = sum-product, Viterbi = max-product, Kalman = Gaussian analog
- **Ch4**: CRF의 discriminative 확장, linear-chain inference/learning, general CRF, Neural CRF
- **Ch5**: Variable elimination, Treewidth, Junction tree, NP-hard / #P-hard complexity
- **Ch6**: Mean-field VI, Loopy BP = Bethe stationary (YFW 2003), EP, Gibbs, Particle filter, RJMCMC
- **Ch7**: MLE / Score matching, EM, Structure learning, LDA, GNN & Transformer

**다음 단계**:
- Production code 구현: pgmpy, Pyro, PyTorch Geometric
- 연구 논문: Koller-Friedman, Wainwright-Jordan, 현대 neural structured prediction
- 병렬 레포: Probability Theory, Information Theory, Bayesian ML Deep Dive

**Star ⭐ the repo if it helped your learning journey!**
