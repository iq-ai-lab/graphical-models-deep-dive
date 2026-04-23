# 03. General CRF와 구조화 예측

## 🎯 핵심 질문

- **Skip-chain CRF**는 long-range dependency (coreference)를 어떻게 표현하는가?
- **Tree CRF**와 **Grid CRF** (이미지 분할)의 inference 복잡도는?
- **Structured SVM**과 CRF의 차이는? (margin vs likelihood)
- $\alpha$-expansion이 multi-label image segmentation에서 어떻게 작동하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**General CRF**는 linear-chain 이상의 **복잡한 구조적 출력**을 위한 프레임워크. Parse tree (constituency, dependency), image segmentation, coreference resolution, SRL(semantic role labeling), relational extraction — 모두 general CRF. **DeepLab** 같은 semantic segmentation SOTA는 CNN + dense CRF. Structured SVM(Tsochantaridis 2005)은 CRF의 margin-based 변형으로 scalability 개선. 이 문서는 CRF framework를 **임의 그래프 구조**로 확장하는 방법.

---

## 📐 수학적 선행 조건

- [Ch4-01 CRF의 정의](./01-crf-definition.md)
- [Ch4-02 Linear-Chain CRF](./02-linear-chain-crf.md)
- [Ch2-04 Junction Tree Algorithm](../ch2-factor-graph/04-junction-tree.md): general graph inference
- [Ch2-05 Loopy BP](../ch2-factor-graph/05-loopy-bp-bethe.md): loopy graph에서 approximate

---

## 📖 직관적 이해

### Skip-Chain CRF

NER에서 같은 entity가 문서에 여러 번 나타남 — 일관성 필요.

"Bill Clinton ... The president ... He ..."

모두 PERSON. Linear chain으로는 연결 안 됨 (non-adjacent). **Skip-chain**: coreference heuristic (같은 string, 대명사 등)으로 non-adjacent edge 추가.

```
y_1 - y_2 - y_3 - ... - y_{10} - ... - y_{50}
      |                    |
      └────────────────────┘    (skip edge: y_2와 y_{10} 같은 entity?)
```

**구조**: chain + skip edges → **tree가 아님** → exact inference 어려움. Loopy BP 사용.

### Tree CRF (Parsing)

Dependency parsing: 각 word가 head word를 가짐 → tree structure.
$$p(\text{tree} | \text{sentence}) \propto \exp(\sum_{\text{arcs}} w \cdot f(\text{arc}))$$

**Inference**: MST algorithm (Eisner, Chu-Liu-Edmonds)로 $O(n^2)$ 또는 $O(n^3)$.

Constituency parsing: CKY algorithm은 **max-product on parse forest**.

### Grid CRF (Image Segmentation)

각 픽셀에 label (object class). Pairwise potentials로 인접 픽셀이 같은 label일 가능성 장려.

$$p(y | x) \propto \exp\left(\sum_i \psi_i(y_i, x_i) + \sum_{(i, j) \in E} \psi_{ij}(y_i, y_j, x)\right)$$

- $\psi_i$: unary (CNN으로부터)
- $\psi_{ij}$: pairwise (Potts model, edge-preserving)

**Grid graph의 treewidth**: $O(\sqrt{N})$ where $N$ = pixels. $256 \times 256$ image에서 treewidth $\approx 256$ → **exact inference 불가능**. Approximation 필요.

### $\alpha$-Expansion (Boykov-Veksler-Zabih 2001)

Multi-label image segmentation의 approximate MAP:

1. 각 label $\alpha$에 대해 반복:
2. 각 pixel은 "현재 label 유지" 또는 "$\alpha$로 변경" 중 선택 → **binary** MRF
3. **Graph cut** (min-cut via max-flow)으로 optimal binary assignment
4. 수렴까지 반복

**Guarantee**: Potts 유사 pairwise에서 $2 \cdot \text{OPT}$ 근사 (factor-2 optimality).

### Structured SVM

CRF의 margin-based alternative (Tsochantaridis et al. 2005):
$$\min_w \frac{1}{2} \|w\|^2 + C \sum_i \max_{y \neq y^{(i)}} [L(y, y^{(i)}) + w \cdot f(y, x^{(i)}) - w \cdot f(y^{(i)}, x^{(i)})]$$

$L(y, y^{(i)})$: margin loss (e.g., Hamming). 가장 violating $y$에 대한 loss-augmented inference 필요.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — General CRF

임의 graph $\mathcal{G}$ over output variables $y = (y_1, \ldots, y_n)$와 observation $x$에 대해:
$$p(y | x; w) = \frac{1}{Z(x; w)} \prod_{C \in \text{cliques}(\mathcal{G})} \phi_C(y_C, x; w)$$

$\phi_C(y_C, x; w) = \exp(\sum_k w_k f_k(y_C, x))$.

### 정의 3.2 — Skip-Chain CRF

Linear chain + skip edges $S$:
$$\mathcal{G}: y_1 - y_2 - \cdots - y_n, \quad (y_i, y_j) \in S \text{ for some non-adjacent pairs}$$

Pairwise potential on chain + skip edges.

### 정의 3.3 — Tree CRF

Tree-structured output (dependency tree):
$$p(\text{tree} | x) \propto \prod_{(\text{parent}, \text{child}) \in \text{tree}} \phi(\text{parent}, \text{child}, x)$$

**Matrix-Tree Theorem** (Tutte 1948, Koo et al. 2007): non-projective dependency tree의 partition function은 Kirchhoff matrix determinant로 $O(n^3)$ 계산 가능.

### 정의 3.4 — Structured SVM

$$\min_w \frac{1}{2} \|w\|^2 + C \sum_i \xi_i$$
$$\text{s.t. } w \cdot [f(y^{(i)}, x^{(i)}) - f(y, x^{(i)})] \geq L(y, y^{(i)}) - \xi_i, \forall i, y$$

"$y^{(i)}$가 다른 어떤 $y$보다 $L$만큼 더 높은 score를 가져야 함".

---

## 🔬 정리와 증명

### 정리 3.1 — Loopy CRF의 복잡도

**명제**: General graph $\mathcal{G}$의 CRF inference 복잡도는 $\mathcal{G}$의 treewidth로 결정:
$$O(|V| \cdot K^{\omega(\mathcal{G}) + 1})$$

Exact inference는 low treewidth에서만 tractable.

**증명**: Ch2-04의 junction tree 복잡도. CRF는 MRF의 conditional 버전이므로 같은 bound. $\square$

### 정리 3.2 — Matrix-Tree Theorem for Dependency CRF

**명제** (Koo et al. 2007): Non-projective dependency tree distribution의 partition function $Z(x) = \sum_{\text{trees}} \exp(\sum w f)$은 **Kirchhoff matrix**의 determinant:
$$Z(x) = \det(L_{11}(x))$$

여기서 $L(x)$는 Laplacian matrix with edge weights = potential, $L_{11}$은 root 제거한 minor.

**증명 개요**: 

Tutte의 Matrix-Tree Theorem: 그래프의 spanning tree 수 = Laplacian의 principal minor determinant. 

Weighted extension: weight의 곱으로 spanning tree 열거.

**복잡도**: $O(n^3)$ for determinant. Linear-chain parsing이 $O(n^3)$ CKY와 비슷한 복잡도지만, **non-projective** dependency 허용.

$\square$

### 정리 3.3 — $\alpha$-Expansion의 Approximation Guarantee

**명제** (Boykov-Veksler-Zabih 2001): Potts model $\psi_{ij}(y_i, y_j) = \mathbb{1}[y_i \neq y_j] \cdot \lambda_{ij}$에서 $\alpha$-expansion은 **factor-2 approximation**:
$$\text{cost}(\hat y) \leq 2 \cdot \text{cost}(y^*)$$

**증명 개요** (rough):

각 iteration에서 $\alpha$-expansion은 global optimum의 일부. 전체 iteration 후 cost의 upper bound가 $2 \cdot \text{OPT}$로 bound됨.

정확한 증명은 cutting plane argument 사용 — 각 label class에 대해 optimal assignment가 binary MRF의 solution임을 보임. $\square$

### 정리 3.4 — Structured Perceptron의 수렴

**명제** (Collins 2002): Data가 margin $\gamma$로 분리 가능하면 structured perceptron은 $\leq R^2 / \gamma^2$ mistakes로 수렴.

**증명**: Novikoff의 perceptron theorem의 structured 확장. $R$ = max feature norm.

---

## 💻 NumPy로 검증

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 예시 1: Grid CRF로 image segmentation (binary)
def grid_crf_binary(unary, pairwise_strength, n_iter=50):
    """
    Simple binary grid CRF with mean-field.
    unary: (H, W, 2) — log P(y_i | x_i)
    pairwise_strength: Potts prior strength
    """
    H, W, K = unary.shape
    # Initialize: argmax of unary
    q = np.zeros((H, W, K))
    q[:] = 1.0 / K
    
    for it in range(n_iter):
        # Mean-field update
        new_q = np.copy(unary)  # log probabilities
        # Add pairwise contribution
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_q = np.roll(q, (dy, dx), axis=(0, 1))
            # Boundary: set to 0 (approximate)
            if dy == -1: shifted_q[-1] = 0
            elif dy == 1: shifted_q[0] = 0
            if dx == -1: shifted_q[:, -1] = 0
            elif dx == 1: shifted_q[:, 0] = 0
            
            # Potts: -strength * 1[y_i != y_j]
            # Add pairwise_strength * q[neighbor] for same label
            new_q += pairwise_strength * shifted_q
        
        # Normalize (softmax)
        new_q -= new_q.max(axis=-1, keepdims=True)
        new_q = np.exp(new_q)
        new_q /= new_q.sum(axis=-1, keepdims=True)
        
        q = new_q
    
    return q

# Synthetic noisy image
np.random.seed(0)
H, W = 40, 40
# True segmentation: simple shapes
true_seg = np.zeros((H, W))
true_seg[10:30, 10:30] = 1
# Noisy observation
noisy = true_seg + np.random.randn(H, W) * 0.8

# Unary: based on observed value
unary = np.zeros((H, W, 2))
unary[:, :, 0] = -(noisy - 0)**2 / 2.0  # log p(x | y=0)
unary[:, :, 1] = -(noisy - 1)**2 / 2.0  # log p(x | y=1)

# Mean-field CRF
q = grid_crf_binary(unary, pairwise_strength=0.5, n_iter=30)
seg_crf = q.argmax(axis=-1)

# 비교: unary만 (argmax, no smoothing)
seg_unary = unary.argmax(axis=-1)

# 정확도
acc_unary = (seg_unary == true_seg).mean()
acc_crf = (seg_crf == true_seg).mean()
print(f"Unary only accuracy:  {acc_unary:.3f}")
print(f"CRF accuracy:         {acc_crf:.3f}")
print(f"Improvement: {(acc_crf - acc_unary) * 100:.1f}%")

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(true_seg, cmap='gray')
axes[0].set_title('True segmentation')
axes[1].imshow(noisy, cmap='gray')
axes[1].set_title('Noisy observation')
axes[2].imshow(seg_unary, cmap='gray')
axes[2].set_title(f'Unary argmax ({acc_unary:.2f})')
axes[3].imshow(seg_crf, cmap='gray')
axes[3].set_title(f'CRF mean-field ({acc_crf:.2f})')
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.savefig('grid_crf_segmentation.png', dpi=120, bbox_inches='tight')
plt.show()

# 예시 2: Skip-chain CRF visualization
fig, ax = plt.subplots(figsize=(10, 3))
T_seq = 10
positions = {f'y_{t+1}': (t, 0) for t in range(T_seq)}

G = nx.Graph()
G.add_nodes_from(positions.keys())

# Chain edges
for t in range(T_seq - 1):
    G.add_edge(f'y_{t+1}', f'y_{t+2}')

# Skip edges (coreference example)
G.add_edge('y_1', 'y_5')
G.add_edge('y_1', 'y_9')
G.add_edge('y_5', 'y_9')

edge_colors = []
for u, v in G.edges():
    t_u = int(u.split('_')[1])
    t_v = int(v.split('_')[1])
    if abs(t_u - t_v) == 1:
        edge_colors.append('gray')
    else:
        edge_colors.append('red')

nx.draw(G, positions, ax=ax, with_labels=True, node_color='lightblue',
        edge_color=edge_colors, width=[1 if c == 'gray' else 2 for c in edge_colors],
        node_size=800, font_size=10)
ax.set_title('Skip-chain CRF (red = skip edges)')
plt.tight_layout()
plt.savefig('skip_chain_crf.png', dpi=120, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
Unary only accuracy:  0.762
CRF accuracy:         0.923
Improvement: 16.1%
```

CRF의 smoothness prior로 accuracy 크게 개선.

---

## 🔗 AI/ML 연결

### DeepLab Semantic Segmentation

Chen et al. 2016:
- CNN이 per-pixel class logits 생성 (unary)
- **Fully-Connected CRF** (Krähenbühl-Koltun 2011): 모든 pixel pair 연결 with Gaussian pairwise
- Mean-field inference로 efficient (5-10 iterations)
- 결과: PASCAL VOC mIoU +5% 개선

**Key insight**: Pairwise potential $\psi_{ij} = w_1 \exp(-\|p_i - p_j\|^2 / 2 \sigma_{\text{pos}}^2 - \|c_i - c_j\|^2 / 2\sigma_{\text{color}}^2)$. Object boundary에서 color 달라지면 pairwise 약해져 label 다를 수 있음 — **edge-aware smoothing**.

### Constituency Parsing

**CKY Algorithm** for probabilistic context-free grammar:
- Span $(i, j)$의 best parse score
- Recursion: $\delta(i, j, A) = \max_{k, B, C} \delta(i, k, B) \cdot \delta(k, j, C) \cdot p(A \to BC)$
- $O(n^3 \cdot |G|)$ — $n$ = sentence length, $|G|$ = grammar size

**Neural CKY**: Stern et al. 2017, Kitaev-Klein 2018.

### Dependency Parsing

**Non-projective** (순서 제약 없음): Chu-Liu-Edmonds MST algorithm $O(n^2)$.

**Projective**: Eisner algorithm $O(n^3)$.

**CRF training**: Matrix-tree theorem으로 $Z$ 계산 → gradient.

### Coreference Resolution

**Mention-pair model**: 각 mention pair $(i, j)$에 대해 "coreferent?" 이진 분류.
- Independence 가정이 비현실적 (cluster structure)
- **Cluster-ranking model**: 각 mention이 이전 cluster 중 하나에 attached → skip-chain structure
- Neural approaches: e2e-coref (Lee et al. 2017, 2018) — transformer-based features + antecedent selection

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Tractable inference | General graph는 intractable → approximation |
| Convex loss | Structured SVM은 non-smooth (max); cutting plane 필요 |
| Potts model | Non-metric pairwise는 graph cut 불가 |
| Exact decoding | Beam search 또는 dynamic programming approximation |

**주의**: Grid CRF with long-range edges (dense CRF)는 mean-field 가 유일한 실용적 방법. $N$-pixel에서 pair $O(N^2)$이므로 **permutohedral lattice** 같은 filter-based trick 필요.

---

## 📌 핵심 정리

| 구조 | 예시 | Inference |
|------|------|-----------|
| Linear chain | POS, NER | Forward-Backward $O(K^2 T)$ |
| Skip chain | Coreference | Loopy BP / approximate |
| Tree | Dependency parse | Matrix-tree / Eisner $O(n^3)$ |
| Grid | Image segmentation | Mean-field / graph cut |
| Arbitrary | Relational | Loopy BP / JT if treewidth small |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Grid CRF inference가 $\text{treewidth} = O(\sqrt N)$라서 unusable하지만 mean-field는 $O(N \cdot K \cdot I)$ ($I$ iterations)로 tractable. 왜?

<details>
<summary>힌트 및 해설</summary>

**Exact inference** (JT):
- Grid graph에서 treewidth $O(\sqrt{N})$
- Complexity $O(K^{\sqrt N + 1} \cdot N)$ — **exponential in $\sqrt N$**
- $N = 10^4$ pixel, $\sqrt N = 100$, $K = 20$ classes → $20^{101}$ — 불가능

**Mean-field**:
- Approximation: $q(y) = \prod_i q_i(y_i)$ (fully factorized)
- Update: $q_i^{\text{new}} \propto \exp(\text{unary} + \sum_{j \in N(i)} \text{expected pairwise})$
- Complexity: $O(N \cdot K \cdot |N(i)| \cdot I)$ — **linear in $N$**
- 일반적으로 $I = 5-10$ iterations에 수렴

**Trade-off**:
- Exact: 정확하지만 불가능
- Mean-field: 빠르지만 근사 (mode collapse, posterior underestimate)

**왜 mean-field가 실용적으로 작동**:
- Grid MRF with strong unary (good CNN output) → posterior가 거의 unimodal → mean-field가 reasonable
- 최근 **DC-CRF** (Differentiable CRF, Zheng et al. 2015): mean-field를 RNN으로 해석 → end-to-end 학습 가능

**대안**:
- **Graph cut** (binary, Potts): exact for binary, $\alpha$-expansion factor-2 approx for multi-label
- **ICM** (Iterated Conditional Modes): coordinate descent, local optima

결론: Intractable exact를 해결하려면 **structured approximation** 또는 **amortized inference** (FCN + dense CRF).

</details>

**문제 2** (심화): Dependency parsing의 **Eisner algorithm** ($O(n^3)$, projective)과 **MST** ($O(n^2)$, non-projective)의 trade-off를 분석하라.

<details>
<summary>힌트 및 해설</summary>

**Eisner (Projective)**:
- Assumption: dependency arc가 교차하지 않음 (projective)
- 영어, 중국어 등에서 95%+ 문장이 projective → 유효
- CKY-like DP: span $(i, j)$의 best subtree
- Exact MAP + $Z$ 계산 가능 (inside-outside for CRF training)
- $O(n^3)$, memory $O(n^2)$

**MST (Chu-Liu-Edmonds, Non-projective)**:
- 어떤 dependency graph도 허용 (교차 가능)
- 체코어, 독일어의 일부 — non-projective 빈도 높음
- MAP만 가능 ($O(n^2)$) — **partition function은 Matrix-Tree로 $O(n^3)$**
- Training은 either MST-driven margin 또는 CRF w/ Matrix-Tree

**언어별 선택**:
- English: Eisner — projective dominant
- Czech: MST — non-projective 필수
- 혼합: 종종 **arc-factored + projective** 사용 (간단하면서도 대부분 올바름)

**Neural parsers**:
- **Biaffine attention** (Dozat-Manning 2017): score $s(h, m)$ for arc $h \to m$, MST로 decode
- **Transformer + CKY** for constituency

**현대적 trend**: Non-projective + Matrix-Tree로 training (fully differentiable), decoding은 MST.

</details>

**문제 3** (AI 연결): Image segmentation에서 modern "dense CRF + CNN" 아키텍처가 "end-to-end U-Net / DeepLab"으로 대체되었다. CRF는 왜 사라졌고, 무엇이 그 역할을 대신하는가?

<details>
<summary>힌트 및 해설</summary>

**Dense CRF era (2012-2017)**:
- FCN (Fully Convolutional Network) + dense CRF post-processing
- CRF가 boundary refinement, smoothness 담당
- mIoU +3-5%

**End-to-end era (2018-)**:
- **DeepLab v3+**, **HRNet**: atrous convolution, multi-scale features, skip connections
- **U-Net / FPN**: encoder-decoder with skip connections → boundary 정보 보존
- CRF 없이도 boundary quality 우수

**CRF가 대체된 이유**:

1. **Deep features는 이미 local context**: Atrous/multi-scale conv가 CRF의 pairwise smoothing을 implicit하게 학습
2. **Computational cost**: Dense CRF inference가 느림 (5-10 iteration × large graph)
3. **End-to-end training이 더 강력**: CRF parameter를 별도 학습하는 것보다 전체 network가 학습하는 게 flexible
4. **Transformer의 등장**: **Segformer, SAM** — attention이 long-range dependency 자연스럽게 처리 → CRF의 역할 대체

**CRF의 유산 (modern contexts)**:
1. **Mean-field as RNN** (Zheng et al. 2015): CRF를 differentiable layer로 학습 가능한 형태
2. **CRF layer in NER**: BiLSTM-CRF, BERT-CRF는 여전히 small data에서 유용
3. **Structured prediction theory**: Consistency, calibration 분석의 framework
4. **Graph Neural Networks**: CRF message passing의 neural generalization

**언제 여전히 CRF 유용**:
- Small dataset + strong prior (medical imaging)
- Interpretability 필요 (transition matrix를 직접 볼 수 있음)
- Structured output with **hard constraints** (valid parse tree 등)

**결론**: CRF = "structural prior의 explicit 형식". Neural network가 이 prior를 **implicit하게** 학습하게 되면서 CRF는 sub-routine으로 축소. 하지만 structured prediction의 **이론적 토대**는 여전히 중요.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Linear-Chain CRF의 Inference와 Learning](./02-linear-chain-crf.md) | [📚 README](../README.md) | [04. Neural CRF와 딥러닝 통합 ▶](./04-neural-crf.md) |

</div>
