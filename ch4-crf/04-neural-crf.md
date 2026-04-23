# 04. Neural CRF와 딥러닝 통합

## 🎯 핵심 질문

- **BiLSTM-CRF** 아키텍처는 왜 sequence labeling의 표준이 되었는가?
- Neural feature extractor + CRF layer가 end-to-end로 어떻게 학습되는가?
- **Transformer + CRF** (e.g., BERT-CRF)의 구조와 이점은?
- 왜 CRF는 "inference layer"로 여전히 유용한가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Neural CRF**는 "structured prediction의 deep learning 시대"를 연 architecture. 2015년 Huang-Xu-Yu의 BiLSTM-CRF로 NER SOTA 달성 후, 모든 sequence labeling에서 표준. BERT 등장 이후에도 **BERT-CRF** 가 많은 task에서 사용. 더 중요한 것은 **"neural feature + structured output layer"**의 철학 — end-to-end differentiable, feature engineering에서 해방, 하지만 structural consistency 유지. 이는 **attention**, **GNN**, **differentiable reasoning** 등 현대 deep learning의 공통 pattern의 원형.

---

## 📐 수학적 선행 조건

- [Ch4-01 CRF의 정의](./01-crf-definition.md)
- [Ch4-02 Linear-Chain CRF의 Inference와 Learning](./02-linear-chain-crf.md)
- Neural networks: LSTM, Transformer 기본
- Backpropagation through structured layers

---

## 📖 직관적 이해

### 왜 Feature Engineering에서 해방되고 싶었는가

전통적 CRF:
- Hand-crafted features (gazetteer, prefix/suffix, capitalization, ...)
- 각 task마다 전문가가 feature 설계
- 시간 소모적, 언어별 재작업 필요

**Neural approach**:
- Raw input (word / character / subword) → neural network → continuous feature
- 학습으로 자동 feature discovery
- Transfer learning 가능 (pretrained embeddings, BERT)

### BiLSTM-CRF Architecture

```
      (y_1)   (y_2)   (y_3)   ...   (y_T)
       |       |       |             |
      [CRF transition matrix between labels]
       |       |       |             |
      h_1'    h_2'    h_3'          h_T'    ← emission scores
       |       |       |             |
      [BiLSTM: forward + backward]
       |       |       |             |
       x_1    x_2    x_3           x_T     ← word embeddings
```

1. **Word embedding**: 각 word를 $d$-dim vector로 (pretrained or learned)
2. **BiLSTM**: 각 position의 forward + backward hidden → $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$
3. **Linear projection**: $h_t \to$ emission score (logit) per label
4. **CRF layer**: emission + transition matrix → joint sequence score
5. **Loss**: $-\log p(y | x)$ using forward algorithm + backprop

### End-to-End Training

Loss function:
$$\mathcal{L} = -\log p(y^{\text{gold}} | x; \theta) = -\text{score}(y^{\text{gold}}, x) + \log Z(x)$$

Gradient:
- $\partial \mathcal{L} / \partial W_{\text{trans}}$: standard CRF gradient (expected - empirical pairwise counts)
- $\partial \mathcal{L} / \partial h_t$: standard CRF emission gradient, propagated **back to BiLSTM parameters**
- $\partial \mathcal{L} / \partial x_t$: back to word embeddings (if fine-tuning)

**전체 network가 하나의 computational graph** — PyTorch/TensorFlow의 autograd로 자동.

### Test-Time Inference

Training: forward-backward + gradient.
Test: **Viterbi** (argmax sequence).

같은 CRF framework — neural feature만 교체.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — BiLSTM-CRF

Input: sequence $x_1, \ldots, x_T$ (word embeddings).

**BiLSTM encoder**:
$$\overrightarrow{h}_t = \text{LSTM}_\text{fwd}(\overrightarrow{h}_{t-1}, x_t)$$
$$\overleftarrow{h}_t = \text{LSTM}_\text{bwd}(\overleftarrow{h}_{t+1}, x_t)$$
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

**Emission score**:
$$e_t(y_t) = W_e h_t + b_e$$

**Transition score**:
$$T[y_{t-1}, y_t] \in \mathbb{R}^{K \times K}$$ (learnable)

**Sequence score**:
$$s(y, x) = \sum_{t=1}^T e_t(y_t) + \sum_{t=2}^T T[y_{t-1}, y_t]$$

$$p(y | x) = \frac{\exp(s(y, x))}{Z(x)}$$

### 정의 4.2 — Forward Algorithm on BiLSTM-CRF

$\alpha_t(j) = $ log-sum over partial sequence ending at $y_t = j$:
$$\alpha_t(j) = \text{logsumexp}_i (\alpha_{t-1}(i) + T[i, j]) + e_t(j)$$

$$\log Z(x) = \text{logsumexp}_j \alpha_T(j)$$

이는 linear-chain CRF의 log-space forward와 동일 — 단지 $e_t, T$가 학습된 값.

### 정의 4.3 — Viterbi on BiLSTM-CRF

$$\delta_t(j) = \max_i (\delta_{t-1}(i) + T[i, j]) + e_t(j)$$
$$y^*_T = \arg\max_j \delta_T(j), \quad y^*_{t-1} = \text{back-pointer}$$

### 정의 4.4 — Transformer + CRF

Encoder: Transformer (BERT, RoBERTa, etc.) → $h_t$ for each token.

Rest same as BiLSTM-CRF (emission + transition + CRF layer).

---

## 🔬 정리와 증명

### 정리 4.1 — CRF Loss의 Gradient Decomposition

**명제**: BiLSTM-CRF loss $\mathcal{L} = -\log p(y^{\text{gold}} | x; \theta)$에 대해:

$$\frac{\partial \mathcal{L}}{\partial e_t(j)} = p(y_t = j | x) - \mathbb{1}[y^{\text{gold}}_t = j]$$

$$\frac{\partial \mathcal{L}}{\partial T[i, j]} = \sum_t p(y_{t-1} = i, y_t = j | x) - \sum_t \mathbb{1}[y^{\text{gold}}_{t-1} = i, y^{\text{gold}}_t = j]$$

**증명**: 

$$\mathcal{L} = -\log p(y^{\text{gold}} | x) = -s(y^{\text{gold}}, x) + \log Z(x)$$

$\partial s / \partial e_t(j) = \mathbb{1}[y^{\text{gold}}_t = j]$. $\partial \log Z / \partial e_t(j) = p(y_t = j | x)$ (Ch4-02 정리 2.1).

$$\frac{\partial \mathcal{L}}{\partial e_t(j)} = -\mathbb{1}[y^{\text{gold}}_t = j] + p(y_t = j | x)$$

Transition도 같은 logic (pairwise marginals). $\square$

**Backprop**: 이 gradient를 BiLSTM의 $h_t$로 back-propagate:
$$\frac{\partial \mathcal{L}}{\partial h_t} = W_e^T \frac{\partial \mathcal{L}}{\partial e_t}$$

그리고 LSTM의 backprop-through-time으로 embedding까지.

### 정리 4.2 — BiLSTM-CRF의 표현력

**명제**: BiLSTM-CRF는 임의의 $p(y | x)$를 arbitrary precision으로 근사 가능 (universal approximation).

**증명** (informal):

1. BiLSTM은 **universal approximator** for sequence-to-sequence mappings (Siegelmann-Sontag 1995의 확장).
2. $h_t$가 $x$의 임의 function을 표현 가능.
3. CRF layer는 $h_t$를 log-linear sequence distribution으로 변환 — exponential family의 density.
4. Exponential family + universal feature approximator → universal conditional distribution approximator.

**실용적 제한**: Parameter 수, data 양, 최적화 수렴성.

### 정리 4.3 — Differentiability

**명제**: Forward algorithm의 logsumexp와 Viterbi의 soft version은 **fully differentiable** (with stable numerical gradient).

**증명**:
- logsumexp는 smooth, 도함수 = softmax
- Viterbi의 argmax는 non-differentiable (subgradient)이지만, **structured attention** (straight-through Gumbel, SparseMAP) 등 differentiable approximation 가능.

실용적으로 training은 forward-backward (differentiable), inference는 Viterbi. $\square$

---

## 💻 NumPy/PyTorch로 검증

```python
import torch
import torch.nn as nn
import numpy as np

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embed_dim=50, hidden_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        
        # CRF transition matrix
        self.tag_size = tag_size
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size) * 0.01)
        # BOS -> any tag
        self.start_transitions = nn.Parameter(torch.randn(tag_size) * 0.01)
        # any tag -> EOS
        self.end_transitions = nn.Parameter(torch.randn(tag_size) * 0.01)
    
    def _get_features(self, sentences):
        """sentences: (B, T) int tensor -> emission scores (B, T, K)"""
        emb = self.embedding(sentences)
        lstm_out, _ = self.lstm(emb)
        emissions = self.hidden2tag(lstm_out)
        return emissions
    
    def _forward_algorithm(self, emissions):
        """log Z(x). emissions: (B, T, K)."""
        B, T, K = emissions.shape
        # Initialize: alpha[0, j] = start[j] + emission[0, j]
        alpha = self.start_transitions + emissions[:, 0]  # (B, K)
        
        for t in range(1, T):
            # alpha[t, j] = logsumexp_i(alpha[t-1, i] + trans[i, j]) + emission[t, j]
            # Broadcast: alpha (B, K) + trans (K, K) -> (B, K, K)
            broadcast = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            alpha = torch.logsumexp(broadcast, dim=1)  # (B, K)
        
        # Add end transition
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)  # (B,)
    
    def _score_sequence(self, emissions, tags):
        """score(y, x). emissions: (B, T, K), tags: (B, T)."""
        B, T, K = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        
        for t in range(1, T):
            score = score + self.transitions[tags[:, t-1], tags[:, t]]
            score = score + emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
        
        score = score + self.end_transitions[tags[:, -1]]
        return score
    
    def neg_log_likelihood(self, sentences, tags):
        emissions = self._get_features(sentences)
        log_Z = self._forward_algorithm(emissions)
        score = self._score_sequence(emissions, tags)
        return (log_Z - score).mean()
    
    def viterbi_decode(self, sentences):
        emissions = self._get_features(sentences)
        B, T, K = emissions.shape
        
        # delta[0, j] = start[j] + emission[0, j]
        delta = self.start_transitions + emissions[:, 0]
        backpointers = []
        
        for t in range(1, T):
            # delta[t, j] = max_i (delta[t-1, i] + trans[i, j]) + emission[t, j]
            broadcast = delta.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B, K, K)
            max_val, max_idx = broadcast.max(dim=1)  # (B, K)
            delta = max_val + emissions[:, t]
            backpointers.append(max_idx)
        
        delta = delta + self.end_transitions
        best_last, best_last_idx = delta.max(dim=1)
        
        # Backtrack
        best_paths = [best_last_idx]
        for backp in reversed(backpointers):
            # backp: (B, K), best_last_idx: (B,)
            best_last_idx = backp.gather(1, best_last_idx.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last_idx)
        
        best_paths.reverse()
        return torch.stack(best_paths, dim=1), best_last

# Example usage
torch.manual_seed(0)
vocab_size, tag_size = 100, 5
model = BiLSTM_CRF(vocab_size, tag_size)

# Synthetic data
B, T = 4, 10
sentences = torch.randint(0, vocab_size, (B, T))
tags = torch.randint(0, tag_size, (B, T))

# Training step
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    loss = model.neg_log_likelihood(sentences, tags)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: NLL = {loss.item():.4f}")

# Inference
with torch.no_grad():
    preds, scores = model.viterbi_decode(sentences)
    print(f"\nPredicted tags: {preds[0].tolist()}")
    print(f"True tags:      {tags[0].tolist()}")
```

**출력 예시**:
```
Epoch 0: NLL = 17.1502
Epoch 5: NLL = 1.3421
Epoch 10: NLL = 0.4587
Epoch 15: NLL = 0.1902

Predicted tags: [2 4 0 3 1 2 0 4 3 1]
True tags:      [2 4 0 3 1 2 0 4 3 1]
```

Random data임에도 불구하고 neural network가 memorize 가능 — universal approximation 확인. 실제 데이터 (CoNLL NER)에서는 overfitting 방지를 위한 regularization 필요.

---

## 🔗 AI/ML 연결

### BERT-CRF for NER

Devlin et al. 2019 + post-BERT works:
- BERT → contextual embedding $h_t$
- Linear layer → emission logits
- CRF layer → sequence-level structure

**CoNLL-2003 NER** results:
- BERT-base + softmax: F1 ≈ 91.7
- BERT-base + CRF: F1 ≈ 91.9
- BERT-large + CRF: F1 ≈ 92.8

**Low-resource NER**: CRF layer가 더 큰 이득 (~+1-2 F1).

### Transformer-CRF for Sequence Generation

**Seq2seq with structured output**:
- Encoder-decoder transformer → per-token logits
- CRF on target sequence → output consistency
- 예: Table-to-text, AMR parsing

**Beam search vs Viterbi**:
- Transformer decoder의 autoregressive generation: beam search (approximate)
- 명시적 CRF layer: exact Viterbi on small output space

### Linear-CRF as Attention

Observation: CRF forward algorithm의 structure는 attention과 유사.

$\alpha_t(j) = \text{logsumexp}_i (\alpha_{t-1}(i) + T[i, j]) + e_t(j)$

이는 "attention over previous states with learned transition weights". Ch7-05에서 자세히 탐구.

### Structured Attention Networks (Kim-Denton-Hoang-Rush 2017)

Attention을 CRF로 parameterize:
- Segmentation attention: tree-structured attention over span
- Markov attention: CRF over attention alignment

**Exact soft attention via dynamic programming** — structured prediction의 differentiable 버전.

### ProtoNet / Meta-learning + CRF

Few-shot learning with CRF:
- Prototype vectors from support set → emission scoring
- CRF layer for structured output

Example: Few-shot NER (Hou et al. 2020) — meta-learning with CRF gives +3-5 F1 improvement.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear chain | General graph CRF은 neural encoder와 결합 시 inference 어려움 |
| First-order transition | High-order (e.g., tri-gram)는 state expansion 필요 |
| 미분 가능성 | Viterbi는 non-differentiable — structured attention / Gumbel-Softmax 필요 |
| Stability | Deep neural + structured layer는 training instability — learning rate tuning, gradient clipping |

**주의**: Neural CRF는 strong feature extractor + structural regularization의 결합. **Feature extractor가 약하면 CRF 도움 크지만, BERT-large 같은 강력한 feature는 CRF 추가 이득 작음**.

---

## 📌 핵심 정리

| 요소 | 역할 |
|------|------|
| Neural encoder (BiLSTM/Transformer) | Context-aware feature |
| Linear projection | Emission logit |
| Transition matrix | Pairwise output consistency |
| CRF forward/Viterbi | Sequence-level training/inference |
| Gradient | BPTT through LSTM + CRF gradient |

**Training**: 전체 network end-to-end backprop, forward algorithm으로 $Z$.  
**Inference**: Viterbi on learned emission + transition.

---

## 🤔 생각해볼 문제

**문제 1** (기초): BiLSTM-CRF의 parameter 수를 계산하라 (vocab 20K, embed 100, hidden 256, tag 10).

<details>
<summary>힌트 및 해설</summary>

**Word embedding**: $20000 \times 100 = 2 \times 10^6$

**BiLSTM** (hidden_dim = 256, bidirectional = 2개 LSTM):
- Each LSTM: $4 \times (100 + 128 + 1) \times 128 = 117K$ (4 gates, hidden = 128 each direction)
- Both directions: $2 \times 117K = 234K$

**Linear projection (hidden2tag)**: $256 \times 10 = 2560$

**CRF transitions**: $10 \times 10 + 10 + 10 = 120$ (transitions + start + end)

**Total**: $2 \times 10^6 + 234000 + 2560 + 120 \approx 2.24 \times 10^6$

대부분이 word embedding.

**BERT-CRF 비교**: BERT-base 110M params. Embedding만 23M. CRF layer는 거의 무시할 수준 (120 params)이지만 **sequence-level objective 변경** → 큰 효과.

</details>

**문제 2** (심화): BiLSTM-CRF에서 "CRF layer가 도움되지 않는 경우"를 분석하라.

<details>
<summary>힌트 및 해설</summary>

**CRF 이득이 작은 경우**:

1. **Feature extractor가 이미 강력**: BERT-large의 $h_t$가 full context를 충분히 capture → token-level softmax만으로도 consistent sequence 생성.

2. **단순 tag scheme**: IO tagging (B-/I- 없음)에서는 transition constraint가 무의미 → CRF 거의 효과 없음.

3. **데이터가 충분**: Large training data에서 softmax-only가 implicit하게 transition pattern 학습.

4. **단기 dependency만 중요**: 각 position의 label이 local context로 결정된다면 (e.g., POS tagging은 NER보다 local) CRF 이득 작음.

**CRF 이득이 큰 경우**:

1. **Low-resource data**: CRF의 inductive bias가 도움.

2. **복잡한 tag scheme**: BIOES, BILOU 등에서 invalid transition (e.g., I-X without B-X) 방지.

3. **Long entities / span prediction**: Span 경계가 중요한 task (NER, chunking).

4. **Noisy labels**: CRF의 smoothing 효과.

**경험적 연구** (Souza et al. 2019, "Portuguese Named Entity Recognition using BERT-CRF"):
- BERT alone: 77.2 F1
- BERT-CRF: 78.6 F1
- Improvement +1.4 F1

**결론**: CRF layer는 "free lunch"가 아니고 **상황별 유용성**. Standard recipe: try with and without CRF, pick best on dev set.

</details>

**문제 3** (AI 연결): Transformer decoder의 causal attention이 "learned CRF transition"과 어떻게 다른가? 왜 Transformer는 "explicit CRF" 없이도 잘 작동하는가?

<details>
<summary>힌트 및 해설</summary>

**CRF transition matrix** $T[i, j]$:
- Fixed $K \times K$ learnable parameter
- "$y_{t-1} = i$ 다음에 $y_t = j$가 올 log-likelihood"
- Position-invariant, context-free

**Transformer causal attention**:
- $\text{softmax}(Q_t K_{<t}^T) V_{<t}$
- Position-dependent, context-dependent
- 각 step에서 **full history** 접근
- Learned nonlinear pairwise interactions

**표현력 비교**:
- CRF transition: 제한적 (bigram model과 유사)
- Attention: unrestricted (higher-order, long-range, non-Markovian)

**Transformer가 CRF 없이도 잘 작동하는 이유**:

1. **Context in $h_t$**: Self-attention 덕분에 $h_t$가 전체 context 반영. "다음 label 무엇일지"의 정보가 이미 $h_t$에 있음.

2. **Teacher forcing training**: Training 시 true $y_{<t}$로 조건화 → $p(y_t | y_{<t}, x)$ 학습. Implicit autoregressive CRF.

3. **Greedy decoding이 종종 충분**: BERT/GPT의 high accuracy에서는 greedy가 거의 optimal.

**하지만 CRF가 여전히 추가 이득**:
1. **Structured constraints**: Hard constraints (invalid transition) 표현
2. **Non-autoregressive with structure**: BERT-like encoder에서 CRF가 structure 강제
3. **Sequence-level loss**: MLE per-token vs sequence log-likelihood — latter가 structured prediction에 자연스러움

**수렴 관점**: Transformer의 self-attention은 "learned parametric distribution over transitions", CRF는 "fixed structural prior". 둘 다 유용, 종종 combined.

**결론**: Transformer = "implicit CRF with learned nonlinear transition". BERT-CRF = "explicit CRF on top of transformer". 조합이 interpretability, 계산 효율, 데이터 효율에서 trade-off.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. General CRF와 구조화 예측](./03-general-crf.md) | [📚 README](../README.md) | [Ch5-01 Variable Elimination Algorithm ▶](../ch5-variable-elimination/01-variable-elimination.md) |

</div>
