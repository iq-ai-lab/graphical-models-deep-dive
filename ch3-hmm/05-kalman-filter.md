# 05. Linear Dynamical System과 Kalman Filter

## 🎯 핵심 질문

- Linear Dynamical System(LDS)은 HMM의 어떤 연속 상태 일반화인가?
- **Kalman Filter**가 **Forward algorithm의 Gaussian analog**임을 어떻게 증명하는가?
- Rauch-Tung-Striebel smoother가 Backward algorithm의 Gaussian 버전인 이유는?
- Gaussian conjugacy가 왜 closed-form recursion을 가능하게 하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Kalman Filter**는 **연속 상태 sequence model의 원조**. Apollo 11 navigation, GPS, radar tracking, 자동차 센서 fusion, robotic SLAM, 경제 시계열 (Kalman-Hamilton 1989) — 공학의 거대한 영역. **State Space Model** 계보(Structured State Space, S4, Mamba)의 기초. Kalman = "Forward algorithm for continuous state" = Gaussian BP. 이 연결을 이해하면 HMM·LDS·Particle Filter(Ch6-05)·VAE가 **모두 같은 factor graph 구조**임이 드러난다. 또한 **EKF, UKF** 등 비선형 확장과 deep learning의 결합 (Recurrent Kalman Network) 등 현대 응용의 기반.

---

## 📐 수학적 선행 조건

- [Ch3-01 HMM의 정의](./01-hmm-definition.md)
- [Ch3-02 Forward-Backward Algorithm](./02-forward-backward.md)
- Multivariate Gaussian, matrix inversion lemma (Woodbury)
- Linear algebra: 양정치 행렬, matrix square root

---

## 📖 직관적 이해

### LDS: HMM의 Gaussian 일반화

LDS의 생성 모델:
$$z_t = F z_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$x_t = H z_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$
$$z_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

- $z_t \in \mathbb{R}^n$: **continuous hidden state**
- $x_t \in \mathbb{R}^m$: **observation**
- $F$: transition matrix
- $H$: observation matrix
- $Q, R$: process, measurement noise covariances

HMM과 비교:
| | HMM | LDS |
|--|-----|-----|
| State | Discrete $\{1, ..., N\}$ | Continuous $\mathbb{R}^n$ |
| Transition | Matrix $A$ | Linear + Gaussian noise |
| Emission | Matrix $B$ (discrete) | Linear + Gaussian noise |
| Inference | $\alpha, \beta$ vectors | $\mu, \Sigma$ (mean, covariance) |
| Exact | Yes (tree BP) | Yes (Gaussian conjugacy) |

### Kalman Filter의 직관

매 time step마다 두 단계:

**Predict** ($z_t | x_{1:t-1}$):
$$\hat z_{t|t-1} = F \hat z_{t-1|t-1}$$
$$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$

이전 posterior를 dynamics로 propagate — noise 더해 uncertainty 증가.

**Update** ($z_t | x_{1:t}$):
$$K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$$
$$\hat z_{t|t} = \hat z_{t|t-1} + K_t (x_t - H \hat z_{t|t-1})$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

**Kalman Gain** $K_t$는 "prediction vs observation" 신뢰도의 optimal weighting.

### Gaussian Conjugacy — 왜 Closed Form인가

$z_{t-1} | x_{1:t-1} \sim \mathcal{N}(\mu, \Sigma)$, linear Gaussian dynamics → $z_t | x_{1:t-1}, x_t$도 Gaussian.

이는 **conjugate prior의 linear Gaussian case**. 각 step에서 mean + covariance만 유지하면 exact posterior — Forward algorithm이 Gaussian에 대해 closed-form.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Linear Dynamical System

LDS는 다음 관계로 정의:
$$z_t = F z_{t-1} + G u_t + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$x_t = H z_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

- $u_t$: (optional) control input
- $w_t, v_t$: i.i.d. Gaussian noise (서로 독립)
- $z_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$

모든 변수 Gaussian → joint $(z_{1:T}, x_{1:T})$도 Gaussian.

### 정의 5.2 — Kalman Filter

**Goal**: $p(z_t | x_{1:t})$를 재귀적으로 계산.

Induction: $p(z_{t-1} | x_{1:t-1}) = \mathcal{N}(\hat z_{t-1}, P_{t-1})$라 가정.

**Predict**:
$$\hat z_{t|t-1} = F \hat z_{t-1|t-1} + G u_t$$
$$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$

**Innovation**:
$$\nu_t := x_t - H \hat z_{t|t-1} \quad (\text{prediction error})$$
$$S_t := H P_{t|t-1} H^T + R \quad (\text{innovation covariance})$$

**Update**:
$$K_t := P_{t|t-1} H^T S_t^{-1}$$
$$\hat z_{t|t} = \hat z_{t|t-1} + K_t \nu_t$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

### 정의 5.3 — RTS Smoother (Rauch-Tung-Striebel)

**Goal**: $p(z_t | x_{1:T})$ — 미래 관측까지 고려.

Backward pass (forward 완료 후):
$$J_t := P_{t|t} F^T P_{t+1|t}^{-1}$$
$$\hat z_{t|T} = \hat z_{t|t} + J_t (\hat z_{t+1|T} - \hat z_{t+1|t})$$
$$P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t^T$$

초기: $\hat z_{T|T}, P_{T|T}$ (filter result).

---

## 🔬 정리와 증명

### 정리 5.1 — Gaussian Conjugacy

**명제**: $z \sim \mathcal{N}(\mu, \Sigma)$이고 $x = Hz + v$, $v \sim \mathcal{N}(0, R)$이면 posterior $p(z | x) = \mathcal{N}(\mu_z, \Sigma_z)$로
$$\mu_z = \mu + \Sigma H^T (H \Sigma H^T + R)^{-1}(x - H\mu)$$
$$\Sigma_z = \Sigma - \Sigma H^T (H\Sigma H^T + R)^{-1} H \Sigma$$

**증명**:

Joint $(z, x)$는 Gaussian:
$$\begin{pmatrix} z \\ x \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \mu \\ H\mu \end{pmatrix}, \begin{pmatrix} \Sigma & \Sigma H^T \\ H\Sigma & H\Sigma H^T + R \end{pmatrix}\right)$$

Conditional Gaussian formula:
$$p(z | x) = \mathcal{N}(\mu + \Sigma H^T (H\Sigma H^T + R)^{-1}(x - H\mu), \Sigma - \Sigma H^T (H\Sigma H^T + R)^{-1} H \Sigma)$$

이것이 바로 Kalman update 공식. $\square$

### 정리 5.2 — Kalman Filter = Forward Algorithm on LDS

**명제**: Kalman filter의 $(\hat z_{t|t}, P_{t|t})$는 LDS factor graph에서의 sum-product forward message의 Gaussian 표현.

**증명**:

Forward message (HMM Forward algorithm, sum-product):
$$\alpha_t(z_t) = p(z_t, x_{1:t}) = p(x_t | z_t) \int p(z_t | z_{t-1}) \alpha_{t-1}(z_{t-1}) dz_{t-1}$$

$\alpha_{t-1}(z_{t-1})$이 Gaussian이면:
- $\int p(z_t | z_{t-1}) \alpha_{t-1}(z_{t-1}) dz_{t-1}$ = Gaussian convolution → Gaussian (predict)
- $p(x_t | z_t) \cdot \text{Gaussian}$ = Gaussian (Bayesian update by Gaussian conjugacy)

각 step에서 Gaussian 유지. Mean과 covariance의 recursion이 정확히 Kalman filter 공식.

**Explicit**: 
- $\alpha_t(z_t)$의 정규화된 Gaussian: $\mathcal{N}(\hat z_{t|t}, P_{t|t})$
- $p(z_t, x_{1:t}) = p(x_{1:t}) \cdot p(z_t | x_{1:t}) = c_t \cdot \mathcal{N}(\hat z_{t|t}, P_{t|t})$

$c_t$ = marginal likelihood of $x_{1:t}$. Log-likelihood accumulate 가능 (HMM의 scaling factor와 유사). $\square$

### 정리 5.3 — RTS Smoother = Forward-Backward on LDS

**명제**: RTS의 $(\hat z_{t|T}, P_{t|T})$는 LDS의 sum-product 메시지를 forward-backward로 결합한 Gaussian posterior.

**증명 개요**:

HMM의 posterior $\gamma_t(z_t) \propto \alpha_t \beta_t$에 해당하는 Gaussian version. RTS의 유도:

$p(z_t | x_{1:T}) = \int p(z_t | z_{t+1}, x_{1:T}) p(z_{t+1} | x_{1:T}) dz_{t+1}$

Markov property로 $p(z_t | z_{t+1}, x_{1:T}) = p(z_t | z_{t+1}, x_{1:t})$:
$$= \int p(z_t | z_{t+1}, x_{1:t}) p(z_{t+1} | x_{1:T}) dz_{t+1}$$

Gaussian conjugacy로 각 항이 Gaussian → 결합도 Gaussian. RTS 공식이 나옴. $\square$

### 정리 5.4 — 복잡도

**명제**: Kalman filter: $O(n^3 T)$ time ($n$ = state dim), RTS smoother: $O(n^3 T)$. 복잡도는 $n \times n$ matrix inversion 때문.

**증명**: 각 step에서 matrix inversion $O(n^3)$. $T$ steps. $\square$

Observations이 $m \ll n$이면 Information filter form이 더 efficient ($O(m^3)$ inversion).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

def kalman_filter(F, H, Q, R, mu0, Sigma0, obs):
    """Kalman filter."""
    T = len(obs)
    n = F.shape[0]
    
    mu = np.zeros((T, n))
    Sigma = np.zeros((T, n, n))
    mu_pred = np.zeros((T, n))
    Sigma_pred = np.zeros((T, n, n))
    log_likelihood = 0.0
    
    for t in range(T):
        # Predict
        if t == 0:
            mu_pred[t] = mu0
            Sigma_pred[t] = Sigma0
        else:
            mu_pred[t] = F @ mu[t-1]
            Sigma_pred[t] = F @ Sigma[t-1] @ F.T + Q
        
        # Innovation
        innovation = obs[t] - H @ mu_pred[t]
        S = H @ Sigma_pred[t] @ H.T + R
        K = Sigma_pred[t] @ H.T @ np.linalg.inv(S)
        
        # Log-likelihood (Gaussian)
        log_likelihood += -0.5 * (
            len(obs[t]) * np.log(2 * np.pi) +
            np.log(np.linalg.det(S)) +
            innovation @ np.linalg.inv(S) @ innovation
        )
        
        # Update
        mu[t] = mu_pred[t] + K @ innovation
        Sigma[t] = (np.eye(n) - K @ H) @ Sigma_pred[t]
    
    return mu, Sigma, mu_pred, Sigma_pred, log_likelihood

def rts_smoother(F, mu, Sigma, mu_pred, Sigma_pred):
    """RTS smoother."""
    T = len(mu)
    n = F.shape[0]
    
    mu_smooth = np.zeros((T, n))
    Sigma_smooth = np.zeros((T, n, n))
    mu_smooth[-1] = mu[-1]
    Sigma_smooth[-1] = Sigma[-1]
    
    for t in range(T - 2, -1, -1):
        J = Sigma[t] @ F.T @ np.linalg.inv(Sigma_pred[t+1])
        mu_smooth[t] = mu[t] + J @ (mu_smooth[t+1] - mu_pred[t+1])
        Sigma_smooth[t] = Sigma[t] + J @ (Sigma_smooth[t+1] - Sigma_pred[t+1]) @ J.T
    
    return mu_smooth, Sigma_smooth

# Example: 1D tracking with constant velocity
# State: [position, velocity]
dt = 0.1
F = np.array([[1, dt],
              [0, 1]])
H = np.array([[1, 0]])  # observe only position
Q = np.eye(2) * 0.01
R = np.array([[0.1]])

# True trajectory
np.random.seed(42)
T = 50
z_true = np.zeros((T, 2))
x_obs = np.zeros((T, 1))
z_true[0] = [0, 1]  # start at 0 with velocity 1
for t in range(T):
    if t > 0:
        z_true[t] = F @ z_true[t-1] + np.random.multivariate_normal([0, 0], Q)
    x_obs[t] = H @ z_true[t] + np.random.multivariate_normal([0], R)

# Initial
mu0 = np.array([0, 0])
Sigma0 = np.eye(2) * 1.0

# Filter
mu_f, Sigma_f, mu_p, Sigma_p, logL = kalman_filter(F, H, Q, R, mu0, Sigma0, x_obs)
print(f"Kalman filter log-likelihood: {logL:.4f}")

# Smoother
mu_s, Sigma_s = rts_smoother(F, mu_f, Sigma_f, mu_p, Sigma_p)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Position
axes[0].plot(z_true[:, 0], 'g-', label='True position', linewidth=2)
axes[0].plot(x_obs[:, 0], 'r.', label='Observations', markersize=5)
axes[0].plot(mu_f[:, 0], 'b--', label='Filtered mean', linewidth=1.5)
axes[0].plot(mu_s[:, 0], 'k-', label='Smoothed mean', linewidth=1.5)
# Filter uncertainty
std_f = np.sqrt(Sigma_f[:, 0, 0])
axes[0].fill_between(range(T), mu_f[:, 0] - 2*std_f, mu_f[:, 0] + 2*std_f, alpha=0.2, color='blue', label='Filter ±2σ')
std_s = np.sqrt(Sigma_s[:, 0, 0])
axes[0].fill_between(range(T), mu_s[:, 0] - 2*std_s, mu_s[:, 0] + 2*std_s, alpha=0.2, color='black', label='Smoother ±2σ')
axes[0].set_ylabel('Position')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Velocity (hidden — not directly observed)
axes[1].plot(z_true[:, 1], 'g-', label='True velocity', linewidth=2)
axes[1].plot(mu_f[:, 1], 'b--', label='Filter', linewidth=1.5)
axes[1].plot(mu_s[:, 1], 'k-', label='Smoother', linewidth=1.5)
axes[1].set_ylabel('Velocity (hidden)')
axes[1].set_xlabel('t')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('kalman_filter_demo.png', dpi=120, bbox_inches='tight')
plt.show()

# Smoothing improves accuracy
mse_filter = np.mean((mu_f - z_true)**2)
mse_smooth = np.mean((mu_s - z_true)**2)
print(f"\nFilter MSE: {mse_filter:.4f}")
print(f"Smoother MSE: {mse_smooth:.4f}")
print(f"Smoother improvement: {(mse_filter - mse_smooth) / mse_filter * 100:.1f}%")
```

**출력 예시**:
```
Kalman filter log-likelihood: -43.8215

Filter MSE: 0.0318
Smoother MSE: 0.0184
Smoother improvement: 42.1%
```

Smoother가 filter보다 42% 낮은 MSE — 미래 정보 활용의 이점.

---

## 🔗 AI/ML 연결

### Apollo 11 Navigation (1969)

Rudolf Kalman의 1960 논문이 Apollo 우주선 navigation에 적용 — trajectory estimation, guidance. 오늘날 GPS 수신기도 Kalman filter 내장.

### Robotics SLAM

Simultaneous Localization and Mapping (SLAM):
- State: robot pose + map landmarks
- Observations: sensor readings (LiDAR, camera, wheel encoder)
- Non-linear → **EKF-SLAM** (Extended Kalman Filter)
- Modern: **particle filter** (FastSLAM) 또는 **factor graph SLAM** (iSAM, GTSAM)

### Financial Time Series

State space model for:
- Regime-switching: state = bull/bear market
- Stochastic volatility: state = latent volatility
- Kalman-Hamilton filter for regime-switching GARCH

### Deep Learning: S4, Mamba

Recent work (Gu et al. 2021, Mamba 2023):
$$h_t' = \bar A h_t + \bar B x_t, \quad y_t = C h_t + D x_t$$

Discretized LDS where $(\bar A, \bar B, C, D)$ learned. HiPPO theory로 long-range dependency. Transformer alternative with $O(T)$ complexity.

### Recurrent Kalman Network

Neural network + Kalman filter hybrid (Becker et al. 2019):
- Encoder: observations → latent space
- LDS in latent space
- Decoder: latent → predictions

Combines Kalman's uncertainty quantification with neural feature learning.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear dynamics | 비선형은 EKF (linearization) 또는 UKF (sigma points) 필요 |
| Gaussian noise | Heavy-tailed, multi-modal은 particle filter |
| Known $F, H, Q, R$ | System ID로 학습 (EM for LDS) |
| Numerical stability | $P_{t \mid t}$가 ill-conditioned 되면 divergence; square-root filter 사용 |

**주의**: EKF는 강한 non-linearity에서 발산 가능. **Unscented Kalman Filter (UKF)**는 sigma-point propagation으로 개선. 극단적 non-linearity / multi-modal에서는 **particle filter** (Ch6-05).

---

## 📌 핵심 정리

$$\boxed{\text{Predict: } (\hat z, P) \to (F\hat z, FPF^T + Q); \quad \text{Update: } \hat z \to \hat z + K(x - H\hat z)}$$

| 단계 | HMM 대응 |
|------|---------|
| Predict | Transition marginalization $\sum_{z_{t-1}} A \alpha_{t-1}$ |
| Update | Emission conditioning $B \cdot \alpha$ |
| RTS smoother | Forward-Backward gamma |
| Gaussian conjugacy | 각 step이 closed-form |

**Kalman = Forward on LDS = Gaussian sum-product**.  
**RTS = Forward-Backward on LDS**.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Kalman gain $K_t = P_{t|t-1} H^T S_t^{-1}$의 극단 케이스를 분석하라.
(a) Process noise $Q \to \infty$ (dynamics 매우 불확실)
(b) Measurement noise $R \to \infty$ (관측이 매우 noisy)

<details>
<summary>힌트 및 해설</summary>

(a) **$Q \to \infty$**:
$P_{t|t-1} = F P_{t-1|t-1} F^T + Q \to \infty$. 따라서:
$$K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1} \to H^T (HH^T)^{-1}$$

Update: $\hat z_{t|t} = \hat z_{t|t-1} + K_t (x_t - H \hat z_{t|t-1})$. 
$K_t H \to I$ (approximately), 따라서 $\hat z_{t|t} \approx x_t$ (projected onto observation subspace). **Prediction ignored, observation만 믿음**.

(b) **$R \to \infty$**:
$S_t = H P_{t|t-1} H^T + R \to \infty$. 따라서:
$$K_t = P_{t|t-1} H^T S_t^{-1} \to 0$$

Update: $\hat z_{t|t} \to \hat z_{t|t-1}$, $P_{t|t} \to P_{t|t-1}$. **Observation ignored, prediction만 신뢰**.

**일반적 해석**: Kalman gain은 **prediction과 observation의 신뢰도 비율**. $Q$ 커질수록 prediction 덜 신뢰, $R$ 커질수록 observation 덜 신뢰. 이것이 Kalman filter의 **optimal Bayesian fusion**.

</details>

**문제 2** (심화): LDS의 EM algorithm (parameter 학습)에서 E-step과 M-step을 유도하라.

<details>
<summary>힌트 및 해설</summary>

**LDS EM**: $\theta = (F, H, Q, R, \mu_0, \Sigma_0)$.

**E-step**: Kalman filter + RTS smoother로
- $\mathbb{E}[z_t | x] = \hat z_{t|T}$
- $\text{Cov}[z_t | x] = P_{t|T}$
- $\mathbb{E}[z_t z_{t-1}^T | x] = J_{t-1} P_{t|T} + \hat z_{t-1|T} \hat z_{t|T}^T$ (lag-1 covariance)

**M-step** (maximize Q function):

**$F$**:
$$F = \left(\sum_{t=2}^T \mathbb{E}[z_t z_{t-1}^T]\right) \left(\sum_{t=2}^T \mathbb{E}[z_{t-1} z_{t-1}^T]\right)^{-1}$$

**$Q$**:
$$Q = \frac{1}{T-1} \sum_{t=2}^T \left(\mathbb{E}[z_t z_t^T] - F \mathbb{E}[z_{t-1} z_t^T] - \mathbb{E}[z_t z_{t-1}^T] F^T + F \mathbb{E}[z_{t-1} z_{t-1}^T] F^T\right)$$

**$H$** and **$R$**: 비슷한 normal equation.

**$\mu_0, \Sigma_0$**: $\hat z_{1|T}$ and $P_{1|T}$.

**수렴**: Baum-Welch처럼 monotonic ELBO 증가. Local optima 문제 동일.

**역사**: Shumway-Stoffer 1982가 LDS EM을 시계열 econometrics에 확립. 오늘날 state-space time series analysis의 표준.

</details>

**문제 3** (AI 연결): S4 (Structured State Space)의 discretization $\bar A = e^{A \Delta t}$, $\bar B = A^{-1}(e^{A\Delta t} - I) B$가 LDS의 어떤 연속-이산 변환인지, 그리고 왜 "sequence length generalization"을 가능하게 하는지 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Continuous LDS** (S4의 기반):
$$h'(t) = A h(t) + B u(t)$$
$$y(t) = C h(t) + D u(t)$$

연속 ODE. Exact solution:
$$h(t + \Delta) = e^{A \Delta} h(t) + \int_0^\Delta e^{A(\Delta - s)} B u(t + s) ds$$

**Zero-Order Hold (ZOH) discretization**: $u(t + s) = u(t)$ constant over step:
$$h(t + \Delta) = e^{A\Delta} h(t) + A^{-1}(e^{A\Delta} - I) B u(t)$$

정의: $\bar A := e^{A\Delta}$, $\bar B := A^{-1}(e^{A\Delta} - I) B$. 이산 LDS:
$$h_{t+1} = \bar A h_t + \bar B u_t$$

**Sequence length generalization**:
- Continuous model이 ground truth
- $\Delta t$ 바꿔도 같은 dynamics → 훈련시와 다른 길이의 sequence에 일반화
- 대조: Transformer는 positional encoding이 fixed length, Mamba 이전 RNN은 length에 따라 부동

**HiPPO theory** (Gu et al. 2020): $A$를 **high-order polynomial projection** matrix로 설정 → long-range dependency memorization. 과거 신호를 polynomial basis에 projection.

**Selective 메커니즘 (Mamba)**: $\bar A, \bar B$를 input dependent하게 ($\Delta$가 input에 따라 달라짐). "관심 있는 정보는 오래 기억, 관심 없는 정보는 빨리 잊음" — selective state space.

**결론**: S4/Mamba = **Kalman filter의 learnable, selective version**. HMM → LDS → S4 → Mamba의 계보가 하나의 **state space model evolution**. Ch7-05에서 자세히.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Baum-Welch — EM for HMM](./04-baum-welch.md) | [📚 README](../README.md) | [Ch4-01 CRF의 정의 — Logistic Regression의 구조화 확장 ▶](../ch4-crf/01-crf-definition.md) |

</div>
