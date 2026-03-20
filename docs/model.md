 # Hidden Markov Model for Go/Nogo Behavioral Data

## Overview

We implement a Hidden Markov Model (HMM) to identify latent **performance states** in a mouse Go/Nogo discrimination task. The core idea is that the animal is not behaving consistently across trials — it switches between a small number of hidden behavioral modes (e.g. optimal, impulsive, disengaged), and we want to recover these modes from the sequence of observed (stimulus, lick) pairs.

---

## Model Definition

### Observed variables

On each trial $t$, we observe two quantities:

- **Stimulus** $u_t \in \{+1, -1\}$ — presented to the animal (Go or Nogo), known by the experimenter
- **Lick** $y_t \in \{0, 1\}$ — the animal's binary response

Trials with `answer_window = 0` (state 5) are excluded before fitting.

### Hidden variable

On each trial $t$, the animal is in one of $K$ discrete **hidden states**:

$$z_t \in \{1, 2, \ldots, K\}$$

The state is never directly observed — it is inferred from the sequence of (stimulus, lick) pairs.

### Parameters

The model has three sets of parameters:

**Initial state distribution** $\boldsymbol{\pi}$ — a vector of length $K$:

$$\pi_k = P(z_1 = k)$$

**Transition matrix** $\mathbf{A}$ — a $K \times K$ row-stochastic matrix:

$$A_{kk'} = P(z_{t+1} = k' \mid z_t = k)$$

**Emission parameters** — two probabilities per state $k$:

$$p_{\text{go}}^{(k)} = P(y_t = 1 \mid u_t = +1,\ z_t = k)$$
$$p_{\text{nogo}}^{(k)} = P(y_t = 1 \mid u_t = -1,\ z_t = k)$$

### Why this emission model?

With a binary stimulus $u_t \in \{+1, -1\}$, each hidden state is fully characterized by just two numbers: the probability of licking on Go trials and the probability of licking on Nogo trials. This places each state as a point in the unit square $[0,1]^2$, which maps naturally onto signal detection theory:

| State type | $p_{\text{go}}$ | $p_{\text{nogo}}$ |
|------------|-----------------|-------------------|
| Optimal    | high            | low               |
| Impulsive  | high            | high              |
| Disengaged | low             | low               |
| Confused   | low             | high              |

Note: a GLM formulation (sigmoid of a linear combination) would be equivalent for binary stimuli — it is simply a reparameterization of the same two degrees of freedom, with no gain in expressiveness. Direct probabilities are preferred here because they admit a closed-form M-step.

---

## Emission Probability

The probability of observing lick $y_t$ on trial $t$ given hidden state $k$ is:

$$B_k(t) = P(y_t \mid u_t, z_t = k) = \begin{cases}
p_{\text{go}}^{(k)}     & \text{if } u_t = +1,\ y_t = 1 \\
1 - p_{\text{go}}^{(k)} & \text{if } u_t = +1,\ y_t = 0 \\
p_{\text{nogo}}^{(k)}     & \text{if } u_t = -1,\ y_t = 1 \\
1 - p_{\text{nogo}}^{(k)} & \text{if } u_t = -1,\ y_t = 0
\end{cases}$$

In the implementation, we work with $\log B_k(t)$ throughout for numerical stability.

---

## Multiple Sessions

The data consist of $S$ independent sessions (sequences). Sessions are treated as **independent sequences** sharing the same parameters — the hidden state is reset at the start of each session according to $\boldsymbol{\pi}$. This avoids making spurious transition inferences across the session boundary (overnight gap).

The total log-likelihood is the sum across sessions:

$$\mathcal{L} = \sum_{s=1}^{S} \log P(\mathbf{y}^{(s)} \mid \mathbf{u}^{(s)}, \boldsymbol{\theta})$$

---

## Fitting: the EM Algorithm

Parameters are estimated by maximizing $\mathcal{L}$ using the **Expectation-Maximization (EM)** algorithm, which alternates between two steps until convergence.

### E-step: Forward-Backward Algorithm

For each session $s$, the forward-backward algorithm computes two posterior quantities.

#### Forward variables

$$\alpha_t(k) = P(y_1, \ldots, y_t,\ z_t = k)$$

Recursion (in log-space):

$$\log \alpha_1(k) = \log \pi_k + \log B_k(1)$$

$$\log \alpha_t(k) = \log B_k(t) + \log \sum_{j=1}^{K} \alpha_{t-1}(j) \cdot A_{jk}$$

The $\log \sum$ is computed with `logsumexp` for numerical stability.

The session log-likelihood is:

$$\log P(\mathbf{y}^{(s)}) = \log \sum_{k=1}^{K} \alpha_T(k)$$

#### Backward variables

$$\beta_t(k) = P(y_{t+1}, \ldots, y_T \mid z_t = k)$$

Recursion (in log-space, run backwards):

$$\log \beta_T(k) = 0 \quad \forall k$$

$$\log \beta_t(k) = \log \sum_{k'=1}^{K} A_{kk'} \cdot B_{k'}(t+1) \cdot \beta_{t+1}(k')$$

#### Posterior state probabilities $\gamma$

$$\gamma_t(k) = P(z_t = k \mid \mathbf{y}) = \frac{\alpha_t(k)\, \beta_t(k)}{\sum_j \alpha_t(j)\, \beta_t(j)}$$

In log-space:

$$\log \gamma_t(k) = \log \alpha_t(k) + \log \beta_t(k) - \log P(\mathbf{y})$$

#### Posterior transition probabilities $\xi$

$$\xi_t(k, k') = P(z_t = k,\ z_{t+1} = k' \mid \mathbf{y})$$

$$\log \xi_t(k, k') = \log \alpha_t(k) + \log A_{kk'} + \log B_{k'}(t+1) + \log \beta_{t+1}(k') - \log P(\mathbf{y})$$

### M-step: Closed-Form Parameter Updates

The M-step updates parameters by maximizing the expected complete-data log-likelihood. All updates are closed-form, accumulating sufficient statistics across all sessions.

**Initial state distribution:**

$$\pi_k = \frac{1}{S} \sum_{s=1}^{S} \gamma_1^{(s)}(k)$$

**Transition matrix:**

$$A_{kk'} = \frac{\sum_s \sum_{t=1}^{T_s - 1} \xi_t^{(s)}(k, k')}{\sum_s \sum_{t=1}^{T_s - 1} \gamma_t^{(s)}(k)}$$

**Emission parameters:**

$$p_{\text{go}}^{(k)} = \frac{\sum_s \sum_{t:\, u_t^{(s)}=+1} \gamma_t^{(s)}(k)\cdot y_t^{(s)}}{\sum_s \sum_{t:\, u_t^{(s)}=+1} \gamma_t^{(s)}(k)}$$

$$p_{\text{nogo}}^{(k)} = \frac{\sum_s \sum_{t:\, u_t^{(s)}=-1} \gamma_t^{(s)}(k)\cdot y_t^{(s)}}{\sum_s \sum_{t:\, u_t^{(s)}=-1} \gamma_t^{(s)}(k)}$$

These are simply **weighted empirical lick rates** per state and stimulus type, with $\gamma_t(k)$ as the weight.

### Convergence

EM is run for up to `max_iter` iterations and stops early when the change in total log-likelihood falls below `tol = 1e-6`.

---

## Numerical Stability

All forward-backward computations are performed in **log-space**. The key operation is:

$$\log \sum_k \exp(x_k) = \text{logsumexp}(x_1, \ldots, x_K)$$

implemented via `scipy.special.logsumexp`, which subtracts the maximum before exponentiating to avoid overflow/underflow. Emission probabilities are clipped to $[\epsilon, 1-\epsilon]$ with $\epsilon = 10^{-10}$ to prevent $\log(0)$.

---

## Multiple Random Restarts

EM is sensitive to initialization and can converge to local maxima. To mitigate this, the algorithm is run `n_restarts` times from independent random initializations, and the solution with the highest final log-likelihood is retained.

Each initialization samples:
- $\boldsymbol{\pi}$ from a symmetric Dirichlet distribution
- Each row of $\mathbf{A}$ from a Dirichlet with concentration $> 1$ (encouraging self-transitions)
- $p_{\text{go}}^{(k)}$ and $p_{\text{nogo}}^{(k)}$ uniformly from $(0.1, 0.9)$

---

## State Decoding

After fitting, the **posterior state probabilities** $\gamma_t(k)$ are computed for every trial in every session via the forward-backward algorithm. These give a soft assignment of each trial to each hidden state.

A trial can be assigned to a specific state with confidence by thresholding:

$$\hat{z}_t = \arg\max_k\ \gamma_t(k) \quad \text{if } \max_k \gamma_t(k) \geq 0.8$$

Trials below the threshold are left as "indeterminate", following the convention of Hulsey et al. (2024).

---

## Summary of Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| $K$ | 3 | Number of hidden states (fixed for proof of concept) |
| `n_restarts` | 10 | Number of random EM initializations |
| `max_iter` | 200 | Maximum EM iterations per restart |
| `tol` | 1e-6 | Convergence threshold on log-likelihood |
| `MIN_TRIALS` | 50 | Minimum trials per session for inclusion |
| Confidence threshold | 0.8 | Minimum posterior probability for state assignment |

---

## References

- Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.
- Ashwood, Z.C. et al. (2022). Mice alternate between discrete strategies during perceptual decision-making. *Nature Neuroscience*, 25, 201–212.
- Hulsey, D. et al. (2024). Decision-making dynamics are predicted by arousal and uninstructed movements. *Cell Reports*, 43, 113709.
