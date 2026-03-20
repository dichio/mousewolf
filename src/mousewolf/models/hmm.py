"""
hmm.py
------
From-scratch implementation of a Hidden Markov Model with Bernoulli emissions
conditioned on a binary stimulus (Go/Nogo task).

Model
-----
- K hidden states
- Binary stimulus: stim ∈ {+1, -1}
- Binary observation: lick ∈ {0, 1}
- Emission per state k:
    p(lick=1 | stim=+1, state=k) = p_go[k]
    p(lick=1 | stim=-1, state=k) = p_nogo[k]
- Transition matrix A (K x K)
- Initial state distribution pi (K,)

All forward-backward computations are done in log-space for numerical stability.

This module is pure library code — no I/O, no paths, no side effects.
Import and use from any script or notebook.

Usage
-----
from mousewolf.models.hmm import fit_hmm, decode_states

params, log_lik, traces = fit_hmm(sequences, K=3, n_restarts=10)
posteriors = decode_states(sequences, params)
"""

import numpy as np
from scipy.special import logsumexp


# ── Emission ───────────────────────────────────────────────────────────────────

def compute_log_emissions(stim, lick, p_go, p_nogo):
    """
    Compute log emission probabilities for all trials and states.

    Parameters
    ----------
    stim   : np.ndarray (T,) — stimulus values (+1 or -1)
    lick   : np.ndarray (T,) — lick responses (0 or 1)
    p_go   : np.ndarray (K,) — P(lick=1 | stim=+1, state=k)
    p_nogo : np.ndarray (K,) — P(lick=1 | stim=-1, state=k)

    Returns
    -------
    log_B : np.ndarray (T, K) — log emission probabilities
    """
    T     = len(stim)
    K     = len(p_go)
    log_B = np.zeros((T, K))

    # Clip probabilities to avoid log(0)
    eps    = 1e-10
    p_go   = np.clip(p_go,   eps, 1 - eps)
    p_nogo = np.clip(p_nogo, eps, 1 - eps)

    for k in range(K):
        # Go trials (stim = +1)
        go_mask = stim == 1
        log_B[go_mask & (lick == 1), k] = np.log(p_go[k])
        log_B[go_mask & (lick == 0), k] = np.log(1 - p_go[k])

        # Nogo trials (stim = -1)
        nogo_mask = stim == -1
        log_B[nogo_mask & (lick == 1), k] = np.log(p_nogo[k])
        log_B[nogo_mask & (lick == 0), k] = np.log(1 - p_nogo[k])

    return log_B


# ── Forward pass ───────────────────────────────────────────────────────────────

def forward(log_B, log_A, log_pi):
    """
    Log-space forward algorithm for a single sequence.

    Parameters
    ----------
    log_B  : np.ndarray (T, K) — log emission probabilities
    log_A  : np.ndarray (K, K) — log transition matrix
    log_pi : np.ndarray (K,)   — log initial state distribution

    Returns
    -------
    log_alpha      : np.ndarray (T, K) — log forward variables
    log_likelihood : float             — log P(observations)
    """
    T, K      = log_B.shape
    log_alpha = np.zeros((T, K))

    # Initialisation
    log_alpha[0] = log_pi + log_B[0]

    # Recursion
    for t in range(1, T):
        for k in range(K):
            # log sum_j alpha[t-1, j] * A[j, k]
            log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_A[:, k]) + log_B[t, k]

    log_likelihood = logsumexp(log_alpha[-1])
    return log_alpha, log_likelihood


# ── Backward pass ──────────────────────────────────────────────────────────────

def backward(log_B, log_A):
    """
    Log-space backward algorithm for a single sequence.

    Parameters
    ----------
    log_B : np.ndarray (T, K) — log emission probabilities
    log_A : np.ndarray (K, K) — log transition matrix

    Returns
    -------
    log_beta : np.ndarray (T, K) — log backward variables
    """
    T, K     = log_B.shape
    log_beta = np.zeros((T, K))

    # Initialisation: beta[T-1] = 1 → log_beta[T-1] = 0 (already set)

    # Recursion (backwards)
    for t in range(T - 2, -1, -1):
        for k in range(K):
            # log sum_j A[k, j] * B[j, t+1] * beta[t+1, j]
            log_beta[t, k] = logsumexp(log_A[k] + log_B[t+1] + log_beta[t+1])

    return log_beta


# ── E-step: gamma and xi ───────────────────────────────────────────────────────

def compute_gamma_xi(log_alpha, log_beta, log_B, log_A, log_likelihood):
    """
    Compute posterior state probabilities (gamma) and
    posterior transition probabilities (xi) for a single sequence.

    Parameters
    ----------
    log_alpha      : np.ndarray (T, K)
    log_beta       : np.ndarray (T, K)
    log_B          : np.ndarray (T, K)
    log_A          : np.ndarray (K, K)
    log_likelihood : float

    Returns
    -------
    gamma : np.ndarray (T, K)      — P(z_t = k | observations)
    xi    : np.ndarray (T-1, K, K) — P(z_t=k, z_{t+1}=k' | observations)
    """
    T, K = log_alpha.shape

    # gamma: (T, K)
    log_gamma = log_alpha + log_beta - log_likelihood
    gamma     = np.exp(log_gamma)

    # xi: (T-1, K, K)
    # log xi[t, k, k'] = log_alpha[t,k] + log_A[k,k']
    #                   + log_B[t+1,k'] + log_beta[t+1,k'] - log_likelihood
    log_xi = (log_alpha[:-1, :, np.newaxis]   # (T-1, K, 1)
              + log_A[np.newaxis, :, :]        # (1,   K, K)
              + log_B[1:, np.newaxis, :]       # (T-1, 1, K)
              + log_beta[1:, np.newaxis, :]    # (T-1, 1, K)
              - log_likelihood)
    xi = np.exp(log_xi)

    return gamma, xi


# ── M-step ─────────────────────────────────────────────────────────────────────

def mstep(sequences, gammas, xis):
    """
    Closed-form M-step: update parameters from sufficient statistics
    accumulated across all sequences.

    Parameters
    ----------
    sequences : list of (stim, lick) tuples
    gammas    : list of np.ndarray (T_s, K)     — one per sequence
    xis       : list of np.ndarray (T_s-1, K, K) — one per sequence

    Returns
    -------
    pi     : np.ndarray (K,)   — updated initial state distribution
    A      : np.ndarray (K, K) — updated transition matrix (rows sum to 1)
    p_go   : np.ndarray (K,)   — updated P(lick=1 | stim=+1, state=k)
    p_nogo : np.ndarray (K,)   — updated P(lick=1 | stim=-1, state=k)
    """
    K   = gammas[0].shape[1]
    eps = 1e-10

    # Initial state distribution: average gamma at t=0 across sequences
    pi = np.mean([g[0] for g in gammas], axis=0)
    pi = np.clip(pi, eps, None)
    pi /= pi.sum()

    # Transition matrix: sum xi over time and sequences, normalize by rows
    xi_sum    = sum(xi.sum(axis=0) for xi in xis)        # (K, K)
    gamma_sum = sum(g[:-1].sum(axis=0) for g in gammas)  # (K,)
    A = xi_sum / (gamma_sum[:, np.newaxis] + eps)
    A = np.clip(A, eps, None)
    A /= A.sum(axis=1, keepdims=True)

    # Emission parameters: weighted counts per state and stimulus type
    p_go   = np.zeros(K)
    p_nogo = np.zeros(K)

    for (stim, lick), gamma in zip(sequences, gammas):
        go_mask   = stim == 1
        nogo_mask = stim == -1

        for k in range(K):
            # p_go[k]: weighted fraction of licks on Go trials
            w_go = gamma[go_mask, k]
            if w_go.sum() > eps:
                p_go[k] += (w_go * lick[go_mask]).sum()

            # p_nogo[k]: weighted fraction of licks on Nogo trials
            w_nogo = gamma[nogo_mask, k]
            if w_nogo.sum() > eps:
                p_nogo[k] += (w_nogo * lick[nogo_mask]).sum()

    # Normalise by total weighted trial counts per state
    total_go   = sum(gamma[stim == 1,  :].sum(axis=0) for (stim, _), gamma in zip(sequences, gammas))
    total_nogo = sum(gamma[stim == -1, :].sum(axis=0) for (stim, _), gamma in zip(sequences, gammas))

    p_go   = np.clip(p_go   / (total_go   + eps), eps, 1 - eps)
    p_nogo = np.clip(p_nogo / (total_nogo + eps), eps, 1 - eps)

    return pi, A, p_go, p_nogo


# ── EM algorithm ───────────────────────────────────────────────────────────────

def run_em(sequences, K, pi, A, p_go, p_nogo,
           max_iter=200, tol=1e-6, verbose=False):
    """
    Run the EM algorithm for a fixed initialisation.

    Parameters
    ----------
    sequences : list of (stim, lick) tuples — one per session
    K         : int   — number of hidden states
    pi        : np.ndarray (K,)   — initial state distribution
    A         : np.ndarray (K, K) — initial transition matrix
    p_go      : np.ndarray (K,)   — initial P(lick=1 | stim=+1, state=k)
    p_nogo    : np.ndarray (K,)   — initial P(lick=1 | stim=-1, state=k)
    max_iter  : int   — maximum number of EM iterations
    tol       : float — convergence threshold on log-likelihood
    verbose   : bool  — print log-likelihood at each iteration

    Returns
    -------
    params        : dict — fitted parameters (pi, A, p_go, p_nogo)
    log_lik_trace : list — log-likelihood at each iteration
    """
    log_lik_trace = []

    for iteration in range(max_iter):

        # ── E-step ────────────────────────────────────────────────────────────
        log_A  = np.log(A)
        log_pi = np.log(pi)

        gammas        = []
        xis           = []
        total_log_lik = 0.0

        for stim, lick in sequences:
            log_B              = compute_log_emissions(stim, lick, p_go, p_nogo)
            log_alpha, log_lik = forward(log_B, log_A, log_pi)
            log_beta           = backward(log_B, log_A)
            gamma, xi          = compute_gamma_xi(log_alpha, log_beta,
                                                  log_B, log_A, log_lik)
            gammas.append(gamma)
            xis.append(xi)
            total_log_lik += log_lik

        log_lik_trace.append(total_log_lik)

        if verbose:
            print(f"  Iter {iteration+1:3d} | log-likelihood = {total_log_lik:.4f}")

        # Check convergence
        if iteration > 0 and abs(log_lik_trace[-1] - log_lik_trace[-2]) < tol:
            if verbose:
                print(f"  Converged at iteration {iteration+1}.")
            break

        # ── M-step ────────────────────────────────────────────────────────────
        pi, A, p_go, p_nogo = mstep(sequences, gammas, xis)

    params = {
        "pi"    : pi,
        "A"     : A,
        "p_go"  : p_go,
        "p_nogo": p_nogo,
    }
    return params, log_lik_trace


# ── Random initialisation ──────────────────────────────────────────────────────

def random_init(K, rng=None):
    """
    Sample a random initialisation for the HMM parameters.

    Parameters
    ----------
    K   : int
    rng : np.random.Generator or None — if None, a fresh generator is created

    Returns
    -------
    pi, A, p_go, p_nogo
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initial state distribution: sample from Dirichlet
    pi = rng.dirichlet(np.ones(K))

    # Transition matrix: each row from Dirichlet
    # alpha > 1 encourages staying in the same state
    A = rng.dirichlet(np.ones(K) * 2, size=K)

    # Emission probabilities: uniform random in (0.1, 0.9)
    p_go   = rng.uniform(0.1, 0.9, size=K)
    p_nogo = rng.uniform(0.1, 0.9, size=K)

    return pi, A, p_go, p_nogo


# ── Multi-restart fitting ──────────────────────────────────────────────────────

def fit_hmm(sequences, K, n_restarts=10, max_iter=200,
            tol=1e-6, seed=0, verbose=False):
    """
    Fit the HMM using multiple random restarts, return the best solution.

    Parameters
    ----------
    sequences  : list of (stim, lick) tuples
    K          : int   — number of hidden states
    n_restarts : int   — number of random initialisations
    max_iter   : int   — max EM iterations per restart
    tol        : float — convergence tolerance
    seed       : int   — base random seed (each restart uses seed + restart)
    verbose    : bool  — print progress

    Returns
    -------
    best_params        : dict  — parameters from the best restart
    best_log_lik       : float — final log-likelihood of best solution
    all_log_lik_traces : list  — log-likelihood traces for all restarts
    """
    best_params  = None
    best_log_lik = -np.inf
    all_traces   = []

    for restart in range(n_restarts):
        rng                  = np.random.default_rng(seed + restart)
        pi, A, p_go, p_nogo  = random_init(K, rng)

        if verbose:
            print(f"\nRestart {restart+1}/{n_restarts}")

        params, trace = run_em(sequences, K, pi, A, p_go, p_nogo,
                               max_iter=max_iter, tol=tol, verbose=verbose)
        all_traces.append(trace)

        final_log_lik = trace[-1]
        if final_log_lik > best_log_lik:
            best_log_lik = final_log_lik
            best_params  = params

    print(f"\nBest log-likelihood across {n_restarts} restarts: {best_log_lik:.4f}")
    return best_params, best_log_lik, all_traces


# ── State decoding ─────────────────────────────────────────────────────────────

def decode_states(sequences, params):
    """
    Compute posterior state probabilities (soft decoding) for all sequences.

    Parameters
    ----------
    sequences : list of (stim, lick) tuples
    params    : dict — fitted HMM parameters (pi, A, p_go, p_nogo)

    Returns
    -------
    posteriors : list of np.ndarray (T_s, K) — one per sequence
    """
    log_A  = np.log(params["A"])
    log_pi = np.log(params["pi"])

    posteriors = []
    for stim, lick in sequences:
        log_B              = compute_log_emissions(stim, lick,
                                                   params["p_go"], params["p_nogo"])
        log_alpha, log_lik = forward(log_B, log_A, log_pi)
        log_beta           = backward(log_B, log_A)
        gamma, _           = compute_gamma_xi(log_alpha, log_beta,
                                              log_B, log_A, log_lik)
        posteriors.append(gamma)

    return posteriors
