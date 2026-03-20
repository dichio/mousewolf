"""
fit_hmm.py
----------
Fits the HMM model across all mice, weeks and K values in parallel.
Reads dataset from DATA_DIR and saves results to DATA_DIR.

This is a project-specific script — all I/O, paths and parallelization
settings live here, keeping the core HMM library in mousewolf.models.hmm
clean and reusable.

Usage
-----
    python scripts/fit_hmm.py
"""

from joblib import Parallel, delayed

from mousewolf.models.hmm import fit_hmm
from mousewolf.paths import DATA_DIR, RESULTS_DIR
from mousewolf.io import save_pickle, load_pickle


# ── Configuration ──────────────────────────────────────────────────────────────

K_VALUES   = [2, 3, 4]   # number of hidden states to try
MIN_TRIALS = 50           # minimum trials per session to include
N_JOBS     = 9            # number of parallel CPUs (be considerate on shared servers!)
N_RESTARTS = 10           # number of random restarts per fit


# ── Parallel worker ────────────────────────────────────────────────────────────

def fit_one(mouse, week, K, sequences, n_trials):
    """
    Fit HMM for a single (mouse, week, K) combination.
    Designed to be called in parallel via joblib.

    Parameters
    ----------
    mouse     : str  — mouse identifier (e.g. "M06")
    week      : int  — week number
    K         : int  — number of hidden states
    sequences : list — list of (stim, lick) tuples, one per session
    n_trials  : int  — total number of trials across sessions

    Returns
    -------
    tuple : (mouse, week, K, result_dict)
    """
    print(f"Fitting {mouse} | week {week} | K={K} | "
          f"{len(sequences)} sessions, {n_trials} trials")

    best_params, best_log_lik, all_traces = fit_hmm(
        sequences, K=K, n_restarts=N_RESTARTS, verbose=False
    )

    print(f"  Done  {mouse} | week {week} | K={K} | "
          f"log-lik: {best_log_lik:.4f}")

    return (mouse, week, K, {
        "params"     : best_params,
        "log_lik"    : best_log_lik,
        "traces"     : all_traces,
        "n_sessions" : len(sequences),
        "n_trials"   : n_trials,
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading dataset from {DATA_DIR} ...")
    data = load_pickle(DATA_DIR, dataset=True)

    # ── Build all (mouse, week, K) jobs ───────────────────────────────────────
    jobs = []

    for mouse, df in data.items():
        df_filtered = df[df["state"] != 5]  # all weeks
        sequences = [
            (s["stim"].values, s["lick"].values)
            for _, s in df_filtered.groupby("session")
            if len(s) >= MIN_TRIALS
        ]

        if not sequences:
            print(f"  Skipping {mouse} — no valid sessions")
            continue

        n_trials = sum(len(s[0]) for s in sequences)

        for K in K_VALUES:
            jobs.append((mouse, None, K, sequences, n_trials))

    print(f"\nTotal jobs to run: {len(jobs)} "
          f"({len(data)} mice × weeks × {len(K_VALUES)} K values)\n")

    # ── Run in parallel ────────────────────────────────────────────────────────
    outputs = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(fit_one)(mouse, week, K, sequences, n_trials)
        for mouse, week, K, sequences, n_trials in jobs
    )

    # Collect results: results[mouse][K] instead of results[mouse][week][K]
    results = {}
    for mouse, _, K, result in outputs:
        results.setdefault(mouse, {})[K] = result

    save_pickle(RESULTS_DIR, hmm_results_per_mouse=results)
    print(f"\nSaved: {RESULTS_DIR / 'hmm_results_per_mouse.pkl'}")

