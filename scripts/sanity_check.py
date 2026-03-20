"""
plot_all_posteriors.py
----------------------
Loads hmm_results.pkl and dataset.pkl, decodes posterior state probabilities
for every (mouse, week, K) combination, and saves one plot per combination.

Output files are saved as:
    figures/posteriors_<MOUSE>_week<W>_K<K>.png
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scripts.hmm import decode_states   # adjust import path if needed

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_PATH = "data/hmm_results.pkl"
DATASET_PATH = "data/dataset.pkl"
OUTPUT_DIR   = "figures"
MIN_TRIALS   = 50

# Colors cycle — extended to support up to K=4
STATE_COLORS = ["#5b8db8", "#6abf69", "#e07b54", "#9b59b6"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

with open(RESULTS_PATH, "rb") as f:
    results = pickle.load(f)

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

# ── Plotting function ──────────────────────────────────────────────────────────

def plot_posteriors(mouse, week, K, params, sequences, save_path):
    """
    Decode and plot posterior state probabilities for one (mouse, week, K).

    Parameters
    ----------
    mouse      : str
    week       : int
    K          : int
    params     : dict — fitted HMM parameters
    sequences  : list of (stim, lick) tuples
    save_path  : str — output file path
    """
    # Decode posteriors
    posteriors = decode_states(sequences, params)

    # Concatenate across sessions
    all_gamma = np.concatenate(posteriors, axis=0)
    all_stim  = np.concatenate([s[0] for s in sequences])
    all_lick  = np.concatenate([s[1] for s in sequences])

    # Session boundary indices
    session_lengths    = [len(s[0]) for s in sequences]
    session_boundaries = np.cumsum(session_lengths)[:-1]
    total_trials       = len(all_gamma)
    trial_idx          = np.arange(total_trials)

    # Build state labels from emission parameters
    p_go   = params["p_go"]
    p_nogo = params["p_nogo"]
    state_labels = []
    for k in range(K):
        d = p_go[k] - p_nogo[k]
        if d > 0.4:
            label = f"State {k} (Optimal-like)"
        elif p_go[k] > 0.6 and p_nogo[k] > 0.6:
            label = f"State {k} (Impulsive)"
        elif p_go[k] < 0.3 and p_nogo[k] < 0.3:
            label = f"State {k} (Disengaged)"
        else:
            label = f"State {k} (p_go={p_go[k]:.2f}, p_nogo={p_nogo[k]:.2f})"
        state_labels.append(label)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.08,
                   height_ratios=[4, 0.6, 0.6])

    # Panel 1: posteriors
    ax_post = fig.add_subplot(gs[0])
    for k in range(K):
        color = STATE_COLORS[k % len(STATE_COLORS)]
        ax_post.plot(trial_idx, all_gamma[:, k],
                     color=color, lw=0.8, alpha=0.85, label=state_labels[k])
        ax_post.fill_between(trial_idx, all_gamma[:, k],
                             color=color, alpha=0.15)

    for b in session_boundaries:
        ax_post.axvline(b, color="black", lw=0.8, ls="--", alpha=0.4)

    ax_post.set_ylabel("Posterior probability $\\gamma_t(k)$", fontsize=11)
    ax_post.set_ylim(-0.02, 1.05)
    ax_post.set_xlim(0, total_trials)
    ax_post.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax_post.set_title(
        f"{mouse} — Week {week} — K={K} HMM posterior state probabilities\n"
        f"log-lik = {results[mouse][week][K]['log_lik']:.2f} | "
        f"{results[mouse][week][K]['n_sessions']} sessions | "
        f"{results[mouse][week][K]['n_trials']} trials",
        fontsize=11
    )
    ax_post.set_xticklabels([])

    # Panel 2: stimulus
    ax_stim = fig.add_subplot(gs[1], sharex=ax_post)
    go_trials   = np.where(all_stim == 1)[0]
    nogo_trials = np.where(all_stim == -1)[0]
    ax_stim.scatter(go_trials,   np.ones(len(go_trials)),   s=2,
                    color="#2c7bb6", alpha=0.6, label="Go (+1)")
    ax_stim.scatter(nogo_trials, np.zeros(len(nogo_trials)), s=2,
                    color="#d7191c", alpha=0.6, label="Nogo (−1)")
    for b in session_boundaries:
        ax_stim.axvline(b, color="black", lw=0.8, ls="--", alpha=0.4)
    ax_stim.set_ylabel("Stim", fontsize=9)
    ax_stim.set_yticks([0, 1])
    ax_stim.set_yticklabels(["Nogo", "Go"], fontsize=8)
    ax_stim.set_ylim(-0.5, 1.5)
    ax_stim.legend(loc="upper right", fontsize=8, markerscale=4, framealpha=0.8)
    ax_stim.set_xticklabels([])

    # Panel 3: lick
    ax_lick = fig.add_subplot(gs[2], sharex=ax_post)
    lick_trials   = np.where(all_lick == 1)[0]
    nolick_trials = np.where(all_lick == 0)[0]
    ax_lick.scatter(lick_trials,   np.ones(len(lick_trials)),   s=2,
                    color="#1a9641", alpha=0.6, label="Lick")
    ax_lick.scatter(nolick_trials, np.zeros(len(nolick_trials)), s=2,
                    color="#aaaaaa", alpha=0.4, label="No lick")
    for b in session_boundaries:
        ax_lick.axvline(b, color="black", lw=0.8, ls="--", alpha=0.4)
    ax_lick.set_ylabel("Lick", fontsize=9)
    ax_lick.set_yticks([0, 1])
    ax_lick.set_yticklabels(["No", "Yes"], fontsize=8)
    ax_lick.set_ylim(-0.5, 1.5)
    ax_lick.set_xlabel("Trial index", fontsize=11)
    ax_lick.legend(loc="upper right", fontsize=8, markerscale=4, framealpha=0.8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main loop ──────────────────────────────────────────────────────────────────

for mouse, week_dict in results.items():
    df = data[mouse]

    for week, k_dict in week_dict.items():
        # Rebuild sequences for this mouse/week
        df_filtered = df[(df["week"] == week) & (df["state"] != 5)]
        sequences = [
            (session_df["stim"].values, session_df["lick"].values)
            for _, session_df in df_filtered.groupby("session")
            if len(session_df) >= MIN_TRIALS
        ]

        if len(sequences) == 0:
            continue

        for K, result in k_dict.items():
            save_path = os.path.join(
                OUTPUT_DIR, f"posteriors_{mouse}_week{week}_K{K}.png"
            )
            print(f"Plotting {mouse} | week {week} | K={K} ...")
            plot_posteriors(mouse, week, K, result["params"], sequences, save_path)

print(f"\nDone. All figures saved to '{OUTPUT_DIR}/'")