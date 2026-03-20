"""
analyse_hmm_results.py
----------------------
Loads hmm_results.pkl and produces a structured analysis of the fitted models
across all mice, weeks, and K values.

Analyses
--------
1. Model comparison: log-likelihood vs K per mouse/week
2. Discrimination index d = p_go - p_nogo per state
3. State characterization summary table
4. Transition matrix stickiness (diagonal of A)
5. Summary heatmap: discrimination across mouse x week x K
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import os

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_PATH = "data/hmm_results.pkl"
OUTPUT_DIR   = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_COLORS = ["#5b8db8", "#6abf69", "#e07b54", "#9b59b6"]

# ── Load ───────────────────────────────────────────────────────────────────────

with open(RESULTS_PATH, "rb") as f:
    results = pickle.load(f)

mice  = sorted(results.keys())
weeks = sorted({w for m in results for w in results[m]})
Ks    = sorted({K for m in results for w in results[m] for K in results[m][w]})


# ── Helper: label a state from its emission parameters ────────────────────────

def label_state(p_go, p_nogo, threshold_disc=0.35, threshold_low=0.25):
    """
    Assign a behavioral label to a state based on its emission probabilities.

    Parameters
    ----------
    p_go, p_nogo       : float
    threshold_disc     : float — minimum d = p_go - p_nogo to call 'Optimal'
    threshold_low      : float — maximum p to call 'Disengaged'

    Returns
    -------
    label : str
    """
    d = p_go - p_nogo
    if d >= threshold_disc:
        return "Optimal"
    elif p_go < threshold_low and p_nogo < threshold_low:
        return "Disengaged"
    elif p_go > (1 - threshold_low) and p_nogo > (1 - threshold_low):
        return "Impulsive"
    else:
        return "Intermediate"


# ── 1. Build flat records for all states ──────────────────────────────────────

records = []
for mouse in mice:
    for week in sorted(results[mouse]):
        for K in sorted(results[mouse][week]):
            r = results[mouse][week][K]
            p = r["params"]
            for k in range(K):
                p_go   = p["p_go"][k]
                p_nogo = p["p_nogo"][k]
                d      = p_go - p_nogo
                records.append({
                    "mouse"    : mouse,
                    "week"     : week,
                    "K"        : K,
                    "state"    : k,
                    "p_go"     : p_go,
                    "p_nogo"   : p_nogo,
                    "d"        : d,
                    "stickiness": p["A"][k, k],
                    "pi"       : p["pi"][k],
                    "log_lik"  : r["log_lik"],
                    "n_trials" : r["n_trials"],
                    "label"    : label_state(p_go, p_nogo),
                })

df = pd.DataFrame(records)

# Print full state summary table
print("=" * 75)
print("  FULL STATE SUMMARY")
print("=" * 75)
print(df[["mouse", "week", "K", "state", "p_go", "p_nogo", "d",
          "stickiness", "label"]].to_string(index=False, float_format="{:.3f}".format))


# ── 2. Log-likelihood vs K ─────────────────────────────────────────────────────

print("\n" + "=" * 75)
print("  LOG-LIKELIHOOD VS K")
print("=" * 75)
ll_df = (df[["mouse", "week", "K", "log_lik", "n_trials"]]
         .drop_duplicates()
         .sort_values(["mouse", "week", "K"]))

# Normalise by n_trials for comparability across weeks
ll_df["ll_per_trial"] = ll_df["log_lik"] / ll_df["n_trials"]
print(ll_df.to_string(index=False, float_format="{:.4f}".format))

fig, axes = plt.subplots(len(mice), len(weeks),
                         figsize=(4 * len(weeks), 3 * len(mice)),
                         sharey=False)
axes = np.array(axes).reshape(len(mice), len(weeks))

for i, mouse in enumerate(mice):
    for j, week in enumerate(weeks):
        ax = axes[i, j]
        if week not in results[mouse]:
            ax.set_visible(False)
            continue
        sub = ll_df[(ll_df["mouse"] == mouse) & (ll_df["week"] == week)]
        ax.plot(sub["K"], sub["ll_per_trial"], "o-", color="steelblue", lw=2)
        ax.set_title(f"{mouse} — week {week}", fontsize=9)
        ax.set_xlabel("K", fontsize=8)
        ax.set_ylabel("log-lik / trial", fontsize=8)
        ax.set_xticks(Ks)
        ax.grid(True, alpha=0.3)

fig.suptitle("Log-likelihood per trial vs K", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loglik_vs_K.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: figures/loglik_vs_K.png")


# ── 3. Discrimination heatmap: best d per (mouse, week, K) ────────────────────

# For each (mouse, week, K), find the maximum discrimination across states
best_d = (df.groupby(["mouse", "week", "K"])["d"]
            .max()
            .reset_index()
            .rename(columns={"d": "best_d"}))

print("\n" + "=" * 75)
print("  BEST DISCRIMINATION d = p_go - p_nogo PER (MOUSE, WEEK, K)")
print("=" * 75)
print(best_d.to_string(index=False, float_format="{:.3f}".format))

# Plot heatmap for each K
fig, axes = plt.subplots(1, len(Ks), figsize=(5 * len(Ks), 3.5))
for ax, K in zip(axes, Ks):
    sub = best_d[best_d["K"] == K].pivot(index="mouse", columns="week", values="best_d")
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=1.0)
    im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels([f"Week {w}" for w in sub.columns], fontsize=9)
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index, fontsize=9)
    ax.set_title(f"K={K}", fontsize=10)
    # Annotate cells
    for ii in range(sub.shape[0]):
        for jj in range(sub.shape[1]):
            val = sub.values[ii, jj]
            if not np.isnan(val):
                ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label="best d")

fig.suptitle("Best discrimination d = p_go − p_nogo\n(green = optimal-like state present)",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "discrimination_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figures/discrimination_heatmap.png")


# ── 4. State labels count per (mouse, week, K) ────────────────────────────────

print("\n" + "=" * 75)
print("  STATE LABEL COUNTS")
print("=" * 75)
label_counts = (df.groupby(["mouse", "week", "K", "label"])
                  .size()
                  .reset_index(name="count"))
print(label_counts.to_string(index=False))


# ── 5. Stickiness overview ────────────────────────────────────────────────────

print("\n" + "=" * 75)
print("  MEAN STATE STICKINESS (diagonal of A) PER (MOUSE, WEEK, K)")
print("=" * 75)
stickiness = (df.groupby(["mouse", "week", "K"])["stickiness"]
                .mean()
                .reset_index()
                .rename(columns={"stickiness": "mean_stickiness"}))
print(stickiness.to_string(index=False, float_format="{:.3f}".format))


# ── 6. Focus: optimal-like states ─────────────────────────────────────────────

print("\n" + "=" * 75)
print("  OPTIMAL-LIKE STATES (d >= 0.35)")
print("=" * 75)
optimal = df[df["label"] == "Optimal"].sort_values(["mouse", "week", "K", "d"],
                                                     ascending=[True, True, True, False])
if len(optimal) == 0:
    print("  No optimal-like states found across any mouse/week/K.")
else:
    print(optimal[["mouse", "week", "K", "state", "p_go", "p_nogo",
                    "d", "stickiness", "pi"]].to_string(index=False,
                                                         float_format="{:.3f}".format))