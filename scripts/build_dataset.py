"""
build_dataset.py
----------------
Loads all binarized behavior HDF5 files from the project data folder and
builds one tidy DataFrame per mouse, stored in a dictionary keyed by mouse
code. The resulting dataset is saved as a pickle in the same data folder.

Expected file naming convention:
    <MOUSE_CODE>_binarized_behavior.h5     e.g.  M06_binarized_behavior.h5

DataFrame columns (one row = one trial):
    trial       : int   — global trial index (0-based, within mouse)
    session     : int   — session number (1-based)
    week        : int   — week number (1-based, increments after each weekend)
    stim        : int   — stimulus identity  (+1 = Go, -1 = Nogo)
    lick        : int   — lick response      (1 = licked, 0 = no lick)
    ans_win     : int   — answer window flag (1 = valid window, 0 = aborted)
    reward_raw  : int   — reward as stored in H5 (for auditing)
    reward      : int   — reward derived from state  (+1 Hit, 0 Miss/CR, -1 FA)
    state       : int   — behavioral state
                            1 = Hit           (stim=+1, lick=1, ans_win=1)
                            2 = Miss          (stim=+1, lick=0, ans_win=1)
                            3 = False Alarm   (stim=-1, lick=1, ans_win=1)
                            4 = Correct Rej   (stim=-1, lick=0, ans_win=1)
                            5 = No window     (ans_win=0)

Usage:
    from mousewolf.build_dataset import load_all_mice
    data = load_all_mice()        # returns dict[mouse_code -> DataFrame]
    df_m09 = data["M09"]
"""

import re
import numpy as np
import pandas as pd
import h5py

from mousewolf.paths import DATA_DIR      # ← central path definition
from mousewolf.io import save_pickle      # ← package I/O utilities

STATE_LABELS = {
    1: "Hit",
    2: "Miss",
    3: "False Alarm",
    4: "Correct Reject",
    5: "No window",
}


# ── State and reward encoding ──────────────────────────────────────────────────

def compute_state(stim, lick, ans_win):
    """
    Derive per-trial state from raw behavioral variables.

    Parameters
    ----------
    stim    : np.ndarray  (+1 or -1)
    lick    : np.ndarray  (0 or 1)
    ans_win : np.ndarray  (0 or 1)

    Returns
    -------
    state : np.ndarray of int, values in {1, 2, 3, 4, 5}
    """
    state  = np.full(len(stim), 5, dtype=int)          # default: no window
    active = ans_win == 1
    state[active & (stim ==  1) & (lick == 1)] = 1    # Hit
    state[active & (stim ==  1) & (lick == 0)] = 2    # Miss
    state[active & (stim == -1) & (lick == 1)] = 3    # False Alarm
    state[active & (stim == -1) & (lick == 0)] = 4    # Correct Reject
    return state


def compute_reward_from_state(state):
    """
    Derive reward from state.
        +1  for Hit              (state 1) — water reward
         0  for Miss or CR       (states 2, 4) — no feedback
        -1  for False Alarm      (state 3) — alarm sound
        -1  for No window        (state 5) — lick outside answer window

    Parameters
    ----------
    state : np.ndarray of int

    Returns
    -------
    reward : np.ndarray of int
    """
    reward            = np.zeros(len(state), dtype=int)
    reward[state == 1] =  1
    reward[state == 3] = -1
    reward[state == 5] = -1
    return reward


# ── Session and week assignment ────────────────────────────────────────────────

def assign_sessions(n_trials, session_ends):
    """
    Assign a 1-based session number to each trial.

    Parameters
    ----------
    n_trials     : int
    session_ends : np.ndarray — trial indices marking the END of each session

    Returns
    -------
    session_ids : np.ndarray of int (1-based)
    """
    session_ids = np.zeros(n_trials, dtype=int)
    boundaries  = np.concatenate([[0], session_ends])

    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        end = min(end, n_trials)
        session_ids[start:end] = i

    last = min(session_ends[-1], n_trials)
    if last < n_trials:
        session_ids[last:] = len(session_ends)

    return session_ids


def assign_weeks(n_trials, weekend_breaks):
    """
    Assign a 1-based week number to each trial.

    Parameters
    ----------
    n_trials       : int
    weekend_breaks : np.ndarray — trial indices where a weekend occurred

    Returns
    -------
    week_ids : np.ndarray of int (1-based)
    """
    week_ids = np.zeros(n_trials, dtype=int)
    for week_num, break_idx in enumerate(weekend_breaks, start=1):
        week_ids[break_idx:] = week_num
    return week_ids


# ── Per-mouse loader ───────────────────────────────────────────────────────────

def load_mouse(filepath):
    """
    Load one HDF5 file and return a tidy trial-level DataFrame.

    Parameters
    ----------
    filepath : Path — path to <MOUSE>_binarized_behavior.h5

    Returns
    -------
    df : pd.DataFrame
    """
    with h5py.File(filepath, "r") as f:
        stim           = f["behavior/stim"][:]
        lick           = f["behavior/licks"][:]
        reward_raw     = f["behavior/reward"][:]
        ans_win        = f["behavior/answer_window"][:]
        session_ends   = f["sessions"][:]
        weekend_breaks = f["weekends"][:]

    n_trials    = len(stim)
    state       = compute_state(stim, lick, ans_win)
    reward      = compute_reward_from_state(state)
    session_ids = assign_sessions(n_trials, session_ends)
    week_ids    = assign_weeks(n_trials, weekend_breaks)

    return pd.DataFrame({
        "trial"      : np.arange(n_trials),
        "session"    : session_ids,
        "week"       : week_ids,
        "stim"       : stim,
        "lick"       : lick,
        "ans_win"    : ans_win,
        "reward_raw" : reward_raw,
        "reward"     : reward,
        "state"      : state,
    })


# ── Consistency checks ─────────────────────────────────────────────────────────

def check_consistency(mouse_code, df):
    """
    Run consistency checks on a loaded DataFrame and print a report.

    Parameters
    ----------
    mouse_code : str
    df         : pd.DataFrame
    """
    print(f"\n{'='*55}")
    print(f"  Consistency report — {mouse_code}")
    print(f"{'='*55}")
    print(f"  Total trials : {len(df)}")
    print(f"  Sessions     : {df['session'].nunique()}")
    print(f"  Weeks        : {df['week'].nunique()}")

    print(f"\n  --- State distribution ---")
    for s, label in STATE_LABELS.items():
        n   = (df["state"] == s).sum()
        pct = 100 * n / len(df)
        print(f"    State {s} | {label:15s} | n={n:5d} ({pct:.1f}%)")


# ── Main loader ────────────────────────────────────────────────────────────────

def load_all_mice():
    """
    Scan DATA_DIR for files matching *_binarized_behavior.h5, load each one,
    run consistency checks, and return a dictionary of DataFrames.

    Returns
    -------
    mice : dict[str -> pd.DataFrame]
        Keys are mouse codes (e.g. "M06"), values are trial-level DataFrames.
    """
    pattern = re.compile(r"^(.+)_binarized_behavior\.h5$")
    mice    = {}

    h5_files = sorted([
        f for f in DATA_DIR.iterdir()
        if pattern.match(f.name)
    ])

    if not h5_files:
        print(f"[WARNING] No matching H5 files found in '{DATA_DIR}'.")
        return mice

    print(f"Found {len(h5_files)} file(s) in '{DATA_DIR}':")
    for f in h5_files:
        print(f"  {f.name}")

    for fpath in h5_files:
        mouse_code        = pattern.match(fpath.name).group(1)
        print(f"\nLoading {mouse_code} ...")
        df                = load_mouse(fpath)
        mice[mouse_code]  = df
        check_consistency(mouse_code, df)

    print(f"\n{'='*55}")
    print(f"  Done. Loaded {len(mice)} mouse/mice.")
    print(f"  Dictionary keys: {list(mice.keys())}")
    print(f"{'='*55}\n")

    return mice


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    data = load_all_mice()

    # Quick preview of the first mouse
    if data:
        first_key = list(data.keys())[0]
        print(f"Preview of {first_key}:")
        print(data[first_key].head(10).to_string(index=False))

    # Save dataset to DATA_DIR/dataset.pkl
    save_pickle(DATA_DIR, dataset=data)
    print(f"Saved: {DATA_DIR / 'dataset.pkl'}")

    # ── Per-animal summary ─────────────────────────────────────────────────────

    for mouse, df in data.items():

        print(f"\n{'='*55}")
        print(f"  {mouse}  —  {df['week'].nunique()} week(s), {len(df)} total trials")
        print(f"{'='*55}")

        week_counts   = df.groupby("week").size().rename("total_trials")
        week_sessions = df.groupby("week")["session"].nunique().rename("n_sessions")
        print(f"\n  Trials per week (all states):")
        print(pd.concat([week_sessions, week_counts], axis=1).to_string())

        df_active    = df[df["state"] != 5]
        week_active  = df_active.groupby("week").size().rename("hmm_trials (state 1-4)")
        print(f"\n  HMM-eligible trials per week (state 1-4 only):")
        print(week_active.to_string())

        last_week       = df["week"].max()
        df_last         = df[df["week"] == last_week]
        df_last_active  = df_last[df_last["state"] != 5]

        print(f"\n  Last week (week {last_week}) — state breakdown:")
        for s, label in STATE_LABELS.items():
            n   = (df_last["state"] == s).sum()
            pct = 100 * n / len(df_last)
            print(f"    State {s} | {label:15s} | n={n:4d} ({pct:.1f}%)")

        print(f"\n  Last week — HMM-eligible: {len(df_last_active)} trials "
              f"across {df_last['session'].nunique()} sessions")