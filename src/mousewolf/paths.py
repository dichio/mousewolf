"""
paths.py
--------
Centralised path definitions for the mousewolf project.
Auto-detects whether the code is running locally or on the remote server
(morricone) and sets paths accordingly.

Usage
-----
from mousewolf.paths import ROOT_DIR, DATA_DIR, RESULTS_DIR
"""

import os
from pathlib import Path

# ── Environment detection ──────────────────────────────────────────────────────

# Detect remote server by checking the current username
REMOTE_USER = "vdichio"
IS_REMOTE   = os.getenv("USER") == REMOTE_USER

# ── Root directory ─────────────────────────────────────────────────────────────

if IS_REMOTE:
    ROOT_DIR = Path("/home/vdichio/projects/mousewolf")
else:
    # Automatically resolve root as 2 levels up from this file
    # src/mousewolf/paths.py → go up twice → project root
    ROOT_DIR = Path(__file__).resolve().parents[2]

# ── Project subdirectories ─────────────────────────────────────────────────────

DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
SCRIPTS_DIR = ROOT_DIR / "scripts"
SRC_DIR     = ROOT_DIR / "src"

# ── Ensure all directories exist ───────────────────────────────────────────────

for _dir in [DATA_DIR, RESULTS_DIR, SCRIPTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running {'remotely' if IS_REMOTE else 'locally'}")
    print(f"ROOT_DIR    : {ROOT_DIR}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print(f"RESULTS_DIR : {RESULTS_DIR}")
    print(f"SCRIPTS_DIR : {SCRIPTS_DIR}")