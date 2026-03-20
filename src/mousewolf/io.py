"""
io.py
-----
Input/output utilities for the mousewolf project.
Handles saving and loading of pickle files, and creation of subdirectories.

Author : Vito Dichio
Date   : Sept 1, 2025
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path


# ── Pickle I/O ─────────────────────────────────────────────────────────────────

def save_pickle(datapath: str | Path, **nested_dictionaries) -> None:
    """
    Save one or more objects as individual pickle files.

    Each keyword argument is saved as a separate ``<name>.pkl`` file
    inside ``datapath``. The destination directory is created automatically
    if it does not already exist.

    Parameters
    ----------
    datapath : str or Path
        Directory where the pickle files will be written.
    **nested_dictionaries : any
        Named objects to serialise. The keyword name becomes the filename,
        e.g. ``save_pickle(path, results=my_dict)`` writes ``results.pkl``.

    Examples
    --------
    >>> save_pickle(DATA_DIR, results=my_results, metadata=my_metadata)
    # Writes DATA_DIR/results.pkl and DATA_DIR/metadata.pkl
    """
    datapath = Path(datapath)
    datapath.mkdir(parents=True, exist_ok=True)

    for name, obj in nested_dictionaries.items():
        file_path = datapath / f"{name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)


def load_pickle(datapath: str | Path, **kwargs: bool) -> any:
    """
    Load one or more pickle files from a directory.

    Only files whose corresponding keyword argument is ``True`` are loaded.
    Missing files emit a warning and are skipped rather than raising an error.

    Parameters
    ----------
    datapath : str or Path
        Directory containing the pickle files.
    **kwargs : bool
        Names of files to load mapped to a boolean flag.
        e.g. ``load_pickle(DATA_DIR, results=True, metadata=False)``
        will load only ``results.pkl``.

    Returns
    -------
    any
        - If exactly **one** file is requested: returns its contents directly.
        - If **multiple** files are requested: returns a tuple in the same
          order as the keyword arguments.
        - Missing files are returned as ``None`` in the tuple.

    Examples
    --------
    >>> results = load_pickle(DATA_DIR, results=True)

    >>> results, metadata = load_pickle(DATA_DIR, results=True, metadata=True)
    """
    datapath  = Path(datapath)
    loaded    = {}

    for name, should_load in kwargs.items():
        if not should_load:
            continue

        file_path = datapath / f"{name}.pkl"

        if file_path.exists():
            with open(file_path, "rb") as f:
                loaded[name] = pickle.load(f)
        else:
            warnings.warn(f"File not found, returning None: {file_path}")
            loaded[name] = None

    # Return scalar directly when only one object was requested
    requested = [name for name, flag in kwargs.items() if flag]

    if len(requested) == 1:
        return loaded.get(requested[0])

    return tuple(loaded.get(name) for name in requested)


# ── Directory utilities ────────────────────────────────────────────────────────

def mk_subdir(path: str | Path, *subfolders: str | None) -> Path:
    """
    Create a nested subdirectory tree under ``path`` and return it.

    Builds the target path by appending each non-``None`` subfolder in order,
    then ensures the full directory tree exists. The call is idempotent —
    it is safe to call even if the directory already exists.

    Parameters
    ----------
    path : str or Path
        Base directory to build upon.
    *subfolders : str or None
        Sequence of folder names to append to ``path``. ``None`` values
        are silently skipped, which allows conditional path segments like
        ``mk_subdir(RESULTS_DIR, mouse, week if week else None)``.

    Returns
    -------
    Path
        Fully resolved path to the created directory.

    Examples
    --------
    >>> mk_subdir(RESULTS_DIR, "mouse_01", "week_3")
    PosixPath('/home/vdichio/projects/mousewolf/results/mouse_01/week_3')

    >>> mk_subdir(RESULTS_DIR, "mouse_01", None, "week_3")
    PosixPath('/home/vdichio/projects/mousewolf/results/mouse_01/week_3')
    """
    path = Path(path)

    for folder in subfolders:
        if folder is not None:
            path = path / str(folder)

    path.mkdir(parents=True, exist_ok=True)
    return path