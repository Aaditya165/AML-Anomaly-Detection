"""
utils.py
--------
ExSTraQt's expensive stages are graph/feature generation, not model
training -- Leiden community detection and multi-hop flow tracing are the
steps worth caching aggressively so a rerun (e.g. after tweaking the
model or the merge step) doesn't repeat them.

Every expensive function in this project follows the same shape:

    if cache_exists:
        load_cache()
    else:
        compute()
        save_cache()

`cache_pickle` / `cache_parquet` are that pattern as a one-liner, keyed by
an explicit path (not an auto-hash of arguments -- the caller decides what
should invalidate a cache, e.g. by putting the input filename or a config
version in the path, same as the existing project's `CACHE_DIR / "..."`
pattern in the notebook).
"""

import pickle
import time
from pathlib import Path
from typing import Callable

import pandas as pd


def cache_pickle(path: Path, compute_fn: Callable, force: bool = False):
    """Load `path` if it exists (and `force` is False); otherwise call
    `compute_fn()`, pickle the result to `path`, and return it."""
    path = Path(path)
    if path.exists() and not force:
        print(f"[cache hit]  {path}")
        with open(path, "rb") as fl:
            return pickle.load(fl)

    print(f"[cache miss] {path} -- computing...")
    t0 = time.time()
    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fl:
        pickle.dump(result, fl)
    print(f"[cache save] {path} ({time.time() - t0:.1f}s)")
    return result


def cache_parquet(path: Path, compute_fn: Callable, force: bool = False) -> pd.DataFrame:
    """Same pattern as `cache_pickle`, specialized for DataFrames (parquet
    round-trips dtypes -- including pandas 'category' -- more reliably and
    compactly than pickle for this size of data)."""
    path = Path(path)
    if path.exists() and not force:
        print(f"[cache hit]  {path}")
        return pd.read_parquet(path)

    print(f"[cache miss] {path} -- computing...")
    t0 = time.time()
    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=True)
    print(f"[cache save] {path} ({time.time() - t0:.1f}s)")
    return result


def clear_cache(*paths: Path):
    """Delete one or more cache files if they exist -- use this instead of
    manually rm'ing the cache/ tree when you only want to invalidate one
    stage (e.g. after changing FLOW_TOP_N but not the graph itself)."""
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()
            print(f"[cache cleared] {path}")
