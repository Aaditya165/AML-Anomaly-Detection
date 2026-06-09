from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

def normalize_colname(col: str) -> str:
    col = str(col).strip()
    col = re.sub(r"\s+", " ", col)
    col = col.replace("\ufeff", "")
    return col

def find_transaction_files(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    patterns = ["*_trans.csv", "*trans.csv", "*.csv"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(p.glob(pat)))
    # preserve order, deduplicate
    seen = set()
    out = []
    for f in files:
        if f not in seen and f.is_file():
            seen.add(f)
            out.append(f)
    return out

def safe_quantile(series, q: float, default: float = 0.0) -> float:
    try:
        if len(series) == 0:
            return default
        return float(series.quantile(q))
    except Exception:
        return default
