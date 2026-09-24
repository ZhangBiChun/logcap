"""
Inspect DeepLog per-session anomaly scores (numpy .npy).

After a run with DEEPLOG_SAVE_SCORES=1, scores are written to
results/deeplog_debug/session_scores.npy. Example:

  set DEEPLOG_SAVE_SCORES=1
  python baselines.py ... --method DeepLog ...

  python analyze_deeplog_scores.py results/deeplog_debug/session_scores.npy
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize DeepLog session_scores.npy")
    p.add_argument("npy_path", help="Path to session_scores.npy")
    args = p.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("numpy required", file=sys.stderr)
        return 1

    x = np.load(args.npy_path)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    xf = x[finite]
    print(f"path: {args.npy_path}")
    print(f"n={x.size}, finite={finite.sum()}, nonfinite={np.size(x) - finite.sum()}")
    if xf.size == 0:
        return 0
    print(f"min={xf.min():.8g} max={xf.max():.8g} mean={xf.mean():.8g} std={xf.std():.8g}")
    u = np.unique(xf)
    print(f"n_unique={u.size}")
    if u.size <= 20:
        print("unique values:", u)
    # Histogram buckets for near-constant detection
    q = np.quantile(xf, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print("quantiles [0,1,5,50,95,99,100]%:", np.array2string(q, precision=6))
    # If first term of _session_score is always 100 (all windows wrong), scores often sit at 100 + small
    near_100 = np.mean(np.abs(xf - 100.0) < 1.0)
    print(f"fraction within 1.0 of 100.0: {near_100:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
