r"""Compare training data sets fairly: same BC v2 settings, several training seeds, the same
held-out test layouts. Answers "is this improvement more than training-seed noise?"

    .venv\Scripts\python.exe scripts\compare_datasets.py --seeds 0 1 2 --episodes 100

Data sets: "demos" = full demonstrations only; "demos+takeover" = plus takeover corrections.
Writes data/policies/compare_<stamp>.csv (one row per data set x seed) and prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.learning.bc_hand import HandBCConfig, train_hand_bc  # noqa: E402
from teachabot.learning.demos import DEMO_DIR, load_demos  # noqa: E402
from teachabot.learning.evaluate import evaluate_policy  # noqa: E402

POLICY_DIR = ROOT / "data" / "policies"
DATASETS = {"demos": ("demo",), "demos+takeover": ("demo", "takeover")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--demo-sessions", nargs="*", help="limit full demos to these sessions (default all)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for name, kinds in DATASETS.items():
        demos = load_demos(DEMO_DIR, kinds=kinds)
        if args.demo_sessions:
            demos = [d for d in demos if d.kind != "demo" or d.meta["session"] in args.demo_sessions]
        n_demo = sum(d.kind == "demo" for d in demos)
        n_take = sum(d.kind == "takeover" for d in demos)
        for seed in args.seeds:
            pol = train_hand_bc(demos, HandBCConfig(seed=seed))
            out = POLICY_DIR / f"cmp_{stamp}_{name.replace('+', '_')}_s{seed}.npz"
            pol.save(out)
            res = evaluate_policy(out, episodes=args.episodes, workers=args.workers)
            row = {"dataset": name, "n_demos": n_demo, "n_takeovers": n_take, "train_seed": seed,
                   "n_samples": pol.meta["n_samples"], **res.summary(), "policy": out.name}
            rows.append(row)
            print(row, flush=True)
    path = POLICY_DIR / f"compare_{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name in DATASETS:
        r = [x for x in rows if x["dataset"] == name]
        sr = [x["success_rate"] for x in r]
        cb = [x["cubes_mean"] for x in r]
        print(f"{name:16s} success {statistics.mean(sr) * 100:5.1f}% (seeds: {', '.join(f'{v * 100:.0f}' for v in sr)})"
              f"  cubes {statistics.mean(cb):.2f} (seeds: {', '.join(f'{v:.2f}' for v in cb)})")
    print("csv", path)


if __name__ == "__main__":
    main()
