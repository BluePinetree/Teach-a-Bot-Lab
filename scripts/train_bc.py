r"""Train Behavioral Cloning on recorded demonstrations and evaluate it automatically.

    .venv\Scripts\python.exe scripts\train_bc.py                       # all successful demos
    .venv\Scripts\python.exe scripts\train_bc.py --limit 3             # the first 3 (recording order)
    .venv\Scripts\python.exe scripts\train_bc.py --curve 1 2 3 5 10    # success rate vs number of demos

Evaluation = closed-loop rollouts on held-out random layouts (seeds >= EVAL_SEED_BASE),
in parallel processes. Results go next to the policy as JSON (and CSV for --curve).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.learning.bc import BCConfig, train_bc  # noqa: E402
from teachabot.learning.bc_hand import HandBCConfig, train_hand_bc  # noqa: E402
from teachabot.learning.demos import DEMO_DIR, load_demos  # noqa: E402
from teachabot.learning.evaluate import evaluate_policy  # noqa: E402

POLICY_DIR = ROOT / "data" / "policies"


VAL_SEED_BASE = 2_000_000     # tuning layouts; reported numbers use the test layouts (EVAL_SEED_BASE)


def train_and_eval(demos, out: Path, cfg, episodes: int, workers: int, seed_base: int | None = None) -> dict:
    t0 = time.perf_counter()
    trainer = train_hand_bc if isinstance(cfg, HandBCConfig) else train_bc
    pol = trainer(demos, cfg, log=lambda m: print("   ", m, flush=True))
    pol.save(out)
    kw = {"seed_base": seed_base} if seed_base is not None else {}
    res = evaluate_policy(out, episodes=episodes, workers=workers, **kw)
    row = {"algo": pol.meta["algo"], "n_demos": len(demos), "n_samples": pol.meta["n_samples"], "train_s": pol.meta["train_s"],
           "eval_set": "val" if seed_base == VAL_SEED_BASE else "test", **res.summary(), "policy": out.name, "total_s": round(time.perf_counter() - t0, 1)}
    out.with_suffix(".json").write_text(json.dumps({**row, "eval_episodes": res.episodes, "bc": pol.meta},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=Path, default=DEMO_DIR)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--curve", type=int, nargs="*", help="train on the first k demos for each k")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--algo", choices=("hand", "whole"), default="hand",
                    help="hand = per-hand mirrored BC with look-ahead labels (D13); whole = first version (D12)")
    ap.add_argument("--horizon", type=int, default=10, help="hand: label = mean command over the next N active steps")
    ap.add_argument("--keep-idle", action="store_true", help="hand: do not drop waiting frames")
    ap.add_argument("--grip-ahead", type=int, default=10)
    ap.add_argument("--val", action="store_true", help="evaluate on tuning layouts instead of the test layouts")
    args = ap.parse_args()

    demos = load_demos(args.demos, sessions=args.sessions)
    print(f"{len(demos)} successful demos in {args.demos}", flush=True)
    if not demos:
        sys.exit("no successful demonstrations to learn from")
    if args.algo == "hand":
        cfg = HandBCConfig(horizon=args.horizon, grip_ahead=args.grip_ahead, drop_idle=not args.keep_idle,
                           iters=args.iters, seed=args.seed)
    else:
        cfg = BCConfig(iters=args.iters, seed=args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ks = args.curve or [args.limit or len(demos)]
    rows = []
    for k in ks:
        if k > len(demos):
            print(f"skip k={k}: only {len(demos)} demos")
            continue
        out = POLICY_DIR / f"bc_{args.algo}{args.tag}_{stamp}_k{k}.npz"
        print(f"== k={k}", flush=True)
        row = train_and_eval(demos[:k], out, cfg, args.episodes, args.workers,
                             VAL_SEED_BASE if args.val else None)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.curve:
        path = POLICY_DIR / f"bc_curve{args.tag}_{stamp}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("curve", path)


if __name__ == "__main__":
    main()
