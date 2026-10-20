r"""Watch the RL baseline runs: evaluate every saved checkpoint on the same held-out test
layouts as the BC numbers (default 40), and keep a table per run.

    .venv\Scripts\python.exe scripts\monitor_rl.py            # loop: check for new checkpoints every 10 min
    .venv\Scripts\python.exe scripts\monitor_rl.py --once

Writes data/rl_runs/<run>/eval_test.csv: steps, practice hours, success rate, cubes per
episode, hands that lifted their cube per episode (partial skill). Used to pick when the RL
shows "practised a lot, still clearly worse than imitation" for the visit (numbers as measured).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.aloha.env import CONTROL_DT  # noqa: E402
from teachabot.learning.evaluate import evaluate_policy  # noqa: E402

FIELDS = ["steps", "practice_hours", "episodes", "success_rate", "cubes_mean", "lifted_mean", "eval_s"]


def runs() -> list[Path]:
    out = []
    for reward in ("sparse", "dense"):
        r = sorted((ROOT / "data" / "rl_runs").glob(f"{reward}_2*"))
        if r:
            out.append(r[-1])
    return out


def done_steps(table: Path) -> set[int]:
    if not table.exists():
        return set()
    with table.open(encoding="utf-8") as f:
        return {int(r["steps"]) for r in csv.DictReader(f)}


def check_once(episodes: int, workers: int, repeat: int = 5) -> None:
    for run in runs():
        table = run / "eval_test.csv"
        seen = done_steps(table)
        ckpts = sorted(run.glob("ckpt_*.zip"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        for ck in ckpts:
            steps = int(re.findall(r"\d+", ck.stem)[0])
            if steps in seen:
                continue
            res = evaluate_policy(ck, episodes=episodes, workers=workers).summary()
            row = {"steps": steps, "practice_hours": round(steps * repeat * CONTROL_DT / 3600, 1),
                   "episodes": res["episodes"], "success_rate": res["success_rate"], "cubes_mean": res["cubes_mean"],
                   "lifted_mean": res["lifted_mean"], "eval_s": res["wall_s"]}
            new = not table.exists()
            with table.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(run.name, row, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--every", type=float, default=600, help="seconds between checks")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        check_once(args.episodes, args.workers)
        if args.once:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
