r"""Per-hand behaviour of a policy on held-out test layouts.

    .venv\Scripts\python.exe scripts\per_hand_stats.py data\policies\X.npz [--episodes 100]

Counts, for each hand: cube in its tray at the end, cube lifted at some point, and
"stalled" = the gripper target never moved more than 10 cm from its start (the failure
the takeover corrections were aimed at, DECISIONS D14).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.aloha.gym_env import EVAL_SEED_BASE, action_to_commands  # noqa: E402
from teachabot.aloha.scene import HANDS  # noqa: E402

STALL_M = 0.10


def run(args: tuple[str, int]) -> dict:
    path, seed = args
    from teachabot.aloha.env import AlohaSortEnv
    from teachabot.learning.bc import NumpyPolicy
    pol, env = NumpyPolicy.load(Path(path)), AlohaSortEnv()
    obs = env.reset(seed)
    start = {h: env.targets[h].pos.copy() for h in HANDS}
    moved = {h: 0.0 for h in HANDS}
    lifted = {h: False for h in HANDS}
    while True:
        obs, info = env.step(action_to_commands(pol(obs)))
        for h in HANDS:
            moved[h] = max(moved[h], float(np.linalg.norm(env.targets[h].pos - start[h])))
            lifted[h] |= bool(env.cube_pos(h)[2] > 0.05)
        if info.done:
            return {h: {"in_tray": bool(info.in_tray[h]), "lifted": lifted[h], "stalled": moved[h] < STALL_M}
                    for h in HANDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", type=Path)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(run, [(str(args.policy), EVAL_SEED_BASE + k) for k in range(args.episodes)])
    summary = {h: {k: sum(r[h][k] for r in res) for k in ("in_tray", "lifted", "stalled")} for h in HANDS}
    summary["episodes"] = args.episodes
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
