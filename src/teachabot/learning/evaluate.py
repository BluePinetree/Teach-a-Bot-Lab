"""Automatic evaluation: run a policy on held-out random layouts in parallel processes.

Layout seeds start at gym_env.EVAL_SEED_BASE, which no training/demo episode uses
(demos come from the app's own seed sequence). Each evaluation episode is a real
closed-loop rollout of the policy in the physics simulation.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from teachabot.aloha.gym_env import EVAL_SEED_BASE, action_to_commands

_POLICY = None
_ENV = None


@dataclass
class EvalResult:
    success_rate: float
    cubes_mean: float
    episodes: list[dict]
    wall_s: float

    def summary(self) -> dict:
        ok = [e["time"] for e in self.episodes if e["success"]]
        lifted = [e.get("lifted", 0) for e in self.episodes]
        return {"episodes": len(self.episodes), "success_rate": round(self.success_rate, 3),
                "lifted_mean": round(float(np.mean(lifted)), 3),
                "cubes_mean": round(self.cubes_mean, 3), "success_time_mean": round(float(np.mean(ok)), 2) if ok else None,
                "wall_s": round(self.wall_s, 1)}


def run_episode(policy, env, layout_seed: int) -> dict:
    obs = env.reset(layout_seed)
    if hasattr(policy, "reset"):
        policy.reset()
    lifted = {"left": False, "right": False}
    while True:
        obs, info = env.step(action_to_commands(policy(obs)))
        for h in lifted:                       # partial skill: did this hand ever lift its cube?
            lifted[h] |= bool(env.targets[h].grip and env.cube_pos(h)[2] > 0.05)
        if info.done:
            return {"layout_seed": layout_seed, "success": bool(info.success), "time": round(info.time, 2),
                    "cubes": int(sum(info.in_tray.values())), "lifted": int(sum(lifted.values()))}


def _init_worker(policy_path: str, time_limit: float) -> None:
    global _POLICY, _ENV
    from teachabot.aloha.env import AlohaSortEnv
    if str(policy_path).endswith(".zip"):          # PPO checkpoint (RL baseline)
        from teachabot.learning.rl_policy import RLPolicy
        _POLICY = RLPolicy(Path(policy_path))
    else:
        from teachabot.learning.bc import NumpyPolicy
        _POLICY = NumpyPolicy.load(Path(policy_path))
    _ENV = AlohaSortEnv(time_limit=time_limit)


def _worker(layout_seed: int) -> dict:
    return run_episode(_POLICY, _ENV, layout_seed)


def evaluate_policy(policy_path: Path, episodes: int = 20, workers: int = 10, time_limit: float = 40.0,
                    seed_base: int = EVAL_SEED_BASE) -> EvalResult:
    t0 = time.perf_counter()
    seeds = [seed_base + k for k in range(episodes)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, episodes), initializer=_init_worker, initargs=(str(policy_path), time_limit)) as pool:
        eps = pool.map(_worker, seeds)
    return EvalResult(float(np.mean([e["success"] for e in eps])), float(np.mean([e["cubes"] for e in eps])), eps,
                      time.perf_counter() - t0)
