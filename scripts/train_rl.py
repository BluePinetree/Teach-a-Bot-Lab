r"""RL baseline for S1: PPO from scratch, no demonstrations (DECISIONS D11).

    .venv\\Scripts\\python.exe scripts\\train_rl.py --reward sparse
    .venv\\Scripts\\python.exe scripts\\train_rl.py --reward dense

Runs until --total-steps or Ctrl-C. Writes to data/rl_runs/<reward>_<stamp>/:
  progress.csv   one row per rollout: steps, wall time, simulated practice hours, episodes,
                 training success rate / cubes in tray / return over the last 100 episodes,
                 and (every --eval-every steps) deterministic evaluation on held-out layouts
  ckpt_<steps>.zip + vecnorm_<steps>.pkl   the policy at each evaluation (to show its real behaviour)
  config.json
Nothing here is staged: the numbers are whatever the run produces.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.aloha.env import CONTROL_DT, OBS_VERSION  # noqa: E402
from teachabot.aloha.gym_env import EVAL_SEED_BASE, AlohaSortGymEnv  # noqa: E402


def make_env(reward: str, repeat: int, rank: int, seed: int):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(AlohaSortGymEnv(reward, action_repeat=repeat), info_keywords=("success", "cubes_in_tray"))
        env.reset(seed=seed + rank)
        return env
    return _init


def evaluate(model, vecnorm, reward: str, repeat: int, episodes: int) -> dict:
    env = AlohaSortGymEnv(reward, action_repeat=repeat)
    succ, cubes = [], []
    for k in range(episodes):
        obs, _ = env.reset(options={"layout_seed": EVAL_SEED_BASE + k})
        while True:
            a, _ = model.predict(vecnorm.normalize_obs(obs[None]), deterministic=True)
            obs, _, term, trunc, info = env.step(a[0])
            if term or trunc:
                break
        succ.append(info["success"])
        cubes.append(info["cubes_in_tray"])
    return {"eval_success": float(np.mean(succ)), "eval_cubes": float(np.mean(cubes))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=("sparse", "dense"), required=True)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=5, help="control steps per RL decision (5 = 10 Hz)")
    ap.add_argument("--total-steps", type=int, default=500_000_000)
    ap.add_argument("--eval-every", type=int, default=2_000_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(args.torch_threads)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or ROOT / "data" / "rl_runs" / f"{args.reward}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    hyper = dict(n_steps=1024, batch_size=4096, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[256, 256]))
    (out / "config.json").write_text(json.dumps({**vars(args), "out": str(out), "algo": "PPO (stable-baselines3 2.9.0)",
                                                 "hyper": hyper, "obs_version": OBS_VERSION, "decision_hz": 1 / (CONTROL_DT * args.repeat),
                                                 "started": stamp}, indent=2, default=str), encoding="utf-8")

    venv = SubprocVecEnv([make_env(args.reward, args.repeat, i, args.seed * 1000) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=(args.reward == "dense"), clip_obs=10.0)
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0, device="cpu", **hyper)

    fields = ["steps", "wall_s", "practice_hours", "episodes", "train_success_100", "train_cubes_100",
              "train_return_100", "eval_success", "eval_cubes"]
    csv_path = out / "progress.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(fields)

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0 = time.perf_counter()
            self.eps = 0
            self.recent = deque(maxlen=100)
            self.next_eval = args.eval_every

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                ep = info.get("episode")
                if ep is not None:
                    self.eps += 1
                    self.recent.append((info["success"], info["cubes_in_tray"], ep["r"]))
            return True

        def _on_rollout_end(self) -> None:
            steps = self.num_timesteps
            row = {"steps": steps, "wall_s": round(time.perf_counter() - self.t0, 1),
                   "practice_hours": round(steps * args.repeat * CONTROL_DT / 3600, 2), "episodes": self.eps}
            if self.recent:
                r = np.asarray(self.recent, float)
                row.update(train_success_100=round(r[:, 0].mean(), 3), train_cubes_100=round(r[:, 1].mean(), 3),
                           train_return_100=round(r[:, 2].mean(), 3))
            if steps >= self.next_eval:
                self.next_eval += args.eval_every
                vn = self.model.get_env()
                row.update(evaluate(self.model, vn, args.reward, args.repeat, args.eval_episodes))
                self.model.save(out / f"ckpt_{steps}.zip")
                vn.save(str(out / f"vecnorm_{steps}.pkl"))
                print(json.dumps(row), flush=True)
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([row.get(k, "") for k in fields])

    print("run dir", out, flush=True)
    try:
        model.learn(total_timesteps=args.total_steps, callback=Progress())
    except KeyboardInterrupt:
        print("stopped by user", flush=True)
    finally:
        model.save(out / "final.zip")
        model.get_env().save(str(out / "vecnorm_final.pkl"))
        try:
            venv.close()
        except (EOFError, BrokenPipeError):      # workers already stopped by Ctrl-C
            pass
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
