"""Run a PPO checkpoint from scripts/train_rl.py exactly as its own evaluation does:
normalised observation, deterministic action held for `repeat` control steps (10 Hz)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def vecnorm_path(ckpt: Path) -> Path:
    ckpt = Path(ckpt)
    if ckpt.name == "final.zip":
        return ckpt.with_name("vecnorm_final.pkl")
    return ckpt.with_name(ckpt.name.replace("ckpt_", "vecnorm_").replace(".zip", ".pkl"))


class RLPolicy:
    def __init__(self, ckpt: Path, repeat: int = 5):
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        self.model = PPO.load(Path(ckpt), device="cpu")
        with open(vecnorm_path(ckpt), "rb") as f:
            self.vecnorm = pickle.load(f)
        self.repeat, self.k, self.a = repeat, 0, None
        self.meta = {"algo": "ppo", "ckpt": str(ckpt)}

    def reset(self) -> None:
        self.k, self.a = 0, None

    def __call__(self, obs) -> np.ndarray:
        if self.k % self.repeat == 0:
            a, _ = self.model.predict(self.vecnorm.normalize_obs(np.asarray(obs, np.float32)[None]), deterministic=True)
            self.a = a[0]
        self.k += 1
        return self.a
