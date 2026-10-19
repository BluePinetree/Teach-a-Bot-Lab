"""Behavioral Cloning: a small MLP that maps the observation to the person's command.

Training uses torch (CPU); the trained network is exported to plain numpy so evaluation
workers and the app run it without importing torch.

Outputs per hand: vx, vy, vz, pitch, twist (tanh, MSE loss) and grip (logit, BCE loss).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from teachabot.aloha.env import OBS_DIM
from teachabot.aloha.gym_env import ACT_DIM

from .demos import Demo, trim_idle_start

GRIP_IDX = (5, 11)
CONT_IDX = tuple(i for i in range(ACT_DIM) if i not in GRIP_IDX)


@dataclass
class BCConfig:
    hidden: int = 256
    layers: int = 3
    lr: float = 1e-3
    batch: int = 256
    iters: int = 4000
    obs_noise: float = 0.01        # Gaussian noise on normalised obs (robustness to small drift)
    std_floor: float = 0.05        # features (nearly) constant in the demos must not blow up when they move
    weight_decay: float = 1e-4
    seed: int = 0


class NumpyPolicy:
    """Inference-only copy of the trained MLP."""

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]], obs_mean: np.ndarray, obs_std: np.ndarray,
                 meta: dict | None = None):
        self.weights, self.obs_mean, self.obs_std = weights, obs_mean, obs_std
        self.meta = meta or {}

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, np.float64) - self.obs_mean) / self.obs_std
        for i, (W, b) in enumerate(self.weights):
            x = x @ W + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        out = np.empty(ACT_DIM)
        out[list(CONT_IDX)] = np.tanh(x[list(CONT_IDX)])
        out[list(GRIP_IDX)] = np.where(x[list(GRIP_IDX)] > 0.0, 1.0, -1.0)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"obs_mean": self.obs_mean, "obs_std": self.obs_std}
        for i, (W, b) in enumerate(self.weights):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        np.savez(path, meta=json.dumps(self.meta), n_layers=len(self.weights), **arrays)

    @classmethod
    def load(cls, path: Path) -> "NumpyPolicy":
        with np.load(path) as z:
            n = int(z["n_layers"])
            return cls([(z[f"W{i}"], z[f"b{i}"]) for i in range(n)], z["obs_mean"], z["obs_std"],
                       json.loads(str(z["meta"])))


def dataset(demos: list[Demo]) -> tuple[np.ndarray, np.ndarray]:
    obs, act = zip(*(trim_idle_start(d) for d in demos))
    return np.concatenate(obs).astype(np.float32), np.concatenate(act).astype(np.float32)


def train_bc(demos: list[Demo], cfg: BCConfig = BCConfig(), log=None) -> NumpyPolicy:
    import torch
    from torch import nn

    if not demos:
        raise ValueError("no demonstrations")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    X, Y = dataset(demos)
    mean, std = X.mean(0), np.maximum(X.std(0), cfg.std_floor)
    Xn = torch.tensor((X - mean) / std)
    Yc = torch.tensor(Y[:, list(CONT_IDX)])
    Yg = torch.tensor((Y[:, list(GRIP_IDX)] > 0).astype(np.float32))

    dims = [OBS_DIM] + [cfg.hidden] * (cfg.layers - 1) + [ACT_DIM]
    mods: list[nn.Module] = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    net = nn.Sequential(*mods)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters)
    bce = nn.BCEWithLogitsLoss()
    t0 = time.perf_counter()
    n = len(Xn)
    for it in range(cfg.iters):
        idx = torch.as_tensor(rng.integers(0, n, min(cfg.batch, n)))
        x = Xn[idx] + cfg.obs_noise * torch.randn(len(idx), OBS_DIM)
        out = net(x)
        loss_c = ((torch.tanh(out[:, list(CONT_IDX)]) - Yc[idx]) ** 2).mean()
        loss_g = bce(out[:, list(GRIP_IDX)], Yg[idx])
        loss = loss_c + loss_g
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (it % 1000 == 0 or it == cfg.iters - 1):
            log(f"iter {it:5d}  loss {loss.item():.4f} (cont {loss_c.item():.4f}, grip {loss_g.item():.4f})")
    linears = [m for m in net if isinstance(m, nn.Linear)]
    weights = [(m.weight.detach().numpy().T.astype(np.float64), m.bias.detach().numpy().astype(np.float64))
               for m in linears]
    meta = {"algo": "bc_mlp", "config": cfg.__dict__, "n_demos": len(demos), "n_samples": int(n),
            "demo_files": [d.path.name for d in demos], "train_s": round(time.perf_counter() - t0, 2)}
    return NumpyPolicy(weights, mean.astype(np.float64), std.astype(np.float64), meta)
