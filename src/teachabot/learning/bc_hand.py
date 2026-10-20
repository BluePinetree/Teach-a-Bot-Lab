"""BC v2: one small network per *hand*, shared by both hands through left/right mirroring.

Why (measured on the first human demos, DECISIONS D13): people flick the stick in short
pulses and move one hand at a time (each stick active ~11% of frames, both together 5%).
Copying the instantaneous command then teaches "mostly wait", and *when* a hand waits
depends on the person's attention, not on anything the robot can see. So:
  - per hand, the frames where it only waits (job unfinished) are cut out of its timeline,
    so the frame before a pause is followed by what the person did *after* the pause
    (a still state otherwise gets both "wait" and "act" labels and the net averages to
    hovering forever),
  - labels are the mean command over the next `horizon` active frames, and the grip state
    `grip_ahead` active frames later,
  - each hand sees only its own state (target, gripper, its cube, its tray); the right hand
    is mirrored (x -> -x) into the left hand's frame, doubling the data,
  - the hand's own wrist command is not an input (it fed back and drifted).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from teachabot.aloha.env import Z_CARRY
from teachabot.aloha.scene import HANDS, TRAY_INNER, TRAY_WALL_H

from .demos import Demo, trim_idle_start

# obs v2 layout (see AlohaSortEnv.observation)
_HAND_BASE = {"left": 0, "right": 10}
_CUBE_BASE = {"left": 20, "right": 25}
_REL_BASE = {"left": 30, "right": 35}
HAND_OBS_DIM = 18
HAND_ACT_DIM = 6        # vx, vy, vz, pitch, twist, grip
# feature positions holding an x coordinate (negated when mirroring) and sin(4 yaw)
_X_FEATS = (0, 3, 8, 13, 16)
_SIN_FEAT = 11


def hand_obs(obs: np.ndarray, h: str) -> np.ndarray:
    """[target(3), gripper site(3), finger opening, grip cmd, cube(3), sin4yaw, cos4yaw,
    cube - target(3), tray - cube(2)] in the left hand's frame."""
    obs = np.asarray(obs, np.float64)
    b, c, r = _HAND_BASE[h], _CUBE_BASE[h], _REL_BASE[h]
    f = np.concatenate([obs[b:b + 6], obs[b + 8:b + 10], obs[c:c + 5], obs[r:r + 5]])
    if h == "right":
        f[list(_X_FEATS)] *= -1.0
        f[_SIN_FEAT] *= -1.0
    return f


def hand_act(act: np.ndarray, h: str) -> np.ndarray:
    a = np.asarray(act, np.float64)[6 * HANDS.index(h):6 * HANDS.index(h) + 6].copy()
    if h == "right":
        a[0] *= -1.0     # vx
        a[4] *= -1.0     # twist
    return a


def unmirror_act(a: np.ndarray, h: str) -> np.ndarray:
    a = np.asarray(a, np.float64).copy()
    if h == "right":
        a[0] *= -1.0
        a[4] *= -1.0
    return a


@dataclass
class HandBCConfig:
    horizon: int = 10                 # label = mean command over the next 10 (active) frames
    grip_ahead: int = 10              # grip label = grip state 10 (active) frames later (tuned on val layouts)
    drop_idle: bool = True            # cut the frames where the hand only waits
    hidden: int = 256
    layers: int = 3
    lr: float = 1e-3
    batch: int = 256
    iters: int = 4000
    obs_noise: float = 0.01
    std_floor: float = 0.05
    weight_decay: float = 1e-4
    seed: int = 0
    notes: dict = field(default_factory=dict)


def _forward_mean(x: np.ndarray, h: int) -> np.ndarray:
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    n = len(x)
    end = np.minimum(np.arange(n) + h, n)
    return (c[end] - c[np.arange(n)]) / (end - np.arange(n))[:, None]


def _still(o: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Frames where this hand does nothing: stick centred, not lowering, at carry height,
    grip unchanged from the previous frame, and its cube not yet in the tray."""
    grip = a[:, 5] > 0
    changed = np.r_[False, grip[1:] != grip[:-1]]
    still = (np.hypot(a[:, 0], a[:, 1]) < 0.05) & (a[:, 2] > 0) & (o[:, 2] >= Z_CARRY - 1e-3) & ~changed
    in_tray = (np.abs(o[:, 16]) < TRAY_INNER / 2) & (np.abs(o[:, 17]) < TRAY_INNER / 2) & (o[:, 10] < TRAY_WALL_H)
    return still & ~in_tray


def hand_dataset(demos: list[Demo], cfg: HandBCConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per hand: optionally cut the frames where the hand just waits (time compression), then
    label each remaining frame with the mean command over the next `horizon` frames and the
    grip state `grip_ahead` frames later."""
    X, Y = [], []
    kept = dropped = 0
    for d in demos:
        if d.kind == "demo":
            first = len(d.obs) - len(trim_idle_start(d)[0])
        else:
            first = 0
        for i_h, h in enumerate(HANDS):
            # a hand is learned from only where a person drove it and its cube ended in its tray
            mask = d.hand_mask(i_h)[first:]
            if not d.hand_done(h) or not mask.any():
                continue
            obs, act = d.obs[first:][mask], d.act[first:][mask]
            o = np.stack([hand_obs(x, h) for x in obs])
            a = np.stack([hand_act(x, h) for x in act])
            if cfg.drop_idle:
                keep = ~_still(o, a)
                dropped += int((~keep).sum())
                o, a = o[keep], a[keep]
            if not len(o):
                continue
            y = a.copy()
            y[:, :5] = _forward_mean(a[:, :5], cfg.horizon)
            ahead = np.minimum(np.arange(len(a)) + cfg.grip_ahead, len(a) - 1)
            y[:, 5] = a[ahead, 5]
            kept += len(o)
            X.append(o)
            Y.append(y)
    stats = {"kept": kept, "dropped_idle": dropped}
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32), stats


class HandPolicy:
    """Numpy inference; same call signature as NumpyPolicy (40-d obs -> 12-d action)."""
    kind = "hand_mirror"

    def __init__(self, weights, obs_mean, obs_std, meta=None):
        self.weights, self.obs_mean, self.obs_std = weights, obs_mean, obs_std
        self.meta = meta or {}

    def hand(self, f: np.ndarray) -> np.ndarray:
        x = (f - self.obs_mean) / self.obs_std
        for i, (W, b) in enumerate(self.weights):
            x = x @ W + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        out = np.tanh(x)
        out[5] = 1.0 if x[5] > 0.0 else -1.0
        return out

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([unmirror_act(self.hand(hand_obs(obs, h)), h) for h in HANDS])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"obs_mean": self.obs_mean, "obs_std": self.obs_std}
        for i, (W, b) in enumerate(self.weights):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        np.savez(path, meta=json.dumps({**self.meta, "kind": self.kind}), n_layers=len(self.weights), **arrays)


def train_hand_bc(demos: list[Demo], cfg: HandBCConfig = HandBCConfig(), log=None, progress=None) -> HandPolicy:
    """progress(iteration, iters, loss, net): called every 50 iterations (screen progress, live weights)."""
    import torch
    from torch import nn

    if not demos:
        raise ValueError("no demonstrations")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    X, Y, counts = hand_dataset(demos, cfg)
    mean, std = X.mean(0), np.maximum(X.std(0), cfg.std_floor)
    Xn = torch.tensor((X - mean) / std)
    Yc = torch.tensor(Y[:, :5])
    Yg = torch.tensor((Y[:, 5] > 0).astype(np.float32))
    dims = [HAND_OBS_DIM] + [cfg.hidden] * (cfg.layers - 1) + [HAND_ACT_DIM]
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
        x = Xn[idx] + cfg.obs_noise * torch.randn(len(idx), HAND_OBS_DIM)
        out = net(x)
        loss_c = ((torch.tanh(out[:, :5]) - Yc[idx]) ** 2).mean()
        loss_g = bce(out[:, 5], Yg[idx])
        loss = loss_c + loss_g
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if progress and (it % 50 == 0 or it == cfg.iters - 1):
            progress(it + 1, cfg.iters, float(loss.item()), net)
        if log and (it % 1000 == 0 or it == cfg.iters - 1):
            log(f"iter {it:5d}  loss {loss.item():.4f} (cont {loss_c.item():.4f}, grip {loss_g.item():.4f})")
    weights = [(m.weight.detach().numpy().T.astype(np.float64), m.bias.detach().numpy().astype(np.float64))
               for m in net if isinstance(m, nn.Linear)]
    cfg_d = {k: v for k, v in cfg.__dict__.items()}
    meta = {"algo": "bc_hand_mirror", "config": cfg_d, "n_demos": sum(d.kind == "demo" for d in demos),
            "n_takeovers": sum(d.kind == "takeover" for d in demos), "n_samples": int(n), **counts,
            "demo_files": [d.path.name for d in demos], "train_s": round(time.perf_counter() - t0, 2)}
    return HandPolicy(weights, mean.astype(np.float64), std.astype(np.float64), meta)
