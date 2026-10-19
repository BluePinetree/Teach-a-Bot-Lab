"""Demonstration episodes: observation/action pairs at the 50 Hz control rate.

One .npz per episode:
  obs (T, OBS_DIM) float32   observation before the step (meta obs_version)
  act (T, 12) float32   normalised command the person gave (gym_env.commands_to_action)
  meta json             task, layout seed, success, duration, session code, source, created
Session codes are anonymous ("s01"); no names are stored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "data" / "demos" / "s1"


@dataclass
class Demo:
    path: Path
    obs: np.ndarray
    act: np.ndarray
    meta: dict

    @property
    def success(self) -> bool:
        return bool(self.meta.get("success"))


class DemoWriter:
    def __init__(self, session: str, source: str, out_dir: Path = DEMO_DIR):
        self.session, self.source, self.out_dir = session, source, Path(out_dir)
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.saved = 0
        self.saved_success = 0

    def start(self) -> None:
        self.obs, self.act = [], []

    def add(self, obs: np.ndarray, act: np.ndarray) -> None:
        self.obs.append(np.asarray(obs, np.float32))
        self.act.append(np.asarray(act, np.float32))

    def finish(self, layout_seed: int, success: bool, duration: float, extra: dict | None = None) -> Path | None:
        if not self.obs:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.out_dir / f"{self.session}_{stamp}_{'ok' if success else 'fail'}.npz"
        from teachabot.aloha.env import OBS_VERSION
        meta = {"task": "s1", "obs_version": OBS_VERSION, "layout_seed": int(layout_seed), "success": bool(success),
                "duration": round(float(duration), 3), "session": self.session, "source": self.source,
                "created": stamp, "steps": len(self.obs), **(extra or {})}
        np.savez_compressed(path, obs=np.stack(self.obs), act=np.stack(self.act), meta=json.dumps(meta))
        self.saved += 1
        self.saved_success += int(success)
        self.obs, self.act = [], []
        return path


def load_demo(path: Path) -> Demo:
    with np.load(path) as z:
        return Demo(Path(path), z["obs"], z["act"], json.loads(str(z["meta"])))


def load_demos(directory: Path = DEMO_DIR, only_success: bool = True, limit: int | None = None,
               sessions: list[str] | None = None) -> list[Demo]:
    """Oldest first, so "the first k demos" means the order people recorded them."""
    demos = [load_demo(p) for p in sorted(Path(directory).glob("*.npz"))]
    demos.sort(key=lambda d: d.meta["created"])
    if sessions:
        demos = [d for d in demos if d.meta["session"] in sessions]
    from teachabot.aloha.env import OBS_VERSION
    demos = [d for d in demos if d.meta.get("obs_version", 1) == OBS_VERSION]
    if only_success:
        demos = [d for d in demos if d.success]
    return demos[:limit] if limit else demos


def trim_idle_start(demo: Demo, move_eps: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Drop the frames before the person first does anything (after pressing start)."""
    a = demo.act
    active = (np.abs(a[:, [0, 1, 6, 7]]).max(axis=1) > move_eps) | (a[:, 2] < 0) | (a[:, 8] < 0) \
        | (a[:, 5] > 0) | (a[:, 11] > 0)
    first = int(np.argmax(active)) if active.any() else 0
    return demo.obs[first:], demo.act[first:]
