"""Demonstration episodes: observation/action pairs at the 50 Hz control rate.

One .npz per episode:
  obs (T, OBS_DIM) float32   observation before the step (meta obs_version)
  act (T, 12) float32   normalised command that was executed (gym_env.commands_to_action)
  human (T, 2) bool     per hand (left, right): was a person driving it at that step?
  meta json             kind ("demo" = a person drove both hands from the start, "takeover" =
                        the AI drove until a person took a hand over), task, layout seed,
                        success, in_tray per hand, duration, session code, source, created
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
    human: np.ndarray | None = None

    @property
    def success(self) -> bool:
        return bool(self.meta.get("success"))

    @property
    def kind(self) -> str:
        return self.meta.get("kind", "demo")

    def hand_mask(self, i: int) -> np.ndarray:
        return np.ones(len(self.obs), bool) if self.human is None else self.human[:, i]

    def hand_done(self, h: str) -> bool:
        """Did this hand's cube end up in its tray? (older files: only whole-episode success)"""
        in_tray = self.meta.get("in_tray")
        return bool(in_tray[h]) if in_tray else self.success


class DemoWriter:
    def __init__(self, session: str, source: str, out_dir: Path = DEMO_DIR):
        self.session, self.source, self.out_dir = session, source, Path(out_dir)
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.human: list[tuple[bool, bool]] = []
        self.saved = 0
        self.saved_success = 0

    def start(self) -> None:
        self.obs, self.act, self.human = [], [], []

    def add(self, obs: np.ndarray, act: np.ndarray, human: tuple[bool, bool] = (True, True)) -> None:
        self.obs.append(np.asarray(obs, np.float32))
        self.act.append(np.asarray(act, np.float32))
        self.human.append((bool(human[0]), bool(human[1])))

    def finish(self, layout_seed: int, success: bool, duration: float, extra: dict | None = None,
               in_tray: dict | None = None, kind: str = "demo") -> Path | None:
        """Saves the episode; a takeover episode with no human step is not saved (nothing to learn)."""
        if not self.obs or not any(any(h) for h in self.human):
            self.start()
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        tag = "take" if kind == "takeover" else ("ok" if success else "fail")
        path = self.out_dir / f"{self.session}_{stamp}_{tag}.npz"
        from teachabot.aloha.env import OBS_VERSION
        meta = {"task": "s1", "kind": kind, "in_tray": {k: bool(v) for k, v in (in_tray or {}).items()} or None,
                "obs_version": OBS_VERSION, "layout_seed": int(layout_seed), "success": bool(success),
                "duration": round(float(duration), 3), "session": self.session, "source": self.source,
                "created": stamp, "steps": len(self.obs), **(extra or {})}
        np.savez_compressed(path, obs=np.stack(self.obs), act=np.stack(self.act), human=np.asarray(self.human),
                            meta=json.dumps(meta))
        self.saved += 1
        self.saved_success += int(success)
        self.start()
        return path


def load_demo(path: Path) -> Demo:
    with np.load(path) as z:
        human = z["human"] if "human" in z.files else None
        return Demo(Path(path), z["obs"], z["act"], json.loads(str(z["meta"])), human)


def load_demos(directory: Path = DEMO_DIR, only_success: bool = True, limit: int | None = None,
               sessions: list[str] | None = None, kinds: tuple[str, ...] = ("demo",)) -> list[Demo]:
    """Oldest first, so "the first k demos" means the order people recorded them.
    Default: full demonstrations only; pass kinds=("demo", "takeover") for corrections too."""
    demos = [load_demo(p) for p in sorted(Path(directory).glob("*.npz"))]
    demos.sort(key=lambda d: d.meta["created"])
    if sessions:
        demos = [d for d in demos if d.meta["session"] in sessions]
    from teachabot.aloha.env import OBS_VERSION
    demos = [d for d in demos if d.meta.get("obs_version", 1) == OBS_VERSION and d.kind in kinds]
    if only_success:
        demos = [d for d in demos if d.success or (d.kind == "takeover" and any(d.hand_done(h) for h in ("left", "right")))]
    return demos[:limit] if limit else demos


def trim_idle_start(demo: Demo, move_eps: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Drop the frames before the person first does anything (after pressing start)."""
    a = demo.act
    active = (np.abs(a[:, [0, 1, 6, 7]]).max(axis=1) > move_eps) | (a[:, 2] < 0) | (a[:, 8] < 0) \
        | (a[:, 5] > 0) | (a[:, 11] > 0)
    first = int(np.argmax(active)) if active.any() else 0
    return demo.obs[first:], demo.act[first:]
