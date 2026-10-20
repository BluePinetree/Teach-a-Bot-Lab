"""Side-by-side live comparison: the RL baseline and the imitation-learned AIs on the same layout.

2026-10-20 (after rehearsal3): "강화학습으로 학습시켜 놓은 거랑 모방학습 시험 보여 주고 나서,
나아진 거 에피소드 동시에 실행해서 비교해 주는 화면도 있으면 좋을 것 같아."

Every pane is a real closed-loop run of its policy (no recordings): same unseen layout
(test-layout family, never trained on), same start, stepped together at real time. When all are
finished the panes hold for a moment and move on to the next layout. The tally counts only what
was shown here; the measured success rates (40 unseen layouts) are shown next to it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from teachabot.aloha.env import CONTROL_DT, AlohaSortEnv
from teachabot.aloha.gym_env import action_to_commands

COMPARE_SEED_BASE = 1_000_800      # test-layout family, after the 40 evaluation layouts and the trial view
COMPARE_TIME_LIMIT = 25.0          # shorter than a round: the RL baseline never finishes, so it would idle 40 s
PANE_W, PANE_H = 626, 352
HOLD_S = 2.5                       # pause on the finished layout before the next one


@dataclass
class Runner:
    key: str
    label: str
    color: tuple
    path: Path | None
    measured: float | None = None          # success rate on the 40 unseen evaluation layouts
    policy: object = None
    env: AlohaSortEnv | None = None
    renderer: mujoco.Renderer | None = None
    info: object = None
    wins: int = 0
    shown: int = 0
    error: str | None = None
    loading: bool = False
    results: list = field(default_factory=list)


def _load(path: Path):
    if str(path).endswith(".zip"):
        from teachabot.learning.rl_policy import RLPolicy
        return RLPolicy(path)
    from teachabot.learning.bc import NumpyPolicy
    return NumpyPolicy.load(path)


class CompareView:
    def __init__(self):
        self.runners: list[Runner] = []
        self.k = 0
        self.t_last = None
        self.acc = 0.0
        self.done_at = None
        self._preloaded: dict[str, object] = {}

    def preload(self, path: Path) -> None:
        """Load a slow policy (the PPO checkpoint takes ~5 s) in the background."""
        key = str(path)
        if key in self._preloaded or not Path(path).exists():
            return
        self._preloaded[key] = None

        def run():
            try:
                self._preloaded[key] = _load(Path(path))
            except Exception as e:                      # shown on the pane
                self._preloaded[key] = e
        threading.Thread(target=run, daemon=True).start()

    def set_runners(self, specs: list[dict]) -> None:
        """specs: {key, label, color, path, measured}. Keeps running panes whose policy is unchanged."""
        old = {r.key: r for r in self.runners}
        new = []
        changed = False
        for s in specs:
            r = old.get(s["key"])
            if r is None or r.path != s["path"]:
                r = Runner(s["key"], s["label"], s["color"], s["path"], s.get("measured"))
                changed = True
            r.label, r.measured = s["label"], s.get("measured")
            new.append(r)
        if changed or len(new) != len(self.runners):
            self.runners = new
            for r in self.runners:
                r.wins, r.shown, r.results = 0, 0, []
            self._next_layout(restart=True)

    def _ensure(self, r: Runner) -> bool:
        if r.policy is not None:
            return True
        if r.path is None or not Path(r.path).exists():
            r.error = "정책 파일 없음"
            return False
        key = str(r.path)
        if key not in self._preloaded:
            if str(r.path).endswith(".zip"):
                self.preload(r.path)
            else:
                self._preloaded[key] = _load(Path(r.path))
        got = self._preloaded.get(key)
        if got is None:
            r.loading = True
            return False
        if isinstance(got, Exception):
            r.error = f"불러오기 실패: {type(got).__name__}"
            return False
        r.policy, r.loading = got, False
        r.env = AlohaSortEnv(time_limit=COMPARE_TIME_LIMIT)
        r.renderer = mujoco.Renderer(r.env.model, PANE_H, PANE_W)
        self._reset(r)
        return True

    def _reset(self, r: Runner) -> None:
        r.env.reset(COMPARE_SEED_BASE + self.k)
        if hasattr(r.policy, "reset"):
            r.policy.reset()
        r.info = None

    def _next_layout(self, restart: bool = False) -> None:
        if not restart:
            self.k += 1
        self.acc, self.done_at = 0.0, None
        for r in self.runners:
            if r.env is not None:
                self._reset(r)

    def update(self) -> list[tuple[Runner, np.ndarray | None, str]]:
        """Advance all ready panes together (real time) and render them."""
        now = time.perf_counter()
        dt = 0.0 if self.t_last is None or now - self.t_last > 0.5 else now - self.t_last
        self.t_last = now
        ready = [self._ensure(r) for r in self.runners]
        if any(not ok and r.error is None for r, ok in zip(self.runners, ready)):
            # still loading: wait so every pane starts on the same layout at the same time
            return [(r, self._frame(r) if ok else None, self._status(r)) for r, ok in zip(self.runners, ready)]
        active = [r for r, ok in zip(self.runners, ready) if ok]   # a broken pane shows its error, the rest run
        if self.done_at is not None:
            if now - self.done_at > HOLD_S:
                self._next_layout()
        else:
            self.acc = min(self.acc + dt, 3 * CONTROL_DT)
            while self.acc >= CONTROL_DT:
                self.acc -= CONTROL_DT
                for r in active:
                    if r.info is None or not r.info.done:
                        _, r.info = r.env.step(action_to_commands(r.policy(r.env.observation())))
                        if r.info.done:
                            r.shown += 1
                            r.wins += int(r.info.success)
                            r.results.append(bool(r.info.success))
                if all(r.info is not None and r.info.done for r in active):
                    self.done_at = now
                    break
        return [(r, self._frame(r) if ok else None, self._status(r)) for r, ok in zip(self.runners, ready)]

    @staticmethod
    def _frame(r: Runner) -> np.ndarray | None:
        if r.renderer is None:
            return None
        r.renderer.update_scene(r.env.data, camera="demo_cam")
        return r.renderer.render()

    @staticmethod
    def _status(r: Runner) -> str:
        if r.error:
            return r.error
        if r.loading or r.env is None:
            return "불러오는 중…"
        info = r.info
        if info is None:
            return "출발"
        n = sum(info.in_tray.values())
        if info.done:
            return f"성공! {info.time:.1f}초" if info.success else f"실패 · 상자 안 {n}개"
        return f"{info.time:4.1f}초 · 상자 안 {n}개"
