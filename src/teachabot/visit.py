"""Visit mode: the order one group goes through, kept separate from the window loop so the
transitions can be tested on their own.

    ① 시범 보이기   students demonstrate (full demonstrations, group session code)
    ② AI 공부       train on the group's demos (+ optional base sessions), test on held-out layouts
    ③ AI 시연       the trained AI drives; students take a stuck arm over and finish it
    ④ 다시 공부     retrain with the corrections, test again (③ ↔ ④ can repeat)
    ⑤ 결과         the group's bars next to reference numbers measured beforehand

The operator moves on with N (next) and starts training with X / T. Nothing here changes
what the AI learns or how it is tested; it only decides which screen and data come next.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

STEPS = ("demo", "train", "ai", "retrain", "result")
STEP_LABEL = {"demo": "① 시범", "train": "② AI 공부", "ai": "③ AI 시연", "retrain": "④ 다시 공부", "result": "⑤ 결과"}


@dataclass
class VisitFlow:
    group: str
    target_demos: int = 3
    base_sessions: tuple[str, ...] = ()
    step: str = "demo"
    demos_ok: int = 0
    demos_tried: int = 0
    takeovers: int = 0              # takeover episodes saved since the last training
    takeovers_total: int = 0
    trainings: int = 0
    t0: float = field(default_factory=time.perf_counter)
    step_log: list = field(default_factory=list)

    def __post_init__(self):
        self.step_log.append({"step": self.step, "t": 0.0})

    def _go(self, step: str) -> None:
        self.step = step
        self.step_log.append({"step": step, "t": round(time.perf_counter() - self.t0, 1)})

    def sessions(self) -> list[str]:
        return [*self.base_sessions, self.group]

    @property
    def ai_mode(self) -> bool:
        return self.step in ("ai", "retrain", "result")

    def on_episode_saved(self, kind: str, success: bool) -> None:
        if kind == "demo":
            self.demos_tried += 1
            self.demos_ok += int(success)
        elif kind == "takeover":
            self.takeovers += 1
            self.takeovers_total += 1

    def can_train(self) -> bool:
        return (self.step == "demo" and self.demos_ok >= 1) or (self.step == "ai" and self.takeovers >= 1)

    def start_training(self) -> bool:
        """Called when the operator presses X; returns False if there is nothing new to learn."""
        if not self.can_train():
            return False
        self._go("train" if self.step == "demo" else "retrain")
        return True

    def on_trained(self, ok: bool) -> None:
        if ok:
            self.trainings += 1
            self.takeovers = 0
            self._go("ai")
        else:                       # training failed: back to where we were
            self._go("demo" if self.trainings == 0 else "ai")

    def next(self) -> str | None:
        """Operator's N key. Returns "train" when that means: start training now."""
        if self.step == "demo" and self.demos_ok >= 1:
            return "train" if self.start_training() else None
        if self.step == "ai":
            self._go("result")
            return "result"
        return None

    def banner(self) -> list[tuple[str, str]]:
        """(label, done|current|todo) for each step."""
        cur = STEPS.index(self.step)
        out = []
        for i, s in enumerate(STEPS):
            if s == "retrain" and self.step == "ai" and self.trainings >= 2:
                state = "done"
            else:
                state = "done" if i < cur else ("current" if i == cur else "todo")
            out.append((STEP_LABEL[s], state))
        return out

    def hint(self) -> str:
        if self.step == "demo":
            if self.demos_ok >= self.target_demos:
                return f"성공한 시범 {self.demos_ok}개 · X를 누르면 AI가 공부합니다"
            return f"시범 보이기: 성공 {self.demos_ok}/{self.target_demos}판 (실패한 판은 배우지 않아요)"
        if self.step == "ai":
            if self.takeovers:
                return f"고쳐 준 판 {self.takeovers}개 · X: 다시 공부  ·  N: 결과 보기"
            return "AI가 멈추면 그 손의 스틱이나 버튼으로 넘겨받아 마무리해 주세요  ·  N: 결과 보기"
        if self.step in ("train", "retrain"):
            return "AI가 공부하고 시험 보는 중이에요"
        return "오늘의 결과"

    def summary(self) -> dict:
        return {"group": self.group, "base_sessions": list(self.base_sessions), "target_demos": self.target_demos,
                "demos_ok": self.demos_ok, "demos_tried": self.demos_tried, "takeovers_total": self.takeovers_total,
                "trainings": self.trainings, "steps": self.step_log}


def load_reference(path: Path) -> list[dict]:
    """Reference lines for the result screen; every value must carry its measurement source."""
    if not Path(path).exists():
        return []
    rows = json.loads(Path(path).read_text(encoding="utf-8")).get("lines", [])
    return [r for r in rows if r.get("source")]
