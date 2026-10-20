"""Visit mode: the order one group goes through, kept separate from the window loop so the
transitions can be tested on their own.

    AI 구경        the AI from before imitation (RL baseline, no demonstrations) plays alone
    연습           free practice, nothing saved, no time limit; a small checklist per student
    ① 시범         students demonstrate (full demonstrations, group session code)
    ② 공부         train on the group's demos (+ optional base sessions), test on held-out layouts
    ③ 비교         RL baseline vs the new AI, live on the same unseen layout
    ④ AI 시연      the trained AI drives; students take a stuck arm over and finish it
    ⑤ 다시 공부    retrain with the corrections, test again
    ⑥ 비교         the AI before vs after retraining, live on the same layout
    결과           the group's numbers next to reference numbers measured beforehand
(order set by the user on 2026-10-20 after rehearsal3)

The operator moves on with B held 1 s on the right Joy-Con (or N) and starts training with X / T. Nothing here changes
what the AI learns or how it is tested; it only decides which screen and data come next.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

STEPS = ("intro", "practice", "demo", "train", "compare1", "ai", "retrain", "compare2", "result")
PRACTICE_ITEMS = (("move", "이동"), ("lower", "내리기"), ("lift", "집어 들기"), ("wrist", "손목"))
# no circled zero in Malgun Gothic, so practice has no number
STEP_LABEL = {"intro": "AI 구경", "practice": "연습", "demo": "① 시범", "train": "② 공부", "compare1": "③ 비교",
              "ai": "④ AI 시연", "retrain": "⑤ 다시 공부", "compare2": "⑥ 비교", "result": "결과"}
COMPARE_STEPS = ("compare1", "compare2")


@dataclass
class VisitFlow:
    group: str
    target_demos: int = 3
    base_sessions: tuple[str, ...] = ()
    practice: bool = True
    intro: bool = True
    step: str = ""
    practice_done: set = field(default_factory=set)
    practice_rounds: int = 0        # students who started practising (checklist resets)
    demos_ok: int = 0
    demos_tried: int = 0
    takeovers: int = 0              # takeover episodes saved since the last training
    takeovers_total: int = 0
    trainings: int = 0
    t0: float = field(default_factory=time.perf_counter)
    step_log: list = field(default_factory=list)

    def __post_init__(self):
        self.step = self.step or ("intro" if self.intro else "practice" if self.practice else "demo")
        self.step_log.append({"step": self.step, "t": 0.0})

    # -- ⓪ practice ------------------------------------------------------------------------
    def mark_practice(self, item: str) -> None:
        if self.step == "practice":
            self.practice_done.add(item)

    def reset_practice(self) -> None:
        """Next student: start the checklist again."""
        if self.step == "practice":
            self.practice_done.clear()
            self.practice_rounds += 1

    @property
    def practice_complete(self) -> bool:
        return all(k in self.practice_done for k, _ in PRACTICE_ITEMS)

    @property
    def recording(self) -> bool:
        """Intro (the RL AI) and practice episodes are never saved."""
        return self.step not in ("intro", "practice")

    @property
    def steps(self) -> tuple[str, ...]:
        return tuple(s for s in STEPS if (s != "intro" or self.intro) and (s != "practice" or self.practice))

    def _go(self, step: str) -> None:
        self.step = step
        self.step_log.append({"step": step, "t": round(time.perf_counter() - self.t0, 1)})

    def sessions(self) -> list[str]:
        return [*self.base_sessions, self.group]

    @property
    def ai_mode(self) -> bool:
        return self.step in ("compare1", "ai", "retrain", "compare2", "result")

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
            self._go("compare1" if self.step == "train" else "compare2")
        else:                       # training failed: back to where we were
            self._go("demo" if self.trainings == 0 else "ai")

    def next(self) -> str | None:
        """Operator's next key (B held / N). Returns the new step, or "train" for: start training now."""
        if self.step == "intro":
            self._go("practice" if self.practice else "demo")
            return self.step
        if self.step == "practice":
            self._go("demo")
            return "demo"
        if self.step == "compare1":
            self._go("ai")
            return "ai"
        if self.step == "compare2":
            self._go("result")
            return "result"
        if self.step == "demo" and self.demos_ok >= 1:
            return "train" if self.start_training() else None
        if self.step == "ai":
            self._go("result")
            return "result"
        return None

    def banner(self) -> list[tuple[str, str]]:
        """(label, done|current|todo) for each step."""
        steps = self.steps
        cur = steps.index(self.step)
        return [(STEP_LABEL[s], "done" if i < cur else ("current" if i == cur else "todo")) for i, s in enumerate(steps)]

    def hint(self) -> str:
        if self.step == "intro":
            return "시범 없이 혼자 연습(강화학습)한 AI예요  ·  B 길게: 연습 시작"
        if self.step == "compare1":
            return "같은 배치, 동시에: 강화학습 AI vs 여러분이 가르친 AI  ·  B 길게: AI 시연"
        if self.step == "compare2":
            return "같은 배치, 동시에: 다시 공부 전 vs 후  ·  B 길게: 결과"
        if self.step == "practice":
            if self.practice_complete:
                return "준비 완료!  B 길게: 시범 시작  ·  다음 학생은 캡처/홈"
            marks = "  ".join(("■" if k in self.practice_done else "□") + label for k, label in PRACTICE_ITEMS)
            return f"연습 (저장 안 함):  {marks}"
        if self.step == "demo":
            if self.demos_ok >= self.target_demos:
                return f"성공한 시범 {self.demos_ok}개 · X를 누르면 AI가 공부합니다"
            return f"시범 보이기: 성공 {self.demos_ok}/{self.target_demos}판 (실패한 판은 배우지 않아요)"
        if self.step == "ai":
            if self.takeovers:
                return f"고쳐 준 판 {self.takeovers}개  ·  X: 다시 공부"
            return "AI가 멈추면 그 손의 스틱이나 버튼으로 넘겨받아 마무리해 주세요  ·  B 길게: 결과 보기"
        if self.step in ("train", "retrain"):
            return "AI가 공부하고 시험 보는 중이에요"
        return "오늘의 결과"

    def summary(self) -> dict:
        return {"group": self.group, "base_sessions": list(self.base_sessions), "target_demos": self.target_demos,
                "practice_rounds": self.practice_rounds,
                "demos_ok": self.demos_ok, "demos_tried": self.demos_tried, "takeovers_total": self.takeovers_total,
                "trainings": self.trainings, "steps": self.step_log}


def load_reference(path: Path) -> list[dict]:
    """Reference lines for the result screen; every value must carry its measurement source."""
    if not Path(path).exists():
        return []
    rows = json.loads(Path(path).read_text(encoding="utf-8")).get("lines", [])
    return [r for r in rows if r.get("source")]
