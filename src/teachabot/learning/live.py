"""Retrain-and-evaluate during a session (used by the app's "다시 학습" key).

Data: every full demonstration that succeeded plus every takeover episode in which a person
fixed at least one hand (only the frames the person drove, only for hands whose cube ended
in the tray; see bc_hand.hand_dataset). Training runs in a background thread so the window
keeps drawing; evaluation is the same closed-loop test as scripts/train_bc.py (held-out
layouts, parallel processes). Nothing is staged: the bar shows whatever the evaluation gives.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from .bc_hand import HandBCConfig, train_hand_bc
from .demos import DEMO_DIR, load_demos
from .evaluate import evaluate_policy

ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = ROOT / "data" / "policies"


class LiveTrainer:
    def __init__(self, demo_dir: Path = DEMO_DIR, episodes: int = 40, workers: int = 10,
                 cfg: HandBCConfig | None = None):
        self.demo_dir, self.episodes, self.workers = demo_dir, episodes, workers
        self.cfg = cfg or HandBCConfig()
        self.state = "idle"            # idle | training | evaluating | done | error
        self.message = ""
        self.result: dict | None = None
        self.history: list[dict] = []
        self.t_started = 0.0
        self._thread: threading.Thread | None = None

    def add_baseline(self, policy_path: Path) -> None:
        """Show the starting policy's own evaluation (from its JSON, if it was evaluated)."""
        j = Path(policy_path).with_suffix(".json")
        if j.exists():
            d = json.loads(j.read_text(encoding="utf-8"))
            self.history.append({"label": f"시범 {d.get('n_demos', '?')}", "success_rate": d["success_rate"],
                                 "cubes_mean": d["cubes_mean"], "episodes": d.get("episodes"),
                                 "policy": str(policy_path)})

    def data_counts(self) -> tuple[int, int]:
        demos = load_demos(self.demo_dir, kinds=("demo", "takeover"))
        return sum(d.kind == "demo" for d in demos), sum(d.kind == "takeover" for d in demos)

    def busy(self) -> bool:
        return self.state in ("training", "evaluating")

    def start(self) -> bool:
        if self.busy():
            return False
        self.state, self.message, self.result = "training", "", None
        self.t_started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        try:
            demos = load_demos(self.demo_dir, kinds=("demo", "takeover"))
            n_demo = sum(d.kind == "demo" for d in demos)
            n_take = sum(d.kind == "takeover" for d in demos)
            self.message = f"시범 {n_demo}개 + 개입 {n_take}번으로 학습 중"
            pol = train_hand_bc(demos, self.cfg)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = POLICY_DIR / f"live_{stamp}_d{n_demo}_t{n_take}.npz"
            pol.save(path)
            self.state = "evaluating"
            self.message = f"처음 보는 배치 {self.episodes}판으로 시험 중"
            res = evaluate_policy(path, episodes=self.episodes, workers=self.workers)
            row = {"label": f"시범 {n_demo}" + (f" + 개입 {n_take}" if n_take else ""), **res.summary(),
                   "policy": str(path), "n_demos": n_demo, "n_takeovers": n_take,
                   "train_s": pol.meta["train_s"], "total_s": round(time.perf_counter() - self.t_started, 1)}
            path.with_suffix(".json").write_text(json.dumps({**row, "eval_episodes": res.episodes, "bc": pol.meta},
                                                            ensure_ascii=False, indent=2), encoding="utf-8")
            self.history.append(row)
            self.result = row
            self.state = "done"
        except Exception as e:      # shown on screen; the app keeps running with the old policy
            self.state, self.message = "error", f"{type(e).__name__}: {e}"
            traceback.print_exc()
