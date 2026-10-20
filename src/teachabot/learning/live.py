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

import numpy as np

from .bc_hand import HandBCConfig, train_hand_bc
from .demos import DEMO_DIR, load_demos
from .evaluate import evaluate_policy

ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = ROOT / "data" / "policies"


def load_session_demos(demo_dir: Path, sessions: list[str] | None) -> list:
    """Demos + takeovers of the given sessions; "name:k" keeps only the first k full demos of
    that session (recording order). None = everything."""
    if sessions is None:
        return load_demos(demo_dir, kinds=("demo", "takeover"))
    limits = {}
    for item in sessions:
        name, _, k = item.partition(":")
        limits[name] = int(k) if k else None
    demos = load_demos(demo_dir, kinds=("demo", "takeover"), sessions=list(limits))
    out, used = [], {}
    for d in demos:
        name, k = d.meta["session"], limits[d.meta["session"]]
        if k is not None:
            if d.kind != "demo" or used.get(name, 0) >= k:
                continue
            used[name] = used.get(name, 0) + 1
        out.append(d)
    return out


class LiveTrainer:
    def __init__(self, demo_dir: Path = DEMO_DIR, episodes: int = 40, workers: int = 10,
                 cfg: HandBCConfig | None = None, sessions: list[str] | None = None, label_prefix: str = ""):
        """`sessions`: only these session codes are learned from (visit mode: base + group).
        "s01:5" means the first 5 full demonstrations of s01 in recording order (a fixed base)."""
        self.demo_dir, self.episodes, self.workers = demo_dir, episodes, workers
        self.sessions, self.label_prefix = sessions, label_prefix
        self.cfg = cfg or HandBCConfig()
        self.state = "idle"            # idle | training | evaluating | done | error
        self.message = ""
        self.result: dict | None = None
        self.history: list[dict] = []
        self.t_started = 0.0
        self._thread: threading.Thread | None = None
        # for the learning panel: progress, a copy of the network every few hundred iterations,
        # and the new policy file as soon as it is saved (the screen lets it try a layout)
        self.progress = {"it": 0, "iters": 1, "loss": [], "eval_done": 0, "eval_total": episodes}
        self.live_weights: list | None = None
        self.pending_policy: Path | None = None

    def add_baseline(self, policy_path: Path) -> None:
        """Show the starting policy's own evaluation (from its JSON, if it was evaluated)."""
        j = Path(policy_path).with_suffix(".json")
        if j.exists():
            d = json.loads(j.read_text(encoding="utf-8"))
            self.history.append({"label": f"시범 {d.get('n_demos', '?')}", "success_rate": d["success_rate"],
                                 "cubes_mean": d["cubes_mean"], "episodes": d.get("episodes"),
                                 "policy": str(policy_path)})

    def _load(self) -> list:
        return load_session_demos(self.demo_dir, self.sessions)

    def data_counts(self) -> tuple[int, int]:
        demos = self._load()
        return sum(d.kind == "demo" for d in demos), sum(d.kind == "takeover" for d in demos)

    def busy(self) -> bool:
        return self.state in ("training", "evaluating")

    def start(self) -> bool:
        if self.busy():
            return False
        self.state, self.message, self.result = "training", "", None
        self.t_started = time.perf_counter()
        self.progress = {"it": 0, "iters": self.cfg.iters, "loss": [], "eval_done": 0, "eval_total": self.episodes}
        self.live_weights, self.pending_policy = None, None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _on_progress(self, it: int, iters: int, loss: float, net) -> None:
        self.progress["it"], self.progress["iters"] = it, iters
        self.progress["loss"].append(loss)
        if it % 250 == 0 or it == iters:                 # ~70k floats: cheap to copy now and then
            import torch
            with torch.no_grad():
                self.live_weights = [m.weight.detach().numpy().T.copy() for m in net
                                     if isinstance(m, torch.nn.Linear)]

    def _run(self) -> None:
        try:
            demos = self._load()
            if not demos:
                raise ValueError("배울 시범이 없습니다 (성공한 판이 필요해요)")
            n_demo = sum(d.kind == "demo" for d in demos)
            n_take = sum(d.kind == "takeover" for d in demos)
            self.message = f"시범 {n_demo}개 + 개입 {n_take}번으로 학습 중"
            pol = train_hand_bc(demos, self.cfg, progress=self._on_progress)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = POLICY_DIR / f"live_{stamp}_d{n_demo}_t{n_take}.npz"
            pol.save(path)
            self.live_weights = [np.asarray(w) for w, _ in pol.weights]
            self.pending_policy = path
            self.state = "evaluating"
            self.message = f"처음 보는 배치 {self.episodes}판으로 시험 중"
            res = evaluate_policy(path, episodes=self.episodes, workers=self.workers,
                                  on_episode=lambda k, n: self.progress.update(eval_done=k))
            group = self.sessions[-1].partition(":")[0] if self.sessions else None
            n_base = sum(d.kind == "demo" and d.meta["session"] != group for d in demos) if group else 0
            label = self.label_prefix + f"시범 {n_demo - n_base}" + (f" + 개입 {n_take}" if n_take else "")                 + (f" (+기본 {n_base})" if n_base else "")
            row = {"label": label, "n_base_demos": n_base, **res.summary(),
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
