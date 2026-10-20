r"""Write config/visit_reference.json (reference lines on the visit result screen) from the
measurement files, so the numbers shown to students are never typed by hand.

    .venv\Scripts\python.exe scripts\make_visit_reference.py --rl-steps 11001856   # the visit choice (306 h)

Sources: newest data/policies/compare_*.csv (scripts/compare_datasets.py) and, for each RL run,
the row of data/rl_runs/<reward>_*/eval_test.csv (scripts/monitor_rl.py, 40 held-out layouts,
same test as the BC numbers) at --rl-steps; without --rl-steps the newest evaluated checkpoint.
"""
from __future__ import annotations

import csv
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "visit_reference.json"
RL_LABEL = {"sparse": "강화학습 (성공해야만 점수)", "dense": "강화학습 (단계별 점수)"}


def rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-steps", type=int, help="RL checkpoint (steps) to quote; default: newest evaluated")
    args = ap.parse_args()
    lines = []
    compares = sorted(glob.glob(str(ROOT / "data" / "policies" / "compare_*.csv")))
    if compares:
        r = rows(compares[-1])
        for name in dict.fromkeys(x["dataset"] for x in r):
            sel = [x for x in r if x["dataset"] == name]
            sr = statistics.mean(float(x["success_rate"]) for x in sel)
            cb = statistics.mean(float(x["cubes_mean"]) for x in sel)
            n_d, n_t = sel[0]["n_demos"], int(sel[0]["n_takeovers"])
            who = f"연구실에서 가르친 AI (시범 {n_d}" + (f" + 개입 {n_t}번)" if n_t else ")")
            lines.append({"label": who, "value": f"성공 {sr * 100:.0f}% · 판당 큐브 {cb:.2f}개",
                          "source": f"{Path(compares[-1]).name}: 처음 보는 배치 {sel[0]['episodes']}판 × 학습 {len(sel)}번 평균"})
    for reward in ("sparse", "dense"):
        runs = sorted(glob.glob(str(ROOT / "data" / "rl_runs" / f"{reward}_2*")))
        if not runs:
            continue
        table = Path(runs[-1]) / "eval_test.csv"
        if not table.exists():
            continue
        ev = rows(str(table))
        if args.rl_steps is not None:
            ev = [x for x in ev if int(x["steps"]) == args.rl_steps]
            if not ev:
                raise SystemExit(f"{table}: no evaluation at {args.rl_steps} steps")
        e = ev[-1]
        lines.append({"label": f"{RL_LABEL[reward]} {float(e['practice_hours']):.0f}시간 연습 (시범 없이)",
                      "value": f"성공 {float(e['success_rate']) * 100:.0f}% · 판당 큐브 {float(e['cubes_mean']):.2f}개"
                               f" · 집어 든 손 {float(e['lifted_mean']):.2f}개",
                      "source": f"{Path(runs[-1]).name}/eval_test.csv @ {e['steps']} steps "
                                f"(처음 보는 배치 {e['episodes']}판)"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"lines": lines}, ensure_ascii=False, indent=2), encoding="utf-8")
    for x in lines:
        print(f"{x['label']}: {x['value']}   [{x['source']}]")


if __name__ == "__main__":
    main()
