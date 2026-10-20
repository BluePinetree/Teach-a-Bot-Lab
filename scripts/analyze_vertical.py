"""Offline check of up/down motion height (D18) on real Joy-Con recordings.

Replays a recording from tests/manual/record_motion_strokes.py through the same
VerticalMotionEstimator the app uses (causal, sample by sample, gyro bias taken
from the still baseline) and integrates its output into a height track. Per
trial it compares the height change (move start -> end of hold) with the label:
+cm for up, -cm for down, 0 for tilt-only and for horizontal strokes (desk/air
files), where any change is a false height.

    .venv\\Scripts\\python.exe scripts\\analyze_vertical.py data\\logs\\strokes_vertical_R_....jsonl [more]
        [--fig PATH] [--json PATH]

The suggested gain is the least-squares factor true/estimated over the up/down
trials; the app's own clip (MAX_VZ) and height limits are not modelled here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.input.processing import VerticalMotionEstimator  # noqa: E402

DT = 0.005
SIGN = {"up": 1, "up_slow": 1, "down": -1, "down_slow": -1}


def load(path: Path):
    reps, phases, bad, baseline = [], [], set(), None
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        k = r.get("type")
        if k == "report":
            reps.append(r)
        elif k == "phase":
            phases.append(r)
        elif k == "bad":
            bad.add(r["trial"])
        elif k == "baseline":
            baseline = r
    t, acc, gyr = [], [], []
    for r in reps:
        for pos, k in enumerate((2, 1, 0)):          # sub-sample 0 is the newest
            t.append(r["t"] - (2 - pos) * DT)
            acc.append(r["accel_g"][k])
            gyr.append(r["gyro_dps"][k])
    t, acc, gyr = np.asarray(t), np.asarray(acc), np.asarray(gyr)
    trials = {}
    for p in phases:
        tr = trials.setdefault(p["trial"], {"direction": p["direction"], "cm": p["distance_cm"]})
        tr[p["phase"]] = p["t"]
    return t, acc, gyr, trials, bad, baseline


def height_track(t, acc, gyr, baseline, **est_kw):
    still = (t >= baseline["t_start"]) & (t < baseline["t_end"]) if baseline else slice(0, 400)
    bias = gyr[still].mean(axis=0)
    est = VerticalMotionEstimator(dt=DT, **est_kw)
    v = np.empty(len(t))
    for i in range(len(t)):
        est.update(acc[i], gyr[i] - bias)
        v[i] = est.output
    return v, np.cumsum(v) * DT


def analyze(path: Path, hold_s: float = 1.5, **est_kw):
    t, acc, gyr, trials, bad, baseline = load(path)
    v, h = height_track(t, acc, gyr, baseline, **est_kw)
    rows = []
    for i, tr in sorted(trials.items()):
        if "move" not in tr or "hold" not in tr:
            continue
        a = np.searchsorted(t, tr["move"] - 0.1)
        b = min(np.searchsorted(t, tr["hold"] + hold_s), len(t) - 1)
        true = SIGN.get(tr["direction"], 0) * tr["cm"]
        rows.append({"trial": i, "direction": tr["direction"], "true_cm": true,
                     "est_cm": round(float(h[b] - h[a]) * 100, 1), "bad": i in bad})
    return rows, (t, v, h, trials)


def summarize(rows):
    ok = [r for r in rows if not r["bad"]]
    ud = [r for r in ok if r["true_cm"] != 0]
    zero = [r for r in ok if r["true_cm"] == 0]
    out = {"trials": len(rows), "flagged_bad": len(rows) - len(ok)}
    if ud:
        tr = np.array([r["true_cm"] for r in ud], float)
        es = np.array([r["est_cm"] for r in ud], float)
        gain = float(tr @ es / max(es @ es, 1e-9))
        out.update(sign_correct=f"{int(np.sum(np.sign(es) == np.sign(tr)))}/{len(ud)}",
                   ratio_median=round(float(np.median(es / tr)), 2), suggested_gain=round(gain, 2),
                   abs_err_cm_gain1=round(float(np.median(np.abs(es - tr))), 1),
                   abs_err_cm_suggested=round(float(np.median(np.abs(gain * es - tr))), 1))
        for d in ("up", "down", "up_slow", "down_slow"):
            sel = [r for r in ud if r["direction"] == d]
            if sel:
                out[f"ratio_{d}"] = round(float(np.median([r["est_cm"] / r["true_cm"] for r in sel])), 2)
    if zero:
        z = np.abs([r["est_cm"] for r in zero])
        out.update(false_height_cm_median=round(float(np.median(z)), 1), false_height_cm_max=round(float(z.max()), 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--fig", type=Path, help="height track of the first file with trial labels")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    report = {}
    for i, path in enumerate(args.logs):
        rows, (t, v, h, trials) = analyze(path)
        report[path.name] = {"summary": summarize(rows), "trials": rows}
        print(f"\n== {path.name}")
        for r in rows:
            print(f"  {r['trial']:2d} {r['direction']:10s} true {r['true_cm']:+5.0f} cm   est {r['est_cm']:+6.1f} cm"
                  + ("   (flagged)" if r["bad"] else ""))
        print("  ", json.dumps(report[path.name]["summary"], ensure_ascii=False))
        if args.fig and i == 0:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 3.6))
            ax.plot(t - t[0], h * 100, lw=1.2, color="#2b6cb0", label="estimated height (cm)")
            for tr in trials.values():
                if "move" in tr:
                    ax.axvspan(tr["move"] - t[0], tr["move"] - t[0] + 2.5, color="#f6ad55", alpha=0.18, lw=0)
            ax.set_xlabel("time (s)  - orange: move window")
            ax.set_ylabel("cm")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right")
            fig.tight_layout()
            args.fig.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.fig, dpi=130)
            print("saved", args.fig)
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
