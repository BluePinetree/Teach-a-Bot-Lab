"""Offline evaluation of plan A: can a single Joy-Con IMU tell how far and which way the hand moved?

Input: recordings from tests/manual/record_motion_strokes.py (raw reports, all 3
IMU sub-samples, trial timing + labels). For every trial the hand was still
before and after the stroke, and the true direction and distance are known.

Methods (all per trial, in the Joy-Con sensor frame):
  naive : subtract gravity measured just before the stroke, integrate twice
          over the whole move+hold window.
  zupt  : same, but integrate only over the detected motion interval and remove
          the velocity drift linearly so velocity is 0 when the hand is still
          again (zero-velocity update) — the standard pedestrian-INS trick.

World directions are NOT assumed: forward/right unit vectors in the sensor
frame are learned from the other trials (leave-one-out), and so is an optional
scale gain, so every reported number is a held-out prediction.

    .venv\\Scripts\\python.exe scripts\\analyze_strokes.py data\\logs\\strokes_desk_R_....jsonl [more files]
        [--figdir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

G = 9.81
DT = 0.005          # IMU sub-sample spacing
TRUE_VEC = {"forward": (0, 1), "back": (0, -1), "right": (1, 0), "left": (-1, 0)}


def load(path: Path):
    reps, phases, bad, header = [], [], set(), None
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = r.get("type")
            if k == "report":
                reps.append(r)
            elif k == "phase":
                phases.append(r)
            elif k == "bad":
                bad.add(r["trial"])
            elif k == "header":
                header = r
    # 200 Hz stream, oldest -> newest (sub-sample index 0 is the newest)
    t, acc, gyr = [], [], []
    for r in reps:
        for pos, k in enumerate((2, 1, 0)):
            t.append(r["t"] - (2 - pos) * DT)
            acc.append(r["accel_g"][k])
            gyr.append(r["gyro_dps"][k])
    t, acc, gyr = np.asarray(t), np.asarray(acc) * G, np.asarray(gyr)
    trials = {}
    for p in phases:
        tr = trials.setdefault(p["trial"], {"direction": p["direction"], "cm": p["distance_cm"]})
        tr[p["phase"]] = p["t"]
    return header, t, acc, gyr, trials, bad


def integrate(t, acc, gyr, tr, method: str, hold_s: float = 1.5):
    """Displacement vector (m, sensor frame) and diagnostics for one trial."""
    ready = (t >= tr["move"] - 1.0) & (t < tr["move"] - 0.05)
    g_ref = acc[ready].mean(axis=0)
    win = (t >= tr["move"] - 0.3) & (t < tr["hold"] + hold_s)
    tw, lin, w = t[win], acc[win] - g_ref, gyr[win]
    rot_deg = float(np.sum(np.linalg.norm(w, axis=1)) * DT)       # total rotation during the window
    if method == "naive":
        v = np.cumsum(lin, axis=0) * DT
        p = np.cumsum(v, axis=0) * DT
        return p[-1], {"rot_deg": rot_deg, "t": tw, "v": v, "p": p}
    if method == "zupt2":
        # gravity measured before AND after the stroke; a few degrees of tilt while
        # pushing otherwise leaks 0.4-0.9 m/s^2 of gravity into the "motion"
        a = acc[win]
        post = (t >= tr["hold"] + hold_s - 0.8) & (t < tr["hold"] + hold_s)
        g_post = acc[post].mean(axis=0)
        sm = lambda x: np.convolve(x, np.ones(10) / 10, "same")
        on = np.where(sm(np.linalg.norm(a - g_ref, axis=1)) > 0.25)[0]
        off = np.where(sm(np.linalg.norm(a - g_post, axis=1)) > 0.25)[0]
        if len(on) == 0 or len(off) == 0:
            return np.zeros(3), {"rot_deg": rot_deg, "t": tw, "v": np.zeros_like(a), "p": np.zeros_like(a)}
        s, e = max(on[0] - 10, 0), min(off[-1] + 10, len(tw) - 1)
        if e <= s:
            e = min(s + 20, len(tw) - 1)
        frac = np.clip((np.arange(len(tw)) - s) / max(e - s, 1), 0, 1)[:, None]
        g_t = g_ref * (1 - frac) + g_post * frac
        lin2 = a - g_t
        v = np.zeros_like(lin2)
        v[s:e + 1] = np.cumsum(lin2[s:e + 1], axis=0) * DT
        v[s:e + 1] -= np.linspace(0, 1, e - s + 1)[:, None] * v[e]
        p = np.cumsum(v, axis=0) * DT
        return p[-1], {"rot_deg": rot_deg, "t": tw, "v": v, "p": p, "s": s, "e": e,
                       "duration_s": float((e - s) * DT)}
    # motion interval: smoothed |linear accel| above threshold, with margins
    mag = np.convolve(np.linalg.norm(lin, axis=1), np.ones(10) / 10, "same")
    moving = np.where(mag > 0.35)[0]
    if len(moving) == 0:
        return np.zeros(3), {"rot_deg": rot_deg, "t": tw, "v": np.zeros_like(lin), "p": np.zeros_like(lin)}
    s, e = max(moving[0] - 10, 0), min(moving[-1] + 10, len(tw) - 1)
    v = np.zeros_like(lin)
    v[s:e + 1] = np.cumsum(lin[s:e + 1], axis=0) * DT
    ramp = np.linspace(0, 1, e - s + 1)[:, None]
    v[s:e + 1] -= ramp * v[e]                                       # ZUPT: v(e) must be 0
    p = np.cumsum(v, axis=0) * DT
    return p[-1], {"rot_deg": rot_deg, "t": tw, "v": v, "p": p, "s": s, "e": e,
                   "duration_s": float((e - s) * DT)}


def evaluate(files, method, names=None, pooled=False):
    rows = []
    for n, path in enumerate(files):
        header, t, acc, gyr, trials, bad = load(path)
        if names:
            header = {**header, "surface": names[n]}
        for i, tr in sorted(trials.items()):
            if not {"ready", "move", "hold"} <= tr.keys():
                continue
            d, diag = integrate(t, acc, gyr, tr, method)
            rows.append({"file": path.name, "surface": header["surface"], "trial": i, "bad": i in bad,
                         "direction": tr["direction"], "cm": tr["cm"], "d": d, "rot_deg": diag["rot_deg"],
                         "duration_s": diag.get("duration_s")})
    if pooled:
        rows = [{**r, "surface": "+".join(names or ["all"])} for r in rows]
    out = []
    for surface in sorted({r["surface"] for r in rows}):
        rs = [r for r in rows if r["surface"] == surface and not r["bad"]]
        for k, r in enumerate(rs):
            train = [q for j, q in enumerate(rs) if j != k]
            fwd = sum(np.asarray(TRUE_VEC[q["direction"]][1]) * q["d"] for q in train if q["direction"] in ("forward", "back"))
            rgt = sum(np.asarray(TRUE_VEC[q["direction"]][0]) * q["d"] for q in train if q["direction"] in ("right", "left"))
            f = fwd / np.linalg.norm(fwd)
            rr = rgt - np.dot(rgt, f) * f
            rr /= np.linalg.norm(rr)
            proj = lambda q: np.array([np.dot(q["d"], rr), np.dot(q["d"], f)]) * 100     # cm (x=right, z=forward)
            true = lambda q: np.array(TRUE_VEC[q["direction"]]) * q["cm"]
            P = np.array([proj(q) for q in train])
            T = np.array([true(q) for q in train])
            gain = float(np.sum(P * T) / np.sum(P * P))                                   # least-squares scale
            pred_raw, tru = proj(r), true(r)
            pred = pred_raw * gain
            ang = math.degrees(math.atan2(pred[0] * tru[1] - pred[1] * tru[0], np.dot(pred, tru)))
            out.append({**{k2: v for k2, v in r.items() if k2 != "d"}, "method": method,
                        "pred_raw_cm": pred_raw.round(2).tolist(), "pred_cm": pred.round(2).tolist(),
                        "true_cm": tru.tolist(), "gain": round(gain, 3),
                        "err_cm": round(float(np.linalg.norm(pred - tru)), 2),
                        "dist_err_cm": round(float(np.linalg.norm(pred) - r["cm"]), 2),
                        "angle_err_deg": round(ang, 1)})
    return out


def summarize(results):
    summ = {}
    for key in sorted({(r["surface"], r["method"]) for r in results}):
        rs = [r for r in results if (r["surface"], r["method"]) == key]
        err = np.array([r["err_cm"] for r in rs])
        ang = np.abs([r["angle_err_deg"] for r in rs])
        summ[f"{key[0]}/{key[1]}"] = {
            "trials": len(rs),
            "vector_err_cm_median": round(float(np.median(err)), 1),
            "vector_err_cm_p90": round(float(np.percentile(err, 90)), 1),
            "within_5cm": int((err <= 5).sum()),
            "angle_err_deg_median": round(float(np.median(ang)), 1),
            "direction_correct(<45deg)": int((ang < 45).sum()),
            "gain_median": round(float(np.median([r["gain"] for r in rs])), 3),
            "rotation_deg_median": round(float(np.median([r["rot_deg"] for r in rs])), 1),
        }
    return summ


def figures(files, results, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
    figdir.mkdir(parents=True, exist_ok=True)
    # (a) one trial: velocity before/after ZUPT
    header, t, acc, gyr, trials, bad = load(files[0])
    tr = next(v for v in trials.values() if v["direction"] == "forward" and v["cm"] == 20)
    _, dn = integrate(t, acc, gyr, tr, "naive")
    _, dz = integrate(t, acc, gyr, tr, "zupt2")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    tt = dn["t"] - dn["t"][0]
    axes[0].plot(tt, np.linalg.norm(dn["v"], axis=1), color="#c0392b", label="그냥 적분")
    axes[0].plot(tt, np.linalg.norm(dz["v"], axis=1), color="#2f6fe0", lw=2, label="ZUPT + 전후 중력 보간")
    axes[0].set_title("속도 크기 [m/s] — 멈춘 뒤 0이어야 한다")
    axes[0].set_xlabel("시간 [s]")
    axes[0].legend(fontsize=8)
    axes[1].plot(tt, 100 * np.linalg.norm(dn["p"], axis=1), color="#c0392b", label="그냥 적분")
    axes[1].plot(tt, 100 * np.linalg.norm(dz["p"], axis=1), color="#2f6fe0", lw=2, label="ZUPT + 전후 중력 보간")
    axes[1].axhline(20, color="k", ls="--", lw=1)
    axes[1].text(tt[-1], 20.5, "정답 20cm", ha="right", fontsize=8)
    axes[1].set_title("이동 거리 [cm]")
    axes[1].set_xlabel("시간 [s]")
    fig.suptitle("한 번의 스트로크 (공중 세션 1, 앞으로 20cm)", y=1.03)
    fig.tight_layout()
    fig.savefig(figdir / "fig09_stroke_zupt_example.png", bbox_inches="tight")
    plt.close(fig)
    # (b) predicted vs true, per surface, zupt
    surfaces = sorted({r["surface"] for r in results}, key=lambda x: ("+" in x, x))
    fig, axes = plt.subplots(1, len(surfaces), figsize=(4.6 * len(surfaces), 4.4), squeeze=False)
    for ax, surface in zip(axes[0], surfaces):
        rs = [r for r in results if r["surface"] == surface and r["method"] == "zupt2"]
        for r in rs:
            tx, tz = r["true_cm"]
            px, pz = r["pred_cm"]
            ax.plot([tx, px], [tz, pz], color="#999999", lw=0.8)
            ax.plot(tx, tz, "s", color="k", ms=5)
            ax.plot(px, pz, "o", color="#2f6fe0" if r["err_cm"] <= 5 else "#c0392b", ms=5)
        ax.add_patch(plt.Circle((0, 0), 5, fill=False, ls=":", color="#999999"))
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect("equal")
        ok = sum(r["err_cm"] <= 5 for r in rs)
        ax.set_title(f"{surface}: 5cm 이내 {ok}/{len(rs)}")
        ax.set_xlabel("오른쪽 [cm]")
        ax.set_ylabel("앞 [cm]")
    fig.suptitle("■ 실제로 옮긴 위치  ● IMU로 추정(파랑: 오차 5cm 이내, 빨강: 초과) — 다른 시행으로 학습한 값만 사용", y=1.02)
    fig.tight_layout()
    fig.savefig(figdir / "fig10_stroke_pred_vs_true.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--figdir", type=Path)
    ap.add_argument("--names", nargs="+", help="session names (override the recorded surface label)")
    args = ap.parse_args()
    results = []
    for m in ("naive", "zupt", "zupt2"):
        results += evaluate(args.files, m, args.names)
        if len(args.files) > 1:
            results += evaluate(args.files, m, args.names, pooled=True)
    summ = summarize(results)
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    out = args.files[0].with_name("strokes_analysis.json")
    out.write_text(json.dumps({"summary": summ, "trials": results}, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print("details:", out)
    if args.figdir:
        figures(args.files, results, args.figdir)
        print("figures:", args.figdir)


if __name__ == "__main__":
    main()
