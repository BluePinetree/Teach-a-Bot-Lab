"""Analyze a webcam hand recording (tests/manual/record_webcam_hands.py).

Reports, per recording:
  - fps and detection rate (both hands found) per stage
  - position error in cm at the 10 held-out sheet points, using a homography
    fitted only on the 4 corner points (image -> table plane)
  - jitter (cm, std) while both hands are held still
  - camera latency vs the Joy-Con IMU: lag that best aligns hand speed (camera)
    with gyro magnitude (IMU) during free motion
  - L/R assignment: which camera hand track moves with which Joy-Con, compared
    with MediaPipe's own Left/Right label

Hands are followed as two tracks by nearest-neighbour association, so nothing
depends on MediaPipe's handedness label.

    .venv\\Scripts\\python.exe scripts\\analyze_webcam_hands.py data\\logs\\webcam_<cond>_....jsonl [--figdir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CORNERS = ["앞-왼쪽 모서리", "앞-오른쪽 모서리", "뒤-오른쪽 모서리", "뒤-왼쪽 모서리"]
TESTS = ["가운데", "앞 가장자리 가운데", "뒤 가장자리 가운데", "왼쪽 가장자리 가운데", "오른쪽 가장자리 가운데"]
PLAN = ([("cal", "right", p) for p in CORNERS] + [("test", "right", p) for p in TESTS]
        + [("test", "left", p) for p in TESTS] + [("still", "both", None), ("free", "both", None)])


def load(path):
    header, frames, joy = None, [], {"L": [], "R": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "header":
                header = r
            elif r["type"] == "frame":
                frames.append(r)
            elif r["type"] == "joycon":
                joy[r["side"]].append(r)
    return header, frames, joy


def track(frames):
    """Two tracks (0/1) by nearest-neighbour association of palm positions."""
    last = [None, None]
    for fr in frames:
        fr["tracks"] = {}
        pts = [np.asarray(h["palm_px"]) for h in fr["hands"]]
        if not pts:
            continue
        if last[0] is None and last[1] is None:
            order = sorted(range(len(pts)), key=lambda k: pts[k][0])
            for tid, k in enumerate(order[:2]):
                last[tid] = pts[k]
                fr["tracks"][tid] = fr["hands"][k]
            continue
        used = set()
        pairs = sorted(((np.linalg.norm(p - last[t]) if last[t] is not None else 1e9, k, t)
                        for k, p in enumerate(pts) for t in (0, 1)))
        for dist, k, t in pairs:
            if k in used or t in fr["tracks"]:
                continue
            fr["tracks"][t] = fr["hands"][k]
            last[t] = pts[k]
            used.add(k)


def analyze(path, figdir=None, name=None):
    header, frames, joy = load(path)
    name = name or header["condition"]
    pts_cm = {k: np.asarray(v, float) for k, v in header["points_cm"].items()}
    track(frames)
    t = np.array([f["t"] for f in frames])
    dt = np.diff(t)
    out = {"condition": name, "frames": len(frames), "fps": round(1 / np.median(dt), 1),
           "frame_ms_p95": round(float(np.percentile(dt, 95) * 1000), 1)}

    # Per hold window: split all palm detections into 2 clusters (k-means). Robust to
    # frame-to-frame identity swaps; the cluster that moved since the previous hold is
    # the hand that was asked to move.
    def clusters(points):
        P = np.asarray(points, np.float32)
        if len(P) < 4:
            return [np.median(P, axis=0)] if len(P) else []
        if np.ptp(P, axis=0).max() < 40:          # only one hand visible / one blob
            return [np.median(P, axis=0)]
        _, lab, _ = cv2.kmeans(P, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5,
                               cv2.KMEANS_PP_CENTERS)
        return [np.median(P[lab.ravel() == k], axis=0) for k in (0, 1)]

    holds = {}
    for i, (kind, hand, target) in enumerate(PLAN):
        fs = [f for f in frames if f["step"] == i and f["phase"] == "hold"]
        holds[i] = clusters([h["palm_px"] for f in fs for h in f["hands"]])
        if kind in ("still", "free"):
            n = len([f for f in frames if f["step"] == i and f["phase"] == "hold"])
            both = len([f for f in frames if f["step"] == i and f["phase"] == "hold" and len(f["hands"]) == 2])
            out[f"detect_both_{kind}"] = round(both / max(n, 1), 3)
    pre = [f for f in frames if f["step"] < 0][-30:]
    prev = clusters([h["palm_px"] for f in pre for h in f["hands"]])

    active = {}
    for i, (kind, hand, target) in enumerate(PLAN):
        if kind not in ("cal", "test") or not holds[i]:
            continue
        # displacement of each current cluster from its nearest previous cluster
        disp = [min(np.linalg.norm(c - q) for q in prev) if prev else 0.0 for c in holds[i]]
        active[i] = holds[i][int(np.argmax(disp))]
        prev = holds[i] if len(holds[i]) == 2 else prev
    out["held_points_found"] = f"{len(active)}/{sum(1 for k in PLAN if k[0] in ('cal', 'test'))}"

    cal_idx = [i for i in range(4) if i in active]
    if len(cal_idx) < 4:
        out["error"] = "corner points missing; cannot fit homography"
        return out, None
    src = np.array([active[i] for i in cal_idx], np.float32)
    dst = np.array([pts_cm[PLAN[i][2]] for i in cal_idx], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    to_cm = lambda p: cv2.perspectiveTransform(np.asarray(p, np.float32).reshape(-1, 1, 2), Hm).reshape(-1, 2)

    errs, rows = [], []
    for i, (kind, hand, target) in enumerate(PLAN):
        if kind != "test" or i not in active:
            continue
        est = to_cm(active[i])[0]
        e = float(np.linalg.norm(est - pts_cm[target]))
        errs.append(e)
        rows.append({"hand": hand, "target": target, "true": pts_cm[target].tolist(), "est": est.round(1).tolist(),
                     "err_cm": round(e, 1)})
    out["test_err_cm_median"] = round(float(np.median(errs)), 1) if errs else None
    out["test_err_cm_max"] = round(float(np.max(errs)), 1) if errs else None
    out["test_points"] = rows

    # jitter while still: points assigned to the nearest of the two cluster centres
    si = PLAN.index(("still", "both", None))
    still_pts = [h["palm_px"] for f in frames if f["step"] == si and f["phase"] == "hold" for h in f["hands"]]
    jit = {}
    if holds[si]:
        P = np.asarray(still_pts, float)
        lab = np.argmin([[np.linalg.norm(p - c) for c in holds[si]] for p in P], axis=1)
        for k in range(len(holds[si])):
            c = to_cm(P[lab == k])
            if len(c) > 10:
                jit[f"hand{k}"] = round(float(np.sqrt(np.mean(np.sum((c - c.mean(0)) ** 2, axis=1)))), 2)
    out["still_jitter_cm_rms"] = jit

    # free motion: speed vs Joy-Con gyro -> L/R assignment and latency
    fi = PLAN.index(("free", "both", None))
    free = [f for f in frames if f["step"] == fi]
    if free and joy["L"] and joy["R"]:
        tf = np.array([f["t"] for f in free])
        grid = np.arange(tf[0], tf[-1], 0.02)
        speed = {}
        for tid in (0, 1):
            tt = np.array([f["t"] for f in free if tid in f["tracks"]])
            if len(tt) < 20:
                continue
            c = to_cm([f["tracks"][tid]["palm_px"] for f in free if tid in f["tracks"]])
            v = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(tt), 1e-3)
            speed[tid] = np.interp(grid, tt[1:], np.convolve(v, np.ones(3) / 3, "same"))
        gyro = {}
        for side in ("L", "R"):
            tj = np.array([r["t"] for r in joy[side]])
            g = np.array([np.linalg.norm(r["gyro_dps"][0]) for r in joy[side]])
            gyro[side] = np.interp(grid, tj, g)
        z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
        corr = {(tid, s): float(np.mean(z(speed[tid]) * z(gyro[s]))) for tid in speed for s in gyro}
        out["speed_gyro_corr"] = {f"track{tid}-{s}": round(v, 2) for (tid, s), v in corr.items()}
        if len(speed) == 2:
            a = corr[(0, "L")] + corr[(1, "R")]
            b = corr[(0, "R")] + corr[(1, "L")]
            assign = {0: "L", 1: "R"} if a >= b else {0: "R", 1: "L"}
            out["imu_assignment"] = {f"track{k}": v for k, v in assign.items()}
            out["imu_assignment_margin"] = round(abs(a - b) / 2, 2)
            labels = {}
            for tid in (0, 1):
                ls = [f["tracks"][tid]["label"] for f in free if tid in f["tracks"]]
                labels[tid] = max(set(ls), key=ls.count) if ls else None
            out["mediapipe_label"] = {f"track{k}": v for k, v in labels.items()}
            # latency: lag maximizing correlation (positive = camera later than IMU)
            lags = np.arange(-15, 16)
            best = []
            for tid, s in assign.items():
                cs = [np.mean(z(np.roll(speed[tid], -L))[20:-20] * z(gyro[s])[20:-20]) for L in lags]
                best.append(int(lags[int(np.argmax(cs))]))
            out["camera_lag_ms_vs_imu"] = [b * 20 for b in best]

    fig = None
    if figdir and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130})
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.add_patch(plt.Rectangle((-14.85, -10.5), 29.7, 21, fill=False, color="#999999"))
        for r in rows:
            col = "#2f6fe0" if r["hand"] == "left" else "#f08a24"
            ax.plot(*r["true"], "s", color="k", ms=5)
            ax.plot(*r["est"], "o", color=col, ms=6)
            ax.plot([r["true"][0], r["est"][0]], [r["true"][1], r["est"][1]], color=col, lw=1)
        for k in CORNERS:
            ax.plot(*pts_cm[k], "^", color="#3a9d5d", ms=7)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="s", color="k", ls="", label="실제 점"),
                           Line2D([], [], marker="o", color="#f08a24", label="오른손 추정"),
                           Line2D([], [], marker="o", color="#2f6fe0", label="왼손 추정"),
                           Line2D([], [], marker="^", color="#3a9d5d", ls="", label="보정 모서리(오른손)")],
                  fontsize=7, loc="lower right")
        ax.set_aspect("equal")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-15, 15)
        ax.set_xlabel("오른쪽 [cm]")
        ax.set_ylabel("앞 [cm]")
        ax.set_title(f"웹캠 손 위치 ({name}): 보정 밖 점 오차 중앙값 {out['test_err_cm_median']} cm", fontsize=10)
        ax.grid(alpha=.25)
        fig.tight_layout()
        Path(figdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(figdir) / f"fig11_webcam_points_{name}.png", bbox_inches="tight")
        plt.close(fig)
    return out, fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--figdir", type=Path)
    ap.add_argument("--names", nargs="+")
    args = ap.parse_args()
    results = []
    for i, p in enumerate(args.files):
        r, _ = analyze(p, args.figdir, args.names[i] if args.names else None)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "test_points"}, ensure_ascii=False, indent=1))
    out = args.files[0].with_name("webcam_analysis.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("details:", out)


if __name__ == "__main__":
    main()
