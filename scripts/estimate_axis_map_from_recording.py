"""Re-estimate the Right->Left Joy-Con IMU axis map offline from an input recording.

Slides an 8 s window over the frames where both hands were connected and runs
estimate_axis_map on each. Windows where the two Joy-Cons were held together
give a verified map with low residual; others are rejected. Prints every window
and, with --save, writes the map only if all verified windows agree.

    .venv\\Scripts\\python.exe scripts\\estimate_axis_map_from_recording.py data\\logs\\input_joycon_....jsonl [--save]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.input.joycon import DEFAULT_AXIS_MAP_PATH  # noqa: E402
from teachabot.input.processing import AxisMap, estimate_axis_map, save_axis_map  # noqa: E402
from teachabot.input.replay import load_recording  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("recording", type=Path)
ap.add_argument("--window", type=float, default=8.0)
ap.add_argument("--step", type=float, default=2.0)
ap.add_argument("--save", action="store_true")
args = ap.parse_args()

_, frames = load_recording(args.recording)
both = [f for f in frames if f.left.connected and f.right.connected
        and f.left.accel_sensor and f.right.accel_sensor]
t0 = both[0].timestamp
verified = []
t = both[0].timestamp
while t + args.window <= both[-1].timestamp:
    win = [f for f in both if t <= f.timestamp < t + args.window]
    m = estimate_axis_map([f.left.accel_sensor for f in win], [f.right.accel_sensor for f in win])
    d = m.detail or {}
    tag = f"OK signs={m.signs} rms={d.get('rms_g')}" if m.verified else f"-- {d.get('reason', '')}"
    print(f"{t - t0:7.1f}-{t - t0 + args.window:6.1f}s  n={len(win):4d}  std={d.get('left_axis_std_g')}  {tag}")
    if m.verified:
        verified.append(m)
    t += args.step

votes = Counter(m.signs for m in verified)
print(f"\nverified windows: {len(verified)}  sign votes: {dict(votes)}")
if args.save:
    if len(votes) == 1:
        best = min(verified, key=lambda m: m.detail["rms_g"])
        detail = {**best.detail, "source": str(args.recording), "verified_windows": len(verified)}
        save_axis_map(DEFAULT_AXIS_MAP_PATH, AxisMap(signs=best.signs, verified=True, detail=detail))
        print(f"saved {best.signs} -> {DEFAULT_AXIS_MAP_PATH}")
    else:
        print("not saved: no verified window or windows disagree")
