"""Phase 3 pre-check: does Ursina work from this (Korean) path, and at what frame rate?

Opens a 1280x720 window for a few seconds, then quits by itself and writes
data/logs/ursina_check_<mode>_<time>.json.

Checks
  - project path contains non-ASCII characters (reported)
  - a texture (PNG) and a model (OBJ) written under assets/_check/ load from that path
  - frame-time statistics for a scene of ~30 entities, optionally while polling
    JoyConInputProvider every frame, with vsync on (capped) or off (headroom)

    .venv\\Scripts\\python.exe scripts\\check_ursina_env.py --input joycon --vsync on
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--input", choices=["none", "joycon", "keyboard"], default="none")
ap.add_argument("--vsync", choices=["on", "off"], default="on")
ap.add_argument("--seconds", type=float, default=8.0)
ap.add_argument("--warmup", type=float, default=2.0)
args = ap.parse_args()

# -- test assets under the project path ---------------------------------------
check_dir = ROOT / "assets" / "_check"
check_dir.mkdir(parents=True, exist_ok=True)
from PIL import Image  # noqa: E402  (ursina dependency)

img = Image.new("RGB", (64, 64))
img.putdata([((x * 4) % 256, (y * 4) % 256, 128) for y in range(64) for x in range(64)])
img.save(check_dir / "check_tex.png")
v = [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5),
     (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]
faces = [(1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6), (1, 5, 6), (1, 6, 2),
         (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1)]
(check_dir / "check_cube.obj").write_text(
    "\n".join([f"v {a} {b} {c}" for a, b, c in v] + [f"f {a} {b} {c}" for a, b, c in faces]) + "\n",
    encoding="utf-8")

from ursina import (Entity, Ursina, application, camera, color, load_model,  # noqa: E402
                    load_texture, window)

app = Ursina(title="Teach-a-Bot env check", size=(1280, 720), borderless=False,
             vsync=(args.vsync == "on"), development_mode=False)
window.fps_counter.enabled = True

result = {"python": sys.version.split()[0], "project_path": str(ROOT),
          "path_has_non_ascii": any(ord(c) > 127 for c in str(ROOT)),
          "asset_folder": str(application.asset_folder),
          "input": args.input, "vsync": args.vsync}

errors = []
# The real app will point asset_folder at <project>/assets and load by name.
application.asset_folder = ROOT / "assets"
result["asset_folder"] = str(application.asset_folder)
tex = mdl = None
try:
    tex = load_texture("check_tex")
    result["texture_loaded"] = tex is not None
except Exception as e:
    result["texture_loaded"] = False
    errors.append(f"texture: {type(e).__name__}: {e}")
try:
    mdl = load_model("check_cube")
    result["obj_model_loaded"] = mdl is not None
except Exception as e:
    result["obj_model_loaded"] = False
    errors.append(f"obj model: {type(e).__name__}: {e}")
# Panda3D's own file loader (used for .bam, and by panda3d-gltf for .glb) on a non-ASCII path
try:
    import builtins
    from panda3d.core import Filename
    bam = check_dir / "check_cube.bam"
    ok_write = mdl is not None and mdl.writeBamFile(Filename.fromOsSpecific(str(bam)))
    loaded = builtins.loader.loadModel(Filename.fromOsSpecific(str(bam))) if ok_write else None
    result["bam_write_and_load"] = bool(ok_write and loaded is not None and not loaded.isEmpty())
except Exception as e:
    result["bam_write_and_load"] = False
    errors.append(f"bam: {type(e).__name__}: {e}")

# scene roughly the size of the Phase 3 table scene
Entity(model="cube", scale=(1.4, 0.05, 0.9), y=-0.025, color=color.light_gray)
for i in range(24):
    Entity(model="cube", scale=0.06, position=((i % 6) * 0.2 - 0.5, 0.03, (i // 6) * 0.2 - 0.3),
           color=color.azure if i % 2 else color.orange)
probe = Entity(model=mdl if result.get("obj_model_loaded") else "cube",
               texture=tex if result.get("texture_loaded") else None, scale=0.15, position=(0, 0.2, 0))
camera.position = (0, 1.1, -1.4)
camera.rotation_x = 38

provider = None
if args.input == "joycon":
    from teachabot.input.joycon import JoyConInputProvider
    provider = JoyConInputProvider()
elif args.input == "keyboard":
    from teachabot.input.keyboard import KeyboardInputProvider
    provider = KeyboardInputProvider()
if provider:
    provider.connect()

t_start = time.perf_counter()
last = [None]
frame_dt, poll_ms, connected_frames = [], [], [0]


def finish():
    if provider:
        result["provider_events"] = [{k: (round(v - t_start, 2) if k == "t" else v) for k, v in e.items()}
                                     for e in getattr(provider, "events", [])]
        result["device_errors"] = {h: d.last_error for h, d in getattr(provider, "_devices", {}).items()}
        result["reports_processed"] = {h: s.reports for h, s in getattr(provider, "_state", {}).items()}
        provider.disconnect()
    ts = sorted(frame_dt)
    q = lambda p: ts[min(len(ts) - 1, int(p * (len(ts) - 1)))]
    result.update({
        "frames_measured": len(ts),
        "fps_mean": round(len(ts) / sum(ts), 1) if ts else None,
        "frame_ms_median": round(1000 * statistics.median(ts), 2) if ts else None,
        "frame_ms_p95": round(1000 * q(0.95), 2) if ts else None,
        "frame_ms_p99": round(1000 * q(0.99), 2) if ts else None,
        "frame_ms_max": round(1000 * ts[-1], 2) if ts else None,
        "frames_over_20ms": sum(1 for d in ts if d > 0.020),
        "poll_ms_median": round(statistics.median(poll_ms), 3) if poll_ms else None,
        "poll_ms_max": round(max(poll_ms), 3) if poll_ms else None,
        "frames_with_both_hands_connected": connected_frames[0] if provider else None,
        "errors": errors,
    })
    out = ROOT / "data" / "logs" / f"ursina_check_{args.input}_vsync{args.vsync}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    application.quit()


def update():
    now = time.perf_counter()
    probe.rotation_y += 60 * (now - last[0]) if last[0] else 0
    if provider:
        t0 = time.perf_counter()
        f = provider.poll()
        if now - t_start > args.warmup:
            poll_ms.append(1000 * (time.perf_counter() - t0))
            connected_frames[0] += int(f.left.connected and f.right.connected)
    if last[0] is not None and now - t_start > args.warmup:
        frame_dt.append(now - last[0])
    last[0] = now
    if now - t_start > args.warmup + args.seconds:
        finish()


app.run()
