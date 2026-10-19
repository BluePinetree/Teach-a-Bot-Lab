r"""Render real closed-loop rollouts of a policy to .mp4 (no window), for slides and the blog.

    .venv\Scripts\python.exe scripts\render_rollout_video.py --policy data\policies\X.npz --seeds 1000000 1000001 --out data\videos\bc10.mp4 --label "시범 10개로 배운 AI"
    .venv\Scripts\python.exe scripts\render_rollout_video.py --rl data\rl_runs\sparse_...\ckpt_N.zip --seeds 1000000 --out data\videos\rl.mp4 --label "강화학습 31시간"

Every frame is the simulation actually running the policy (same HUD as the app, 30 fps of
simulated time). The result card at the end of each episode is the real outcome.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.aloha.env import CONTROL_DT, AlohaSortEnv  # noqa: E402
from teachabot.aloha.gym_env import action_to_commands  # noqa: E402
from teachabot.aloha.scene import HANDS  # noqa: E402
from teachabot.aloha.teleop import TeleopMapper, TeleopSettings  # noqa: E402
from teachabot.aloha_app import H, W, Hud  # noqa: E402

FPS = 30


class RLPolicy:
    """A PPO checkpoint from scripts/train_rl.py, run exactly as in its evaluation:
    normalised observation, deterministic action held for `repeat` control steps."""

    def __init__(self, ckpt: Path, repeat: int = 5):
        from stable_baselines3 import PPO
        self.model = PPO.load(ckpt, device="cpu")
        vn_path = ckpt.with_name(ckpt.name.replace("ckpt_", "vecnorm_").replace(".zip", ".pkl"))
        if ckpt.name == "final.zip":
            vn_path = ckpt.with_name("vecnorm_final.pkl")
        with open(vn_path, "rb") as f:
            self.vecnorm = pickle.load(f)
        self.repeat, self.k, self.a = repeat, 0, None

    def reset(self):
        self.k, self.a = 0, None

    def __call__(self, obs):
        if self.k % self.repeat == 0:
            self.a, _ = self.model.predict(self.vecnorm.normalize_obs(obs[None].astype(np.float32)), deterministic=True)
            self.a = self.a[0]
        self.k += 1
        return self.a


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy", type=Path, help="BC policy .npz")
    g.add_argument("--rl", type=Path, help="PPO checkpoint .zip (vecnorm .pkl next to it)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True, help="text shown top right")
    ap.add_argument("--time-limit", type=float, default=40.0)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the result card")
    args = ap.parse_args()

    if args.policy:
        from teachabot.learning.bc import NumpyPolicy
        policy = NumpyPolicy.load(args.policy)
    else:
        policy = RLPolicy(args.rl)
    env = AlohaSortEnv(time_limit=args.time_limit)
    renderer = mujoco.Renderer(env.model, H, W)
    hud = Hud()
    mapper = TeleopMapper(TeleopSettings(wrist_source="off"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    steps_per_frame = int(round(1 / (FPS * CONTROL_DT)))   # 50 Hz control -> ~30 fps: alternate 2/1 steps
    results = []
    for n, seed in enumerate(args.seeds, 1):
        obs = env.reset(seed)
        if hasattr(policy, "reset"):
            policy.reset()
        info = env.info()
        t_next = 0.0
        while True:
            while env.t < t_next and not info.done:
                obs, info = env.step(action_to_commands(policy(obs)))
            t_next += 1.0 / FPS
            renderer.update_scene(env.data, camera="demo_cam")
            result = None
            if info.done:
                result = f"성공!  {info.time:.1f}초" if info.success else \
                    f"실패 (상자 안 {sum(info.in_tray.values())}/2)"
            out = hud.draw(renderer.render(), env, info, None, mapper, "AI", n, result, None, FPS,
                           mode_line=args.label, controllers={h: "ai" for h in HANDS},
                           guide_text="실제 물리 시뮬레이션에서 AI가 매 순간 직접 판단해 조종한 장면 (녹화된 동작 재생 아님)")
            frame = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            video.write(frame)
            if info.done:
                for _ in range(int(args.hold * FPS)):
                    video.write(frame)
                break
        results.append({"seed": seed, "success": bool(info.success), "time": round(info.time, 2),
                        "cubes": int(sum(info.in_tray.values()))})
        print(results[-1], flush=True)
    video.release()
    print("saved", args.out, "steps/frame", steps_per_frame)


if __name__ == "__main__":
    main()
