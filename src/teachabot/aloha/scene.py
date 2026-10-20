"""Scenes on the MuJoCo Menagerie ALOHA 2 workcell: S1 (two cubes, two trays) and S2 (one tray to
carry with both arms, see _add_s2_objects).

World frame (Menagerie): x = operator's right, y = away from the operator, z = up,
table top at z = 0. The left arm base sits at x = -0.469, the right at x = +0.469;
the operator camera looks along +y from y = -1.35.

Task (S1 "정리하기"): the left arm puts the blue cube into the blue tray (left side),
the right arm puts the orange cube into the orange tray (right side).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[3]
ALOHA_DIR = ROOT / "assets" / "third_party" / "mujoco_menagerie" / "aloha"
SCENE_XML = ALOHA_DIR / "scene.xml"

HANDS = ("left", "right")
ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")

CUBE_HALF = 0.02           # 4 cm cube; the ALOHA gripper opens to ~7 cm
CUBE_MASS = 0.05
TRAY_INNER = 0.13          # inner width of the tray (square)
TRAY_WALL_H = 0.045
TRAY_WALL_T = 0.008

BLUE = (0.20, 0.45, 0.95, 1.0)
ORANGE = (1.00, 0.55, 0.10, 1.0)


@dataclass(frozen=True)
class TaskLayout:
    """Where things go. Cubes are randomised per episode inside their spawn boxes."""
    tray_pos: dict                       # hand -> (x, y) centre of the tray
    cube_spawn: dict                     # hand -> ((xmin, xmax), (ymin, ymax))
    cube_yaw_range: float = 0.5          # rad, uniform in [-r, r]


S1_LAYOUT = TaskLayout(
    tray_pos={"left": (-0.22, 0.20), "right": (0.22, 0.20)},
    cube_spawn={"left": ((-0.26, -0.13), (-0.16, 0.04)), "right": ((0.13, 0.26), (-0.16, 0.04))},
)

CUBE_OF = {"left": "blue_cube", "right": "orange_cube"}
TRAY_OF = {"left": "blue_tray", "right": "orange_tray"}
COLOR_OF = {"left": BLUE, "right": ORANGE}


def _add_tray(world: mujoco.MjsBody, name: str, pos, rgba) -> None:
    body = world.add_body(name=name, pos=[pos[0], pos[1], 0.0])
    h, t, w = TRAY_WALL_H, TRAY_WALL_T, TRAY_INNER / 2
    faded = [rgba[0], rgba[1], rgba[2], 1.0]
    # floor (thin, slightly above the table to avoid z-fighting)
    body.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[w + t, w + t, 0.002], pos=[0, 0, 0.002], rgba=faded)
    for i, (dx, dy, sx, sy) in enumerate([(w + t / 2, 0, t / 2, w + t), (-(w + t / 2), 0, t / 2, w + t),
                                          (0, w + t / 2, w, t / 2), (0, -(w + t / 2), w, t / 2)]):
        body.add_geom(name=f"{name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[sx, sy, h / 2], pos=[dx, dy, h / 2], rgba=faded)


def _add_cube(world: mujoco.MjsBody, name: str, rgba) -> None:
    body = world.add_body(name=name, pos=[0, 0, CUBE_HALF])
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  mass=CUBE_MASS, rgba=list(rgba), friction=[1.5, 0.01, 0.001], condim=4)


def _menagerie_assets() -> dict[str, bytes]:
    """All model files as an in-memory file system. MuJoCo cannot open files under a
    non-ASCII path on Windows (this project lives under "잡동사니"), so the XML and
    meshes are passed as bytes instead of paths."""
    return {p.relative_to(ALOHA_DIR).as_posix(): p.read_bytes()
            for p in ALOHA_DIR.rglob("*") if p.is_file() and p.suffix in (".xml", ".stl", ".obj", ".png")}


_INCLUDE = re.compile(r'<include\s+file="([^"]+)"\s*/>')
_MUJOCO_TAG = re.compile(r"^\s*<mujoco[^>]*>|</mujoco>\s*$", re.S)


def _inline_includes(xml: str, assets: dict[str, bytes]) -> str:
    """Replace <include file=.../> with the file's body (MjSpec.from_string does not
    resolve includes from the in-memory assets)."""
    def body(match: re.Match) -> str:
        inner = assets[match.group(1)].decode("utf-8")
        return _inline_includes(_MUJOCO_TAG.sub("", inner.strip()), assets)
    return _INCLUDE.sub(body, xml)


# -- S2 "양팔 트레이 들기": one tray with a handle at each end, a ball riding on it, a goal mat ----
# Option B (user, 2026-10-20): a ball and a low rim, so carrying smoothly and lifting together matter.
S2_PLATE_HALF = (0.15, 0.08, 0.006)      # 30 x 16 x 1.2 cm
S2_RIM_H, S2_RIM_T = 0.006, 0.006       # 0.6 cm: a jerky carry throws the ball off, a smooth one keeps it (D17)
S2_HANDLE_HALF = (0.015, 0.0125, 0.025)  # 3 x 2.5 x 5 cm, gripped across its 2.5 cm (y)
S2_HANDLE_X = 0.13                       # handle centres at x = +-13 cm from the tray centre
S2_TRAY_MASS, S2_BALL_MASS = 0.25, 0.05
S2_BALL_R = 0.02
S2_GOAL = (0.0, 0.22)                    # goal mat centre (visual only)
S2_GOAL_HALF = (0.07, 0.06)              # tray centre must end inside this
S2_TRAY_SPAWN = ((-0.03, 0.03), (-0.10, 0.0))
TRAY_YELLOW = (1.0, 0.85, 0.25, 1.0)


def _add_s2_objects(world: mujoco.MjsBody) -> None:
    px, py, pz = S2_PLATE_HALF
    tray = world.add_body(name="s2_tray", pos=[0, -0.05, pz + 0.001])
    tray.add_freejoint(name="s2_tray_joint")
    grey = [0.75, 0.77, 0.80, 1.0]
    plate_mass = S2_TRAY_MASS - 2 * 0.025
    tray.add_geom(name="s2_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[px, py, pz], mass=plate_mass, rgba=grey)
    rz = pz + S2_RIM_H / 2
    for i, (dx, dy, sx, sy) in enumerate([(px - S2_RIM_T / 2, 0, S2_RIM_T / 2, py), (-(px - S2_RIM_T / 2), 0, S2_RIM_T / 2, py),
                                          (0, py - S2_RIM_T / 2, px, S2_RIM_T / 2), (0, -(py - S2_RIM_T / 2), px, S2_RIM_T / 2)]):
        tray.add_geom(name=f"s2_rim{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, S2_RIM_H / 2],
                      pos=[dx, dy, rz], mass=0.005, rgba=grey)
    hx, hy, hz = S2_HANDLE_HALF
    for hand, sign in (("left", -1), ("right", 1)):
        r, g, b, _ = COLOR_OF[hand]
        tray.add_geom(name=f"s2_handle_{hand}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                      pos=[sign * S2_HANDLE_X, 0, pz + hz], mass=0.025, rgba=[r, g, b, 1.0], friction=[1.5, 0.01, 0.001],
                      condim=4)
        tray.add_site(name=f"s2_handle_{hand}", pos=[sign * S2_HANDLE_X, 0, pz + hz])
    ball = world.add_body(name="s2_ball", pos=[0, -0.05, 2 * pz + S2_BALL_R + 0.002])
    ball.add_freejoint(name="s2_ball_joint")
    ball.add_geom(name="s2_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[S2_BALL_R, 0, 0], mass=S2_BALL_MASS,
                  rgba=list(TRAY_YELLOW), friction=[0.5, 0.005, 0.0001])
    gx, gy = S2_GOAL_HALF
    goal = world.add_body(name="s2_goal", pos=[S2_GOAL[0], S2_GOAL[1], 0.0006])
    goal.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[gx + px, gy + py, 0.0005], contype=0, conaffinity=0,
                  rgba=[1.0, 0.85, 0.25, 0.35], group=2)


def build_spec(layout: TaskLayout = S1_LAYOUT, with_task_objects: bool = True, task: str = "s1") -> mujoco.MjSpec:
    """Menagerie scene + task objects. `with_task_objects=False` gives the robot-only
    model used for inverse kinematics."""
    assets = _menagerie_assets()
    xml = _inline_includes(assets["scene.xml"].decode("utf-8"), assets)
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    world = spec.worldbody
    # 2560x1440: the 3D view is re-rendered at window size on 4K screens (teachabot.display)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 2560, 1440
    # A key light with shadows from the operator's side, so depth is readable on a TV.
    world.add_light(name="key", pos=[0.6, -1.0, 1.6], dir=[-0.35, 0.6, -0.75], castshadow=True,
                    diffuse=[0.55, 0.55, 0.55], specular=[0.1, 0.1, 0.1])
    world.add_camera(name="demo_cam", pos=[0, -1.05, 0.78], xyaxes=[1, 0, 0, 0, 0.62, 0.78], fovy=50)
    if with_task_objects and task == "s2":
        _add_s2_objects(world)
        for hand in HANDS:
            _add_marker(world, hand)
    elif with_task_objects:
        for hand in HANDS:
            _add_tray(world, TRAY_OF[hand], layout.tray_pos[hand], COLOR_OF[hand])
            _add_cube(world, CUBE_OF[hand], COLOR_OF[hand])
            _add_marker(world, hand)
    return spec


def _add_marker(world: mujoco.MjsBody, hand: str) -> None:
    """Visual-only disc on the table under the gripper target (depth cue on a flat screen)."""
    body = world.add_body(name=f"{hand}/marker", mocap=True, pos=[0, 0, 0.0015])
    r, g, b, _ = COLOR_OF[hand]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.022, 0.0008, 0], contype=0, conaffinity=0,
                  rgba=[r, g, b, 0.45], group=2)


def build_model(layout: TaskLayout = S1_LAYOUT, with_task_objects: bool = True, task: str = "s1") -> mujoco.MjModel:
    return build_spec(layout, with_task_objects, task).compile()
