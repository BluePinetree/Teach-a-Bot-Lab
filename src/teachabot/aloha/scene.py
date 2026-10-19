"""S1 scene: the MuJoCo Menagerie ALOHA 2 workcell plus two cubes and two trays.

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


def build_spec(layout: TaskLayout = S1_LAYOUT, with_task_objects: bool = True) -> mujoco.MjSpec:
    """Menagerie scene + task objects. `with_task_objects=False` gives the robot-only
    model used for inverse kinematics."""
    assets = _menagerie_assets()
    xml = _inline_includes(assets["scene.xml"].decode("utf-8"), assets)
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    world = spec.worldbody
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    # A key light with shadows from the operator's side, so depth is readable on a TV.
    world.add_light(name="key", pos=[0.6, -1.0, 1.6], dir=[-0.35, 0.6, -0.75], castshadow=True,
                    diffuse=[0.55, 0.55, 0.55], specular=[0.1, 0.1, 0.1])
    world.add_camera(name="demo_cam", pos=[0, -1.05, 0.78], xyaxes=[1, 0, 0, 0, 0.62, 0.78], fovy=50)
    if with_task_objects:
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


def build_model(layout: TaskLayout = S1_LAYOUT, with_task_objects: bool = True) -> mujoco.MjModel:
    return build_spec(layout, with_task_objects).compile()
