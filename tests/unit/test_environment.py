import pytest

from teachabot.input.base import BUTTONS, HandInput, InputFrame
from teachabot.simulation.environment import HandAction, World, actions_from_input

DT = 1 / 60


def drive(world, hand, vx=0.0, vz=0.0, grip=False, seconds=0.0, **other):
    for _ in range(max(1, round(seconds / DT))):
        world.step({hand: HandAction(vx, vz, grip), **other}, DT)


def move_to(world, hand, x, z, grip=False):
    g = world.grippers[hand]
    for _ in range(600):
        dx, dz = x - g.x, z - g.z
        if abs(dx) < 0.004 and abs(dz) < 0.004:
            return
        s = max(abs(dx), abs(dz), 1e-9)
        k = min(1.0, max(abs(dx), abs(dz)) / (world.config.hand_speed * DT))
        world.step({hand: HandAction(k * dx / s, k * dz / s, grip)}, DT)
    raise AssertionError(f"{hand} could not reach ({x}, {z}); at ({g.x:.3f}, {g.z:.3f})")


def test_stick_moves_gripper_at_configured_speed():
    w = World.default_scene()
    x0, z0 = w.grippers["left"].x, w.grippers["left"].z
    drive(w, "left", vx=1.0, seconds=0.5)
    assert w.grippers["left"].x - x0 == pytest.approx(0.25, abs=0.01)
    drive(w, "left", vz=-1.0, seconds=0.1)
    assert w.grippers["left"].z == pytest.approx(z0 - 0.05, abs=0.005)


def test_diagonal_speed_is_clipped():
    w = World.default_scene()
    g = w.grippers["right"]
    x0, z0 = g.x, g.z
    drive(w, "right", vx=-1.0, vz=1.0, seconds=0.2)
    assert ((g.x - x0) ** 2 + (g.z - z0) ** 2) ** 0.5 == pytest.approx(0.1, abs=0.005)


def test_hand_limits_and_center_overlap():
    w = World.default_scene()
    drive(w, "left", vx=1.0, seconds=3)
    assert w.grippers["left"].x == pytest.approx(w.config.center_overlap)
    drive(w, "right", vz=1.0, seconds=3)
    assert w.grippers["right"].z == pytest.approx(w.config.table_half_z)


def test_large_dt_is_clamped():
    w = World.default_scene()
    x0 = w.grippers["left"].x
    w.step({"left": HandAction(1.0, 0.0)}, 5.0)
    assert w.grippers["left"].x - x0 == pytest.approx(w.config.hand_speed * w.config.max_dt)


def test_grasp_carry_release():
    w = World.default_scene()
    blue = w.cubes["blue"]
    move_to(w, "left", blue.x, blue.z)
    drive(w, "left", grip=True)
    assert w.grippers["left"].holding == "blue" and blue.held_by == "left"
    assert blue.y == pytest.approx(w.config.carry_height)
    t = w.targets["left_box"]
    move_to(w, "left", t.x, t.z, grip=True)
    assert (blue.x, blue.z) == pytest.approx((t.x, t.z), abs=0.005)
    assert w.cube_in_target("blue") is None          # still held
    drive(w, "left", grip=False)
    assert blue.held_by is None and blue.y == pytest.approx(w.config.cube_size / 2)
    assert w.cube_in_target("blue") == "left_box"
    assert [e["event"] for e in w.events] == ["grasp", "release"]


def test_closing_away_from_cube_grabs_nothing_even_if_moved_over_it():
    w = World.default_scene()
    blue = w.cubes["blue"]
    move_to(w, "left", blue.x + 0.2, blue.z)
    drive(w, "left", grip=True)
    move_to(w, "left", blue.x, blue.z, grip=True)
    assert w.grippers["left"].holding is None and blue.held_by is None


def test_out_of_radius_does_not_grasp():
    w = World.default_scene()
    blue = w.cubes["blue"]
    move_to(w, "left", blue.x + w.config.grasp_radius + 0.01, blue.z)
    drive(w, "left", grip=True)
    assert w.grippers["left"].holding is None


def test_cube_held_by_one_hand_is_not_free():
    w = World.default_scene()
    blue = w.cubes["blue"]
    move_to(w, "left", blue.x, blue.z)
    drive(w, "left", grip=True)
    assert w.nearest_free_cube(blue.x, blue.z) is None


def _frame(left_stick=(0.0, 0.0), right_grip=False, right_connected=True):
    lb = {b: False for b in BUTTONS["left"]}
    rb = {b: False for b in BUTTONS["right"]} | {"grip": right_grip}
    right = HandInput(True, (0.0, 0.0), rb) if right_connected else HandInput.neutral("right")
    return InputFrame(0.0, 0, "test", HandInput(True, left_stick, lb), right)


def test_actions_from_input_mapping():
    a = actions_from_input(_frame(left_stick=(0.5, -1.0), right_grip=True))
    assert (a["left"].vx, a["left"].vz, a["left"].grip) == (0.5, -1.0, False)
    assert a["right"].grip is True


def test_disconnected_hand_stops_and_releases():
    w = World.default_scene()
    orange = w.cubes["orange"]
    move_to(w, "right", orange.x, orange.z)
    w.step(actions_from_input(_frame(right_grip=True)), DT)
    assert orange.held_by == "right"
    w.step(actions_from_input(_frame(right_connected=False)), DT)
    assert orange.held_by is None
