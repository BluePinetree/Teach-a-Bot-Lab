import math

import numpy as np
import pytest

from teachabot.input.processing import (AxisMap, Debouncer, GyroBiasEstimator, StickCalibrator,
                                        estimate_axis_map, radial_deadzone)


class TestRadialDeadzone:
    def test_inside_is_zero(self):
        assert radial_deadzone(0.05, -0.08, 0.12) == (0.0, 0.0)

    def test_full_deflection_stays_one(self):
        x, y = radial_deadzone(1.0, 0.0, 0.12)
        assert x == pytest.approx(1.0) and y == 0.0

    def test_just_outside_is_small(self):
        x, _ = radial_deadzone(0.13, 0.0, 0.12)
        assert 0 < x < 0.02

    def test_direction_preserved_on_diagonal(self):
        x, y = radial_deadzone(0.5, 0.5, 0.12)
        assert x == pytest.approx(y)
        assert math.hypot(x, y) == pytest.approx((math.hypot(0.5, 0.5) - 0.12) / 0.88)


class TestStickCalibrator:
    def test_uncalibrated_outputs_zero(self):
        c = StickCalibrator(settle_samples=5)
        assert c.normalize((3000, 3000)) == (0.0, 0.0)

    def test_center_from_still_samples(self):
        c = StickCalibrator(settle_samples=5)
        for raw in [(1940, 2348), (1941, 2347), (1939, 2349), (1940, 2348), (1942, 2346)]:
            c.feed(raw)
        assert c.center == (1940, 2348)
        assert c.normalize((1940, 2348)) == (0.0, 0.0)

    def test_moving_stick_during_settle_restarts(self):
        c = StickCalibrator(settle_samples=3, max_spread=60)
        for raw in [(1900, 2300), (2600, 2300), (1900, 2300)]:
            c.feed(raw)
        assert not c.calibrated
        for raw in [(1940, 2348)] * 3:
            c.feed(raw)
        assert c.center == (1940, 2348)

    def test_range_expands_and_clips(self):
        c = StickCalibrator(settle_samples=1, initial_half_range=1000, max_half_range=1700)
        c.feed((2000, 2000))
        assert c.normalize((3000, 2000)) == (1.0, 0.0)          # exactly initial range
        assert c.normalize((3300, 2000)) == (1.0, 0.0)          # range grows to 1300
        assert c.normalize((2650, 2000))[0] == pytest.approx(0.5)
        assert c.normalize((0, 2000))[0] == -1.0                # beyond cap, clipped
        assert c.half_range[0] == 1700

    def test_y_up_is_positive_for_larger_raw(self):
        c = StickCalibrator(settle_samples=1)
        c.feed((2000, 2000))
        assert c.normalize((2000, 2900))[1] > 0


class TestDebouncer:
    def test_single_report_glitch_ignored(self):
        d = Debouncer(hold_s=0.012)
        assert d.update(True, 0.000) is False
        assert d.update(False, 0.015) is False

    def test_two_reports_accept(self):
        d = Debouncer(hold_s=0.012)
        d.update(True, 0.000)
        assert d.update(True, 0.015) is True

    def test_release_also_debounced(self):
        d = Debouncer(hold_s=0.012)
        d.update(True, 0.0), d.update(True, 0.015)
        assert d.update(False, 0.030) is True
        assert d.update(False, 0.045) is False

    def test_reset(self):
        d = Debouncer()
        d.update(True, 0.0), d.update(True, 0.1)
        d.reset()
        assert d.state is False


class TestGyroBias:
    def test_still_estimates_bias(self):
        g = GyroBiasEstimator(window=20)
        rng = np.random.default_rng(0)
        for _ in range(20):
            g.update(tuple(np.array([0.5, -0.3, 0.1]) + rng.normal(0, 0.1, 3)))
        assert g.estimated
        assert g.bias == pytest.approx((0.5, -0.3, 0.1), abs=0.1)

    def test_motion_does_not_update(self):
        g = GyroBiasEstimator(window=20)
        for i in range(20):
            g.update((100.0 * math.sin(i), 0.0, 0.0))
        assert not g.estimated and g.bias == (0.0, 0.0, 0.0)


def _tilt_sequence(n=300, seed=0, f_roll=1.0, f_pitch=0.7):
    """Gravity vector in the left sensor frame while slowly tilting in two directions."""
    t = np.linspace(0, 10, n)
    roll, pitch = 0.6 * np.sin(f_roll * t), 0.5 * np.sin(f_pitch * t + 1)
    g = np.stack([np.sin(pitch), -np.sin(roll) * np.cos(pitch), np.cos(roll) * np.cos(pitch)], axis=1)
    return g + np.random.default_rng(seed).normal(0, 0.01, g.shape)


class TestEstimateAxisMap:
    @pytest.mark.parametrize("signs", [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])
    def test_recovers_proper_rotations(self, signs):
        left = _tilt_sequence()
        right = left * np.array(signs)
        m = estimate_axis_map(left, right)
        assert m.verified, m.detail
        assert m.signs == signs
        assert np.allclose(np.array([m.apply(r) for r in right]), left)

    def test_too_little_motion_unverified(self):
        left = np.tile([0.0, 0.0, 1.0], (100, 1)) + np.random.default_rng(1).normal(0, 0.01, (100, 3))
        m = estimate_axis_map(left, left * [1, -1, -1])
        assert not m.verified and "motion" in m.detail["reason"]

    def test_independent_motion_unverified(self):
        m = estimate_axis_map(_tilt_sequence(seed=1), _tilt_sequence(seed=2, f_roll=2.3, f_pitch=0.4))
        assert not m.verified

    def test_single_plane_tilt_is_enough(self):
        # roll only: gravity moves in the y-z plane, x stays ~0
        t = np.linspace(0, 10, 300)
        left = np.stack([np.zeros_like(t), -np.sin(0.6 * np.sin(t)), np.cos(0.6 * np.sin(t))], axis=1)
        left += np.random.default_rng(3).normal(0, 0.01, left.shape)
        m = estimate_axis_map(left, left * [-1, 1, -1])
        assert m.verified, m.detail
        assert m.signs == (-1, 1, -1)

    def test_odd_flip_rejected(self):
        left = _tilt_sequence()
        m = estimate_axis_map(left, left * [1, 1, -1])
        assert not m.verified and "reflection" in m.detail["reason"]

    def test_json_roundtrip(self):
        m = AxisMap(signs=(1, -1, -1), verified=True, detail={"x": 1})
        assert AxisMap.from_json(m.to_json()) == m


# -- vertical motion ----------------------------------------------------------------------

def _stroke(est, dz, duration=0.5, rest=1.0, scale=1.0, dt=0.005):
    """Feed a still hand, a smooth up/down stroke of dz metres (sine-shaped acceleration), then rest.
    Returns the integrated estimator output (the height a gripper would move at gain 1)."""
    import math as _m
    moved = 0.0
    n_rest, n_move = int(rest / dt), int(duration / dt)
    for _ in range(n_rest):
        moved += est.update((0.0, 0.0, scale), (0.0, 0.0, 0.0)) * dt
    for i in range(n_move):                                  # a(t) = A sin(2 pi t / T), displacement dz
        t = i * dt
        a = dz * 2 * _m.pi / duration ** 2 * _m.sin(2 * _m.pi * t / duration)
        moved += est.update((0.0, 0.0, scale + a / 9.81), (0.0, 0.0, 0.0)) * dt
    for _ in range(n_rest):
        moved += est.update((0.0, 0.0, scale), (0.0, 0.0, 0.0)) * dt
    return moved


def test_vertical_estimator_still_hand_stays_still_even_with_scale_error():
    from teachabot.input.processing import VerticalMotionEstimator
    est = VerticalMotionEstimator()
    assert abs(_stroke(est, 0.0, scale=1.09)) < 1e-6         # right Joy-Con reads 1.09 G at rest
    assert est.output == 0.0


def test_vertical_estimator_follows_direction_of_a_stroke():
    from teachabot.input.processing import VerticalMotionEstimator
    up = _stroke(VerticalMotionEstimator(), +0.10)
    down = _stroke(VerticalMotionEstimator(), -0.10)
    assert 0.06 < up < 0.12 and -0.12 < down < -0.06          # direction right, most of the 10 cm
    assert abs(up + down) < 0.01                              # symmetric


def test_vertical_estimator_ignores_how_the_joycon_is_held():
    import numpy as np
    from teachabot.input.processing import VerticalMotionEstimator
    est = VerticalMotionEstimator()
    g = np.array([0.0, np.sin(0.6), np.cos(0.6)])            # held tilted by ~34 deg, not moving
    moved = sum(est.update(g * 1.05, (0, 0, 0)) * 0.005 for _ in range(600))
    assert abs(moved) < 1e-6


def test_vertical_estimator_ignores_start_up_transient_and_impossible_speeds():
    from teachabot.input.processing import VerticalMotionEstimator
    est = VerticalMotionEstimator()
    for i in range(200):                                     # 1 s of IMU start-up: force swings wildly
        assert est.update((0.0, 0.0, 1.0 + 0.8 * (i % 7) / 7), (0.0, 0.0, 0.0)) == 0.0
    assert not est.settled
    assert abs(_stroke(est, 0.10, rest=1.0)) > 0.06          # settles on the next still period, then works
    outs = [est.update((0.0, 0.0, 4.0), (0.0, 0.0, 0.0)) for _ in range(100)]   # 3 G for 0.5 s: impossible
    assert max(abs(o) for o in outs) <= est.max_speed and outs[-1] == 0.0      # cut off, settles again


def test_vertical_estimator_clutch_only_counts_motion_while_held():
    from teachabot.input.processing import VerticalMotionEstimator
    import math as _m
    est = VerticalMotionEstimator()
    _stroke(est, 0.0, rest=1.5)                              # warm up
    moved = 0.0
    for i in range(160):                                     # released: a 20 cm stroke and a tilted hold
        a = 0.2 * 2 * _m.pi / 0.8 ** 2 * _m.sin(2 * _m.pi * i * 0.005 / 0.8)
        moved += est.update((0.0, 0.0, 1.0 + a / 9.81), (0.0, 0.0, 0.0), active=False) * 0.005
    for _ in range(200):
        moved += est.update((0.0, 0.5, 0.87 * 1.05), (0.0, 0.0, 0.0), active=False) * 0.005
    assert moved == 0.0
    for _ in range(200):                                     # pressed while tilted: reference already follows
        moved += est.update((0.0, 0.5, 0.87 * 1.05), (0.0, 0.0, 0.0), active=True) * 0.005
    assert abs(moved) < 1e-6
    n = int(0.5 / 0.005)                                     # held: a 10 cm stroke along the tilted gravity
    g = _m.hypot(0.5, 0.87 * 1.05)
    for i in range(n):
        a = 0.10 * 2 * _m.pi / 0.5 ** 2 * _m.sin(2 * _m.pi * i * 0.005 / 0.5) / 9.81
        moved += est.update((0.0, 0.5 * (1 + a / g), 0.87 * 1.05 * (1 + a / g)), (0.0, 0.0, 0.0), active=True) * 0.005
    for _ in range(200):
        moved += est.update((0.0, 0.5, 0.87 * 1.05), (0.0, 0.0, 0.0), active=True) * 0.005
    assert 0.06 < moved < 0.12
