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
