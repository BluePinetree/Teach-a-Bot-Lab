import pytest

from teachabot.input.base import (BUTTONS, HandInput, InputFrame, InputValidationError)
from teachabot.input.keyboard import VK, KeyboardInputProvider


def _hand(hand, **kw):
    base = HandInput.neutral(hand, connected=True)
    return HandInput(**{**base.__dict__, **kw})


def _frame(**kw):
    d = dict(timestamp=1.0, seq=0, source="test", left=_hand("left"), right=_hand("right"))
    d.update(kw)
    return InputFrame(**d)


class TestFrame:
    def test_roundtrip(self):
        f = _frame(left=_hand("left", stick=(0.5, -0.25), accel=(0.0, 0.1, 1.0), gyro=(1.0, 2.0, 3.0),
                              buttons={**{b: False for b in BUTTONS["left"]}, "grip": True}))
        g = InputFrame.from_dict(f.to_dict())
        assert g == f and g.left.grip is True
        assert f.to_dict()["left"]["grip"] is True

    def test_stick_out_of_range(self):
        with pytest.raises(InputValidationError):
            _frame(left=_hand("left", stick=(1.2, 0.0))).validate()

    def test_wrong_button_set(self):
        with pytest.raises(InputValidationError):
            _frame(right=_hand("right", buttons={"grip": False})).validate()

    def test_disconnected_must_be_neutral(self):
        bad = HandInput(connected=False, stick=(0.3, 0.0), buttons={b: False for b in BUTTONS["left"]})
        with pytest.raises(InputValidationError):
            _frame(left=bad).validate()

    def test_schema_version_checked(self):
        d = _frame().to_dict()
        d["schema_version"] = 99
        with pytest.raises(InputValidationError):
            InputFrame.from_dict(d)


class TestKeyboard:
    def make(self, held):
        t = iter(range(1000))
        p = KeyboardInputProvider(key_reader=lambda vk: vk in {VK[k] for k in held},
                                  clock=lambda: float(next(t)))
        p.connect()
        return p

    def test_wasd_and_arrows(self):
        f = self.make({"W", "D", "LEFT", "RCTRL"}).poll().validate()
        assert f.left.stick == pytest.approx((0.7071, 0.7071), abs=1e-3)
        assert f.right.stick == (-1.0, 0.0) and f.right.grip and not f.left.grip
        assert f.left.accel is None

    def test_opposite_keys_cancel(self):
        f = self.make({"A", "D"}).poll()
        assert f.left.stick == (0.0, 0.0)

    def test_disconnected_neutral(self):
        p = self.make({"W"})
        p.disconnect()
        f = p.poll().validate()
        assert not f.left.connected and f.left.stick == (0.0, 0.0)
