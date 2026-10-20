"""JoyConInputProvider: two Joy-Cons -> standard InputFrame.

Connection management runs in a background thread (open, detect loss, retry
every `reconnect_interval_s`). All per-report processing (stick calibration,
dead zone, debounce, gyro bias, axis unification) happens in poll(), in report
order using each report's own timestamp, so the result does not depend on how
often the game polls.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from teachabot.btlink import status as link_status

from .base import BUTTONS, HANDS, HandInput, InputFrame, InputProvider
from .joycon_device import JoyConDevice, JoyConSample
from .processing import (AxisMap, Debouncer, GyroBiasEstimator, StickCalibrator, VerticalMotionEstimator,
                         load_axis_map, radial_deadzone)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AXIS_MAP_PATH = ROOT / "config" / "joycon_axis_map.json"

# device button name -> canonical name (base.LEFT_BUTTONS / RIGHT_BUTTONS)
_BUTTON_MAP = {
    "left": {"zl": "grip", "l": "shoulder", "stick": "stick_click", "minus": "minus",
             "capture": "capture", "up": "up", "down": "down", "left": "left",
             "right": "right", "sl": "sl", "sr": "sr"},
    "right": {"zr": "grip", "r": "shoulder", "stick": "stick_click", "plus": "plus",
              "home": "home", "a": "a", "b": "b", "x": "x", "y": "y", "sl": "sl", "sr": "sr"},
}
_SIDE = {"left": "L", "right": "R"}


class _HandState:
    def __init__(self, hand: str, debounce_s: float):
        self.vertical = VerticalMotionEstimator()
        self.hand = hand
        self.calibrator = StickCalibrator()
        self.debouncers = {b: Debouncer(debounce_s) for b in BUTTONS[hand]}
        self.gyro_bias = GyroBiasEstimator()
        self.last_sample: JoyConSample | None = None
        self.reports = 0


def _install_reader_error_hook(on_error) -> None:
    """pyjoycon's reader thread ends with OSError('read error') whenever a Joy-Con
    disconnects or the process closes the device. That is expected: record it
    instead of printing a traceback. Other thread errors keep the default hook."""
    previous = threading.excepthook

    def hook(args):
        if issubclass(args.exc_type, OSError) and "_update_input_report" in (args.thread.name or ""):
            on_error(f"{args.thread.name}: {args.exc_value}")
            return
        previous(args)

    threading.excepthook = hook


class JoyConInputProvider(InputProvider):
    source = "joycon"

    def __init__(self, deadzone: float = 0.12, debounce_s: float = 0.012,
                 reconnect_interval_s: float = 2.0, first_report_timeout_s: float = 3.0,
                 axis_map_path: Path = DEFAULT_AXIS_MAP_PATH,
                 status_path: Path | None = link_status.APP_STATUS_PATH):
        """status_path: where to publish per-hand link status for the link manager
        (scripts/joycon_manager.py) once a second; None disables it."""
        self.deadzone = deadzone
        self.status_path = status_path
        # wall time of the last report with any button down, per hand (Joy-Con sleep timer)
        self._last_button_wall: dict[str, float | None] = {h: None for h in HANDS}
        self._connected_since_wall: dict[str, float | None] = {h: None for h in HANDS}
        self.reconnect_interval_s = reconnect_interval_s
        self.first_report_timeout_s = first_report_timeout_s
        self.axis_map_path = axis_map_path
        self.right_axis_map: AxisMap = load_axis_map(axis_map_path)
        self._devices = {h: JoyConDevice(_SIDE[h]) for h in HANDS}
        self._state = {h: _HandState(h, debounce_s) for h in HANDS}
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events_lock = threading.Lock()
        self.events: list[dict] = []  # connection events, newest last
        _install_reader_error_hook(lambda msg: self._event("reader_stopped", None, detail=msg))

    # -- InputProvider ----------------------------------------------------------

    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._manage, name="joycon-provider", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
        for dev in self._devices.values():
            dev.mark_disconnected()

    def poll(self) -> InputFrame:
        hands = {h: self._poll_hand(h) for h in HANDS}
        frame = InputFrame(timestamp=time.perf_counter(), seq=self._seq, source=self.source,
                           left=hands["left"], right=hands["right"],
                           meta={"imu_frame_verified": self.right_axis_map.verified,
                                 # m/s, + = up; Joy-Con moved up/down (VerticalMotionEstimator)
                                 "vertical_speed": {h: (self._state[h].vertical.output if hands[h].connected else 0.0)
                                                    for h in HANDS}})
        self._seq += 1
        return frame

    # -- extras -----------------------------------------------------------------

    def set_right_axis_map(self, axis_map: AxisMap) -> None:
        self.right_axis_map = axis_map

    def link_status(self) -> dict:
        """Per-side status for the link manager (see teachabot.btlink.status)."""
        return {"updated_wall": time.time(), "pid": os.getpid(),
                "hands": {_SIDE[h]: {"connected": self._devices[h].connected,
                                     "connected_since_wall": self._connected_since_wall[h],
                                     "last_button_wall": self._last_button_wall[h]}
                          for h in HANDS}}

    def status_line(self) -> str | None:
        """Sleep warning / repair instruction from the link manager, or this
        process's own estimate when the manager is not running."""
        link = link_status.read_fresh(link_status.LINK_STATUS_PATH)
        if link is not None:
            return link_status.warning_line(link)
        now = time.time()
        sides = {}
        for h in HANDS:
            if not self._devices[h].connected:
                continue
            last = self._last_button_wall[h]      # connecting is not proof of a press (app restart)
            sides[_SIDE[h]] = {"state": "ready", "warn": last is None,
                               "remaining_s": None if last is None else link_status.sleep_remaining_s(last, now)}
        return link_status.warning_line({"sides": sides})

    def calibration_status(self) -> dict:
        return {h: {"stick_center": s.calibrator.center,
                    "stick_half_range": [round(v) for v in s.calibrator.half_range],
                    "gyro_bias_dps": [round(v, 3) for v in s.gyro_bias.bias],
                    "gyro_bias_estimated": s.gyro_bias.estimated,
                    "reports": s.reports}
                for h, s in self._state.items()}

    # -- internals --------------------------------------------------------------

    def _event(self, kind: str, hand: str | None, **extra) -> None:
        with self._events_lock:
            self.events.append({"t": time.perf_counter(), "event": kind, "hand": hand, **extra})

    def _poll_hand(self, hand: str) -> HandInput:
        dev, st = self._devices[hand], self._state[hand]
        samples = dev.drain()
        if not dev.connected:
            for d in st.debouncers.values():
                d.reset()
            st.vertical.reset()
            return HandInput.neutral(hand, connected=False)
        for s in samples:
            st.calibrator.feed(s.stick_raw)
            for dev_name, canon in _BUTTON_MAP[hand].items():
                st.debouncers[canon].update(bool(s.buttons[dev_name]), s.t)
            if any(s.buttons.values()):
                self._last_button_wall[hand] = s.wall
            for sub in s.gyro_dps:
                st.gyro_bias.update(sub)
            for k in (2, 1, 0):                       # oldest -> newest (sub-sample 0 is the newest)
                st.vertical.update(s.accel_g[k], st.gyro_bias.apply(s.gyro_dps[k]))
            st.last_sample = s
            st.reports += 1
        s = st.last_sample or dev.latest()
        if s is None:
            return HandInput.neutral(hand, connected=False)
        stick = radial_deadzone(*st.calibrator.normalize(s.stick_raw), self.deadzone)
        accel_sensor = s.accel_g[0]               # sub-sample 0 is the newest (measured, see JoyConSample)
        gyro_debiased = st.gyro_bias.apply(s.gyro_dps[0])
        if hand == "right":
            accel, gyro = self.right_axis_map.apply(accel_sensor), self.right_axis_map.apply(gyro_debiased)
        else:
            accel, gyro = accel_sensor, gyro_debiased
        return HandInput(connected=True, stick=stick,
                         buttons={b: d.state for b, d in st.debouncers.items()},
                         accel=accel, gyro=gyro,
                         accel_sensor=accel_sensor, gyro_sensor=s.gyro_dps[0])

    def _manage(self) -> None:
        last_attempt = {h: -1e9 for h in HANDS}
        opened_at: dict[str, float | None] = {h: None for h in HANDS}
        was_connected = {h: False for h in HANDS}
        last_status = -1e9
        while not self._stop.is_set():
            now = time.perf_counter()
            if self.status_path is not None and now - last_status >= 1.0:
                last_status = now
                try:
                    link_status.write_json(self.status_path, self.link_status())
                except OSError:
                    pass                  # status is advisory; never break input for it
            for hand, dev in self._devices.items():
                if dev.connected:
                    if not was_connected[hand]:
                        was_connected[hand] = True
                        self._connected_since_wall[hand] = time.time()
                        self._event("connected", hand)
                    continue
                if was_connected[hand]:
                    was_connected[hand], opened_at[hand] = False, None
                    self._connected_since_wall[hand] = None
                    dev.mark_disconnected()
                    self._event("disconnected", hand)
                if opened_at[hand] is not None and dev.awaiting_first_report():
                    if now - opened_at[hand] < self.first_report_timeout_s:
                        continue
                    dev.mark_disconnected()
                    opened_at[hand] = None
                    self._event("no_reports_after_open", hand)
                if now - last_attempt[hand] < self.reconnect_interval_s:
                    continue
                last_attempt[hand] = now
                if dev.connect():
                    opened_at[hand] = time.perf_counter()
                elif dev.last_error and not dev.last_error.startswith("not found"):
                    self._event("open_failed", hand, detail=dev.last_error)
            self._stop.wait(0.1)
