"""Low-level access to Nintendo Switch 1 Joy-Con (L/R) over Bluetooth HID.

Built on joycon-python (pyjoycon). Phase 1 scope only: discover each side,
read sticks / buttons / IMU per HID report, notice disconnects, reconnect.
Game code must not import this module directly; Phase 2 wraps it in
JoyConInputProvider.

pyjoycon notes that shape this wrapper:
- Its reader thread calls our hook once per 0x30 report (~15 ms). Each report
  carries 3 IMU sub-samples ~5 ms apart; pyjoycon's get_status() only exposes
  the first, so we read all three.
- It opens the device by VID/PID (serial ignored), so one L and one R are supported.
- Its constructor does blocking HID reads with no timeout, so connect() runs
  it in a worker thread and gives up after a timeout.
- It has no close/stop for the reader thread. On reconnect we abandon the old
  JoyCon object and drop any samples it still delivers (generation check).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import hid
from pyjoycon import JoyCon
from pyjoycon.constants import JOYCON_L_PRODUCT_ID, JOYCON_R_PRODUCT_ID, JOYCON_VENDOR_ID

# Unit conversion from dekuNukem Nintendo_Switch_Reverse_Engineering/imu_sensor_notes.md
# (default ranges accel ±8 G, gyro ±2000 dps). pyjoycon has already applied the
# factory/user calibration offset and gain, so these turn its output into G and dps.
# Axes are the Joy-Con's own sensor frame; L and R have one axis mirrored relative
# to each other (documented in the same notes) and are NOT remapped here.
ACCEL_G_PER_LSB = 0.000244
GYRO_DPS_PER_LSB = 0.06103

PRODUCT_IDS = {"L": JOYCON_L_PRODUCT_ID, "R": JOYCON_R_PRODUCT_ID}

_BUTTON_GETTERS = {
    "L": {
        "up": "get_button_up", "down": "get_button_down",
        "left": "get_button_left", "right": "get_button_right",
        "l": "get_button_l", "zl": "get_button_zl",
        "minus": "get_button_minus", "stick": "get_button_l_stick",
        "capture": "get_button_capture",
        "sl": "get_button_left_sl", "sr": "get_button_left_sr",
    },
    "R": {
        "a": "get_button_a", "b": "get_button_b",
        "x": "get_button_x", "y": "get_button_y",
        "r": "get_button_r", "zr": "get_button_zr",
        "plus": "get_button_plus", "stick": "get_button_r_stick",
        "home": "get_button_home",
        "sl": "get_button_right_sl", "sr": "get_button_right_sr",
    },
}

_STICK_GETTERS = {
    "L": ("get_stick_left_horizontal", "get_stick_left_vertical"),
    "R": ("get_stick_right_horizontal", "get_stick_right_vertical"),
}


@dataclass(frozen=True)
class JoyConSample:
    """One HID input report (0x30) from one Joy-Con."""

    side: str                     # "L" or "R"
    t: float                      # host time.perf_counter() when the report arrived
    wall: float                   # host time.time() for log alignment
    timer: int                    # Joy-Con report timer byte (wraps at 256)
    battery_level: int            # 0..8 (pyjoycon 3-bit field)
    stick_raw: tuple[int, int]    # 12-bit, uncalibrated (0..4095), not deadzoned
    buttons: dict[str, int]       # name -> 0/1
    # 3 IMU sub-samples ~5 ms apart, index 0 = NEWEST, 2 = oldest (verified on 2026-10-17 logs:
    # ordering 2,1,0 as old->new gives a 40x smoother signal than 0,1,2)
    accel_lsb: tuple[tuple[float, float, float], ...]  # calibrated LSB
    gyro_lsb: tuple[tuple[float, float, float], ...]

    @property
    def accel_g(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(tuple(v * ACCEL_G_PER_LSB for v in s) for s in self.accel_lsb)

    @property
    def gyro_dps(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(tuple(v * GYRO_DPS_PER_LSB for v in s) for s in self.gyro_lsb)

    def to_record(self) -> dict:
        return {
            "type": "report", "side": self.side, "t": round(self.t, 6),
            "wall": round(self.wall, 6), "timer": self.timer,
            "battery": self.battery_level, "stick_raw": list(self.stick_raw),
            "buttons": self.buttons,
            "accel_lsb": [list(s) for s in self.accel_lsb],
            "gyro_lsb": [list(s) for s in self.gyro_lsb],
            "accel_g": [[round(v, 5) for v in s] for s in self.accel_g],
            "gyro_dps": [[round(v, 3) for v in s] for s in self.gyro_dps],
        }


def enumerate_joycons() -> list[dict]:
    """All Switch 1 Joy-Con HID interfaces Windows currently exposes."""
    return [
        d for d in hid.enumerate(JOYCON_VENDOR_ID, 0)
        if d["product_id"] in PRODUCT_IDS.values()
    ]


class JoyConDevice:
    """One side (L or R). Thread-safe accessors for the latest sample and a sample queue."""

    STALE_AFTER_S = 1.0  # no report for this long => treat as disconnected

    def __init__(self, side: str, queue_size: int = 20000):
        if side not in PRODUCT_IDS:
            raise ValueError(f"side must be 'L' or 'R', got {side!r}")
        self.side = side
        self.product_id = PRODUCT_IDS[side]
        self._lock = threading.Lock()
        self._joycon: JoyCon | None = None
        self._generation = 0
        self._latest: JoyConSample | None = None
        self._queue: deque[JoyConSample] = deque(maxlen=queue_size)
        self.last_error: str | None = None

    # -- connection -------------------------------------------------------

    def is_present(self) -> bool:
        return any(d["product_id"] == self.product_id for d in enumerate_joycons())

    def connect(self, timeout_s: float = 5.0) -> bool:
        """Open the device and start receiving reports. Returns True on success."""
        if not self.is_present():
            self.last_error = "not found in HID enumeration (not paired / not connected)"
            return False

        result: dict = {}

        def _open():
            try:
                result["joycon"] = JoyCon(JOYCON_VENDOR_ID, self.product_id)
            except Exception as e:  # pyjoycon raises IOError/AssertionError/etc.
                result["error"] = f"{type(e).__name__}: {e}"

        worker = threading.Thread(target=_open, name=f"joycon-open-{self.side}", daemon=True)
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            self.last_error = f"open timed out after {timeout_s}s"
            return False
        if "error" in result:
            self.last_error = result["error"]
            return False

        joycon = result["joycon"]
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._joycon = joycon
            self._latest = None
        joycon.register_update_hook(lambda jc: self._on_report(jc, generation))
        self.last_error = None
        return True

    def mark_disconnected(self) -> None:
        """Abandon the current handle (its reader thread may still be blocked in read)."""
        with self._lock:
            self._generation += 1
            self._joycon = None

    @property
    def connected(self) -> bool:
        with self._lock:
            if self._joycon is None:
                return False
            thread = getattr(self._joycon, "_update_input_report_thread", None)
            if thread is not None and not thread.is_alive():
                return False
            latest = self._latest
        return latest is not None and time.perf_counter() - latest.t < self.STALE_AFTER_S

    def send_keepalive(self, kind: str = "rumble") -> bool:
        """Write something harmless to the Joy-Con (Phase 0 experiment).

        kind "rumble": rumble-only report 0x10 with neutral data. Measured
            2026-10-19: does NOT stop the ~15 min sleep.
        kind "lamp": subcommand 0x30, player lamp 1 on (what a Switch sends
            after connecting).
        See docs/JOYCON_LINK_MANAGER_PLAN.md section 9.
        """
        with self._lock:
            joycon = self._joycon
        if joycon is None:
            return False
        try:
            if kind == "lamp":
                joycon.set_player_lamp_on(0x1)
            else:
                joycon._write_output_report(b"\x10", b"", b"")
            return True
        except (OSError, ValueError):
            return False

    def awaiting_first_report(self) -> bool:
        with self._lock:
            return self._joycon is not None and self._latest is None

    # -- data -------------------------------------------------------------

    def _on_report(self, jc: JoyCon, generation: int) -> None:
        # Runs in pyjoycon's reader thread, right after it stored the report,
        # so the getters below all read the same report.
        if generation != self._generation:
            return
        side = self.side
        sample = JoyConSample(
            side=side,
            t=time.perf_counter(),
            wall=time.time(),
            timer=jc._input_report[1],
            battery_level=jc.get_battery_level(),
            stick_raw=tuple(getattr(jc, g)() for g in _STICK_GETTERS[side]),
            buttons={name: getattr(jc, g)() for name, g in _BUTTON_GETTERS[side].items()},
            accel_lsb=tuple(
                (jc.get_accel_x(i), jc.get_accel_y(i), jc.get_accel_z(i)) for i in range(3)
            ),
            gyro_lsb=tuple(
                (jc.get_gyro_x(i), jc.get_gyro_y(i), jc.get_gyro_z(i)) for i in range(3)
            ),
        )
        with self._lock:
            if generation != self._generation:
                return
            self._latest = sample
            self._queue.append(sample)

    def latest(self) -> JoyConSample | None:
        with self._lock:
            return self._latest

    def drain(self) -> list[JoyConSample]:
        """Samples received since the previous drain(), oldest first."""
        with self._lock:
            out = list(self._queue)
            self._queue.clear()
        return out
