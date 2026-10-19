"""Per-side link state machine (pure logic, no Windows calls; unit-tested).

Measured behaviour it is built on (docs/JOYCON_LINK_MANAGER_PLAN.md section 9):
- A Joy-Con sleeps 15 min after its last button press. Waking it needs a button,
  so a wake (Bluetooth link coming up) also counts as a button press.
- After a sleep it often comes back "stuck": the Bluetooth link is up (it even
  re-connects every 11-15 s for 2-3 min) but HID carries nothing in either
  direction. Only re-pairing fixes that.

States: unpaired, asleep, waking, ready, stuck, repairing.
update() returns the action the caller should take: "pair", "repair", "probe" or None.
"""
from __future__ import annotations

from dataclasses import dataclass

from teachabot.btlink.status import WARN_BEFORE_S, fmt_mmss, sleep_remaining_s

STUCK_AFTER_S = 20.0        # link up this long without reports => stuck
BRIEF_DROP_S = 5.0          # shorter link drops (stuck Joy-Cons flap) keep the wake window open
PROBE_AFTER_S = 2.0         # without the app, probe HID this long after the link comes up


@dataclass(frozen=True)
class Observation:
    paired: bool
    bt: bool           # Windows Bluetooth link up
    hid: bool          # listed in HID enumeration


@dataclass(frozen=True)
class AppHand:
    connected: bool                  # the app is receiving reports from this Joy-Con
    last_button_wall: float | None   # wall time of the last report with a button down


class SideMonitor:
    def __init__(self, side: str):
        self.side = side
        self.state = "unknown"
        self.message = ""
        self.remaining_s: float | None = None
        self.warn = False
        self._repairing = False
        self._repair_message = ""
        self._wake_wall: float | None = None
        self._down_since: float | None = None
        self._last_input_wall: float | None = None
        self._probe: bool | None = None
        self._app_since: float | None = None

    # -- repair flow (driven by the caller) ---------------------------------------

    def begin_repair(self, message: str) -> None:
        self._repairing, self._repair_message = True, message

    def set_repair_message(self, message: str) -> None:
        self._repair_message = message

    def end_repair(self, ok: bool, now_wall: float) -> None:
        self._repairing = False
        self._probe = None
        if ok:   # pairing needs the SYNC button: that is a button press, and the link is fresh
            self._wake_wall = self._last_input_wall = now_wall
        else:
            self._wake_wall = None

    def set_probe_result(self, ok: bool) -> None:
        self._probe = ok

    # -- main ---------------------------------------------------------------------

    def update(self, now_wall: float, obs: Observation, app: AppHand | None) -> str | None:
        first = self.state == "unknown"
        if app is None:
            self._app_since = None
        elif self._app_since is None:
            self._app_since = now_wall
        if self._repairing:
            return self._set("repairing", self._repair_message)
        if not obs.paired:
            self._wake_wall = None
            return self._set("unpaired", "SYNC 버튼(레일 옆 작은 버튼)을 누르세요", action="pair")

        if not obs.bt:
            if self._down_since is None:
                self._down_since = now_wall
            if self._wake_wall is not None and now_wall - self._down_since < BRIEF_DROP_S:
                return None                       # brief drop: keep the current state
            self._wake_wall, self._probe = None, None
            return self._set("asleep", "잠듦 · 아무 버튼이나 눌러 깨우세요")

        self._down_since = None
        if self._wake_wall is None:
            self._wake_wall = now_wall
            self._probe = None
            if not first:                         # just woke up: someone pressed a button
                self._last_input_wall = max(self._last_input_wall or 0.0, now_wall)
            # already up when we started watching: the last button press is unknown

        healthy = app.connected if app is not None else self._probe is True
        if healthy:
            if app is not None and app.last_button_wall:
                self._last_input_wall = max(self._last_input_wall or 0.0, app.last_button_wall)
            self.state = "ready"
            if self._last_input_wall is None:
                self.remaining_s, self.warn = None, True
                self.message = "준비됨 · 잠들기까지 남은 시간 모름 · 아무 버튼이나 한 번 눌러 주세요"
                return None
            self.remaining_s = sleep_remaining_s(self._last_input_wall, now_wall)
            self.warn = self.remaining_s <= WARN_BEFORE_S
            text = f"준비됨 · 잠들기까지 {fmt_mmss(self.remaining_s)}"
            if self.warn:
                text += " · 아무 버튼이나 한 번 눌러 주세요"
            if app is None:
                text += " (앱 꺼짐: 버튼 입력은 못 봄)"
            self.message = text
            return None

        if app is None:
            if self._probe is None:
                if obs.hid and now_wall - self._wake_wall >= PROBE_AFTER_S:
                    return self._set("waking", "연결 확인 중…", action="probe")
                return self._set("waking", "연결 중…")
            return self._set("stuck", "고착됨(신호가 안 옴) → 재페어링", action="repair")
        watching_since = max(self._wake_wall, self._app_since)
        if now_wall - watching_since >= STUCK_AFTER_S:
            return self._set("stuck", "고착됨(신호가 안 옴) → 재페어링", action="repair")
        return self._set("waking", "연결 중…")

    def _set(self, state: str, message: str, action: str | None = None) -> str | None:
        self.state, self.message = state, message
        self.remaining_s, self.warn = None, False
        return action

    def snapshot(self) -> dict:
        return {"state": self.state, "message": self.message,
                "remaining_s": None if self.remaining_s is None else round(self.remaining_s, 1),
                "warn": self.warn}
