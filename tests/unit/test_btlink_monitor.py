"""Link state machine (teachabot.btlink.monitor) and the shared status helpers."""
from teachabot.btlink import status
from teachabot.btlink.monitor import BRIEF_DROP_S, STUCK_AFTER_S, AppHand, Observation, SideMonitor

UP = Observation(paired=True, bt=True, hid=True)
DOWN = Observation(paired=True, bt=False, hid=False)
UNPAIRED = Observation(paired=False, bt=False, hid=False)
APP_OK = AppHand(connected=True, last_button_wall=None)
APP_NO_REPORTS = AppHand(connected=False, last_button_wall=None)
T0 = 1_000_000.0


def woke(m: SideMonitor, app=APP_OK, at: float = T0) -> SideMonitor:
    """Seen asleep first, then the link comes up at `at` (a button woke it)."""
    m.update(at - 60, DOWN, app)
    m.update(at, UP, app)
    return m


def test_already_connected_at_start_means_unknown_until_a_button():
    """Manager started while the Joy-Con was up: its last press is unknown (seen 2026-10-20:
    treating startup as a press overstated the time left by minutes)."""
    m = SideMonitor("R")
    m.update(T0, UP, APP_OK)
    assert m.state == "ready" and m.remaining_s is None and m.warn
    assert "남은 시간 모름" in status.warning_line({"sides": {"R": m.snapshot()}})
    m.update(T0 + 5, UP, AppHand(connected=True, last_button_wall=T0 + 4))
    assert m.remaining_s == status.SLEEP_AFTER_S - 1 and not m.warn


def test_unpaired_asks_for_pairing():
    m = SideMonitor("L")
    assert m.update(T0, UNPAIRED, None) == "pair"
    assert m.state == "unpaired" and "SYNC" in m.message


def test_asleep_when_link_down():
    m = SideMonitor("L")
    assert m.update(T0, DOWN, APP_OK) is None
    assert m.state == "asleep"


def test_ready_with_app_counts_down_from_wake():
    m = woke(SideMonitor("L"))
    assert m.state == "ready"
    m.update(T0 + 60, UP, APP_OK)
    assert m.remaining_s == status.SLEEP_AFTER_S - 60
    assert not m.warn


def test_button_press_resets_the_sleep_timer():
    m = woke(SideMonitor("R"))
    later = T0 + 14 * 60
    m.update(later, UP, APP_OK)
    assert m.warn                                   # under 3 min left
    m.update(later + 1, UP, AppHand(connected=True, last_button_wall=later))
    assert not m.warn and m.remaining_s == status.SLEEP_AFTER_S - 1


def test_warning_line_for_hud():
    m = woke(SideMonitor("L"))
    m.update(T0 + 13 * 60, UP, APP_OK)
    line = status.warning_line({"sides": {"L": m.snapshot()}})
    assert line.startswith("L 2:00 뒤 잠듦")
    m.update(T0 + 60, UP, AppHand(connected=True, last_button_wall=T0 + 60))
    assert status.warning_line({"sides": {"L": m.snapshot()}}) is None


def test_stuck_with_app_after_timeout_then_repair():
    m = woke(SideMonitor("L"), APP_NO_REPORTS)
    assert m.state == "waking"
    assert m.update(T0 + STUCK_AFTER_S - 1, UP, APP_NO_REPORTS) is None
    assert m.update(T0 + STUCK_AFTER_S, UP, APP_NO_REPORTS) == "repair"
    assert m.state == "stuck"


def test_stuck_joycon_flapping_keeps_the_wake_window():
    """Stuck Joy-Cons drop the link for ~0.2 s every 11-15 s; that must not restart the clock."""
    m = SideMonitor("L")
    m.update(T0, UP, APP_NO_REPORTS)
    m.update(T0 + 12, DOWN, APP_NO_REPORTS)
    m.update(T0 + 12.2, UP, APP_NO_REPORTS)
    assert m.update(T0 + STUCK_AFTER_S, UP, APP_NO_REPORTS) == "repair"


def test_long_drop_means_asleep_and_resets():
    m = SideMonitor("L")
    m.update(T0, UP, APP_NO_REPORTS)
    m.update(T0 + 1, DOWN, APP_NO_REPORTS)
    m.update(T0 + 1 + BRIEF_DROP_S, DOWN, APP_NO_REPORTS)
    assert m.state == "asleep"
    m.update(T0 + 30, UP, APP_NO_REPORTS)          # new wake: clock restarts
    assert m.update(T0 + 30 + STUCK_AFTER_S - 1, UP, APP_NO_REPORTS) is None


def test_app_that_just_started_gets_time_before_stuck():
    m = SideMonitor("L")
    m.update(T0, UP, None)
    m.set_probe_result(True)
    m.update(T0 + 1, UP, None)
    assert m.state == "ready"
    m.update(T0 + 300, UP, APP_NO_REPORTS)          # app launched long after the wake
    assert m.update(T0 + 300 + STUCK_AFTER_S - 1, UP, APP_NO_REPORTS) is None
    assert m.update(T0 + 300 + STUCK_AFTER_S, UP, APP_NO_REPORTS) == "repair"


def test_without_app_probe_decides():
    m = SideMonitor("L")
    assert m.update(T0, UP, None) is None
    assert m.update(T0 + 2.5, UP, None) == "probe"
    m.set_probe_result(False)
    assert m.update(T0 + 3, UP, None) == "repair"


def test_probe_ok_is_ready_and_says_app_is_off():
    m = woke(SideMonitor("R"), None)
    m.set_probe_result(True)
    m.update(T0 + 3, UP, None)
    assert m.state == "ready" and "앱 꺼짐" in m.message


def test_repair_flow_and_fresh_timer_after_pairing():
    m = SideMonitor("L")
    m.begin_repair("SYNC 버튼을 누르세요")
    assert m.update(T0, UNPAIRED, None) is None
    assert m.state == "repairing" and m.message == "SYNC 버튼을 누르세요"
    m.end_repair(True, T0 + 30)
    m.update(T0 + 31, UP, APP_OK)
    assert m.state == "ready" and m.remaining_s == status.SLEEP_AFTER_S - 1


def test_write_json_survives_a_reader_holding_the_file(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    calls = {"n": 0}
    real_replace = status.os.replace

    def flaky_replace(src, dst):      # Windows: destination open in another process
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(status.os, "replace", flaky_replace)
    assert status.write_json(p, {"updated_wall": T0}) is True
    monkeypatch.setattr(status.os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError(5, "x")))
    assert status.write_json(p, {"updated_wall": T0}) is False


def test_status_files_roundtrip_and_staleness(tmp_path):
    p = tmp_path / "s.json"
    status.write_json(p, {"updated_wall": T0, "x": 1})
    assert status.read_fresh(p, now_wall=T0 + 1)["x"] == 1
    assert status.read_fresh(p, now_wall=T0 + status.FRESH_S + 1) is None
    assert status.read_fresh(tmp_path / "missing.json") is None
