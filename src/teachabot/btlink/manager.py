"""LinkManager: keeps the two Joy-Cons paired, connected and awake.

Every 0.5 s, per side: Bluetooth link + pairing (WinRT), HID enumeration, and
the app's status file -> SideMonitor -> action:
  pair    not paired: wait for SYNC and pair (first-time setup, or after an unpair)
  repair  stuck: forget the pairing, wait for SYNC, pair again (auto, our MACs only)
  probe   app not running: briefly open HID to tell healthy from stuck
Publishes data/runtime/joycon_link_status.json for the app's HUD line.
Never holds the HID device open, so it runs next to the app.
"""
from __future__ import annotations

import asyncio
import time

from teachabot.btlink import status as link_status
from teachabot.btlink.monitor import AppHand, Observation, SideMonitor
from teachabot.btlink.probe import probe_hid
from teachabot.btlink.winbt import BtLink, find_pairable, load_macs, pair, save_mac, unpair
from teachabot.input.joycon_device import PRODUCT_IDS, enumerate_joycons

SIDES = ("L", "R")
TICK_S = 0.5
PAIRED_REFRESH_S = 2.0
SYNC_MSG = "SYNC 버튼(레일 옆 작은 버튼)을 한 번 누르세요"


class LinkManager:
    def __init__(self, auto_repair: bool = True, log=print):
        self.auto_repair = auto_repair
        self.log = log
        self.macs = load_macs()
        self.monitors = {s: SideMonitor(s) for s in SIDES}
        self.links: dict[str, BtLink] = {}
        self._paired = {s: False for s in SIDES}
        self._paired_checked = {s: -1e9 for s in SIDES}
        self._tasks: dict[str, asyncio.Task] = {}
        self._probing: set[str] = set()
        self._manual: set[str] = set()
        self.app_alive = False

    # -- public -------------------------------------------------------------------

    def request_repair(self, side: str) -> None:
        """Manual 're-pair' button (call on the manager's event loop)."""
        self._manual.add(side)

    def snapshot(self) -> dict:
        return {"updated_wall": time.time(), "app_alive": self.app_alive, "auto_repair": self.auto_repair,
                "sides": {s: {**m.snapshot(), "mac": self.macs.get(s)} for s, m in self.monitors.items()}}

    async def run(self, stop: asyncio.Event) -> None:
        for s in SIDES:
            await self._open_link(s)
        try:
            while not stop.is_set():
                try:
                    await self._tick()
                except Exception as e:          # keep managing; one bad WinRT call is not fatal
                    self.log(f"tick error: {type(e).__name__}: {e}")
                try:
                    link_status.write_json(link_status.LINK_STATUS_PATH, self.snapshot())
                except OSError as e:            # the status file is advisory
                    self.log(f"status write error: {e}")
                try:
                    await asyncio.wait_for(stop.wait(), TICK_S)
                except asyncio.TimeoutError:
                    pass
        finally:
            for t in self._tasks.values():
                t.cancel()
            for link in self.links.values():
                link.close()

    # -- internals ----------------------------------------------------------------

    async def _open_link(self, side: str) -> None:
        old = self.links.pop(side, None)
        if old is not None:
            old.close()
        mac = self.macs.get(side)
        if not mac:
            self._paired[side] = False
            return
        link = BtLink(side, mac)
        st = await link.open()
        self.links[side] = link
        self._paired[side] = st.paired
        self._paired_checked[side] = time.perf_counter()

    async def _tick(self) -> None:
        now_wall, now = time.time(), time.perf_counter()
        app = link_status.read_fresh(link_status.APP_STATUS_PATH, now_wall)
        self.app_alive = app is not None
        present = {d["product_id"] for d in enumerate_joycons()}
        for s in SIDES:
            link = self.links.get(s)
            busy = s in self._tasks and not self._tasks[s].done()
            if link is not None and not busy and now - self._paired_checked[s] >= PAIRED_REFRESH_S:
                self._paired_checked[s] = now
                try:
                    self._paired[s] = await link.refresh_paired()
                except Exception:
                    pass                        # keep the last known pairing state
            obs = Observation(paired=self._paired[s],
                              bt=link is not None and link.state().connected,
                              hid=PRODUCT_IDS[s] in present)
            hand = (app or {}).get("hands", {}).get(s)
            app_hand = None if hand is None else AppHand(bool(hand.get("connected")), hand.get("last_button_wall"))
            action = self.monitors[s].update(now_wall, obs, app_hand)
            if s in self._manual:
                self._manual.discard(s)
                action = "repair_manual"
            if busy:
                continue
            if action == "pair":
                self._start(s, unpair_first=False)
            elif action == "repair" and self.auto_repair or action == "repair_manual":
                self.log(f"{s}: {'수동' if action == 'repair_manual' else '고착 감지'} → 재페어링")
                self._start(s, unpair_first=True)
            elif action == "probe" and s not in self._probing:
                self._probing.add(s)
                asyncio.get_running_loop().create_task(self._probe(s))

    async def _probe(self, side: str) -> None:
        try:
            ok = await asyncio.to_thread(probe_hid, side)
            self.monitors[side].set_probe_result(ok)
            self.log(f"{side}: HID 확인 {'정상' if ok else '응답 없음(고착)'}")
        finally:
            self._probing.discard(side)

    def _start(self, side: str, unpair_first: bool) -> None:
        self._tasks[side] = asyncio.get_running_loop().create_task(self._pair_flow(side, unpair_first))

    async def _pair_flow(self, side: str, unpair_first: bool) -> None:
        mon = self.monitors[side]
        mon.begin_repair("기존 페어링 지우는 중…" if unpair_first else SYNC_MSG)
        ok = False
        try:
            if unpair_first and self.macs.get(side):
                self.log(f"{side}: 페어링 해제 {await unpair(self.macs[side])}")
                self._paired[side] = False
            mon.set_repair_message(SYNC_MSG)
            while not ok:
                found = await find_pairable({side}, 10.0, self.macs)
                for target in found:
                    mon.set_repair_message("페어링 중…")
                    t0 = time.perf_counter()
                    result = await pair(target)
                    self.log(f"{side}: 페어링 시도 {target.mac} -> {result} ({time.perf_counter() - t0:.1f}s)")
                    if result == "ok":
                        if self.macs.get(side) != target.mac:
                            self.macs[side] = target.mac
                            save_mac(side, target.mac)
                        ok = True
                        break
                    mon.set_repair_message(SYNC_MSG)
                if not ok:
                    await asyncio.sleep(0.5)
            await self._open_link(side)
        finally:
            mon.end_repair(ok, time.time())
