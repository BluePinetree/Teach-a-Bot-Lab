"""Phase 1 (link manager): pair or re-pair Joy-Cons without opening Windows Settings.

    .venv\\Scripts\\python.exe scripts\\joycon_pair.py LR             # pair both (re-pair if already paired)
    .venv\\Scripts\\python.exe scripts\\joycon_pair.py R --keep        # only pair if not paired yet

Steps per side: forget the old pairing -> ask for the SYNC button -> find the
Joy-Con in SYNC mode -> pair -> wait until it shows up in HID enumeration.
First-time pairing stores the MAC in config/joycon_link.json; after that only
that address is accepted. Does not open HID, so the app may keep running.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.btlink.winbt import CONFIG_PATH, BtLink, find_pairable, load_macs, pair, unpair  # noqa: E402
from teachabot.input.joycon_device import PRODUCT_IDS, enumerate_joycons  # noqa: E402

FIND_TIMEOUT_S = 90.0
HID_TIMEOUT_S = 20.0


def save_mac(side: str, mac: str) -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    cfg.setdefault("joycons", {})[side] = mac
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


async def run(sides: list[str], keep: bool, verbose: bool) -> int:
    macs = load_macs() if CONFIG_PATH.exists() else {}
    todo = set()
    for s in sides:
        state = None
        if s in macs:
            link = BtLink(s, macs[s])
            state = await link.open()
            link.close()
        if keep and state is not None and state.paired:
            print(f"{s}: 이미 페어링됨 (--keep), 건너뜀")
            continue
        if state is not None and state.paired:
            print(f"{s}: 기존 페어링 해제 ... {await unpair(macs[s])}")
        todo.add(s)
    if not todo:
        return 0

    names = " / ".join(sorted(todo))
    print(f"\n>>> {names} 조이콘 옆면(레일)의 작은 SYNC 버튼을 한 번 누르세요. 불이 좌우로 흐르면 됩니다.")
    t0 = time.perf_counter()
    seen: set[tuple[str, str]] = set()

    def on_seen(name: str, mac: str) -> None:
        if verbose and (name, mac) not in seen:
            seen.add((name, mac))
            print(f"  (보임 +{time.perf_counter() - t0:.1f}s) {mac} {name!r}")

    # Windows lists a remembered Joy-Con even when it is not in SYNC mode, so a
    # pairing attempt can fail until the button is pressed: retry until the deadline.
    remaining, paired = set(todo), set()
    deadline = t0 + FIND_TIMEOUT_S
    while remaining and time.perf_counter() < deadline:
        found = await find_pairable(remaining, deadline - time.perf_counter(), macs, on_seen)
        for target in sorted(found, key=lambda p: p.side):
            result = await pair(target)
            print(f"{target.side}: {target.mac} 페어링 시도 (+{time.perf_counter() - t0:.1f}s) -> {result}")
            if result == "ok":
                remaining.discard(target.side)
                paired.add(target.side)
                if target.side not in macs and target.mac:
                    save_mac(target.side, target.mac)
        if remaining:
            await asyncio.sleep(1.0)
    failed = len(remaining)
    for s in sorted(remaining):
        print(f"{s}: {FIND_TIMEOUT_S:.0f}초 안에 페어링 못 함 (SYNC 불이 좌우로 흐르는지 확인)")

    deadline = time.perf_counter() + HID_TIMEOUT_S
    while paired and time.perf_counter() < deadline:
        present = {d["product_id"] for d in enumerate_joycons()}
        for s in sorted(paired):
            if PRODUCT_IDS[s] in present:
                print(f"{s}: HID 준비 완료 (+{time.perf_counter() - t0:.1f}s)")
                paired.discard(s)
        await asyncio.sleep(0.5)
    for s in paired:
        print(f"{s}: 페어링은 됐지만 {HID_TIMEOUT_S:.0f}초 안에 HID 목록에 안 나타남")
        failed += 1
    return 1 if failed else 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sides", nargs="+", help="L, R, or LR for both")
    p.add_argument("--keep", action="store_true", help="skip sides that are already paired")
    p.add_argument("--verbose", action="store_true", help="print every unpaired device seen")
    args = p.parse_args()
    sides = [c for word in args.sides for c in word.upper()]
    if not sides or set(sides) - {"L", "R"}:
        p.error("sides must be L, R or LR")
    sys.exit(asyncio.run(run(list(dict.fromkeys(sides)), args.keep, args.verbose)))


if __name__ == "__main__":
    main()
