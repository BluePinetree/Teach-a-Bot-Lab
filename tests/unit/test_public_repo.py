"""Nothing private goes into the tracked files of this repository.

The checks run on `git ls-files`, so a file that is ignored (local data, the Joy-Con
pairing file with its MAC addresses) never trips them; a file that gets added does.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 5_000_000
MAC = re.compile(rb"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
USER_PATH = re.compile(rb"[A-Za-z]:\\Users\\(?!<)[^\\%<>\s`'\"]+", re.IGNORECASE)
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ALLOWED_EMAILS = (b"noreply@anthropic.com", b"users.noreply.github.com", b"example.com")


def _tracked() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / p.decode("utf-8") for p in out.split(b"\0") if p]


def test_no_joycon_pairing_file_is_tracked():
    assert ROOT / "config" / "joycon_link.json" not in _tracked()


def test_tracked_files_hold_no_mac_address_user_path_or_private_email():
    hits = []
    for path in _tracked():
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".npz", ".zip"}:
            continue
        data = path.read_bytes()
        for name, rx in (("MAC address", MAC), ("user path", USER_PATH)):
            hits += [f"{path.relative_to(ROOT)}: {name} {m.group().decode(errors='replace')}" for m in rx.finditer(data)]
        hits += [f"{path.relative_to(ROOT)}: email {m.group().decode(errors='replace')}" for m in EMAIL.finditer(data)
                 if not m.group().endswith(ALLOWED_EMAILS)]
    assert not hits, "\n".join(hits)


def test_no_large_files_are_tracked():
    big = [f"{p.relative_to(ROOT)} {p.stat().st_size} B" for p in _tracked() if p.is_file() and p.stat().st_size > MAX_BYTES]
    assert not big, "large files go to a release asset, not git:\n" + "\n".join(big)
