r"""Download the MuJoCo Menagerie ALOHA 2 model at a pinned commit and verify sha256.

    .venv\Scripts\python.exe scripts\download_aloha.py

Source: github.com/google-deepmind/mujoco_menagerie (aloha/, BSD-3-Clause, ~16 MB).
Files go to assets/third_party/mujoco_menagerie/aloha (not tracked by git).
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "scripts" / "aloha_manifest.json").read_text(encoding="utf-8"))
DEST = ROOT / "assets" / "third_party" / "mujoco_menagerie" / "aloha"


def main() -> int:
    base = f"https://raw.githubusercontent.com/{MANIFEST['repo']}/{MANIFEST['commit']}/{MANIFEST['path']}"
    bad = 0
    for rel, sha in MANIFEST["files"].items():
        out = DEST / rel
        if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == sha:
            continue
        data = urllib.request.urlopen(f"{base}/{rel}").read()
        if hashlib.sha256(data).hexdigest() != sha:
            print("HASH MISMATCH", rel)
            bad += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print("ok", rel)
    (DEST / "SOURCE.txt").write_text(f"{MANIFEST['repo']} @ {MANIFEST['commit']}\npath: {MANIFEST['path']}/\n"
                                     f"license: {MANIFEST['license']} (see LICENSE)\n", encoding="utf-8")
    print("done" if not bad else f"{bad} file(s) failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
