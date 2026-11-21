"""Download (or adopt) corpus PDFs, validate them, and record SHA-256 in a lockfile.

Run from the repo root:  python scripts/download_corpus.py
Vendor PDFs go to data/raw/ (gitignored). The lockfile configs/corpus.lock.json IS committed.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "configs" / "corpus.yaml"
LOCK = ROOT / "configs" / "corpus.lock.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (datasheet-assistant student project)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_pdf(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(5) == b"%PDF-"


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
    except BaseException:
        # Never leave a partial file behind (network error, disk full, Ctrl+C).
        tmp.unlink(missing_ok=True)
        raise


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    docs = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["documents"]
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    problems = 0

    for doc in docs:
        dest = RAW / f"{doc['id']}.pdf"
        downloaded = False
        if not dest.exists():
            if not doc.get("url"):
                print(f"[SKIP] {doc['id']}: no url and no file. Put the PDF at {dest}")
                problems += 1
                continue
            print(f"[DOWN] {doc['id']} ...")
            try:
                download(doc["url"], dest)
            except requests.RequestException as e:
                print(f"[FAIL] {doc['id']}: {e}\n       Download it in a browser and save as {dest}")
                problems += 1
                continue
            downloaded = True
        if not is_pdf(dest):
            if downloaded:
                # Our own bad download: remove it so the next run retries instead of adopting it.
                dest.unlink(missing_ok=True)
                print(f"[FAIL] {doc['id']}: downloaded file is not a PDF (blocked page or wrong link?); removed it")
            else:
                # Not ours: never delete a file the user placed there.
                print(f"[FAIL] {doc['id']}: {dest} is not a PDF; left in place, delete or replace it manually")
            problems += 1
            continue

        digest = sha256(dest)
        prev = lock.get(doc["id"], {}).get("sha256")
        status = "new" if prev is None else ("unchanged" if prev == digest else "CHANGED")
        lock[doc["id"]] = {
            "title": doc["title"],
            "url": doc.get("url"),
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"[ OK ] {doc['id']}: {dest.stat().st_size/1e6:.1f} MB, sha256={digest[:12]}... ({status})")

    LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())