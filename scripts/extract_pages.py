"""Extract text from the configured pages of each available PDF.

Reads configs/corpus.yaml, writes data/pages/extracted.json.
Page numbers are 1-based PDF page numbers (same as page_number in inventory.json).

`pages:` in corpus.yaml may be:
  - null / missing   -> every page
  - an int           -> pages: 112
  - a string         -> pages: "10-20"  or  "1-5, 12, 30-40"
  - a list of these  -> pages: [112, "200-210"]

Run from the repo root:  python scripts/extract_pages.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pymupdf
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "configs" / "corpus.yaml"
OUT = ROOT / "data" / "pages" / "extracted.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}-{hi}"


def parse_token(token) -> tuple:
    """One int, "12" or "10-20" -> inclusive (lo, hi). Raises ValueError if malformed."""
    if isinstance(token, bool):
        raise ValueError(f"invalid page entry: {token!r}")
    if isinstance(token, int):
        return token, token
    if isinstance(token, str):
        text = token.strip()
        try:
            if "-" in text[1:]:  # [1:] so a leading minus is not mistaken for a range
                lo_s, hi_s = text.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
            else:
                lo = hi = int(text)
        except ValueError:
            raise ValueError(f"invalid page entry: {token!r}") from None
        if lo > hi:
            raise ValueError(f"range start is after range end: {token!r}")
        return lo, hi
    raise ValueError(f"invalid page entry: {token!r}")


def parse_spec(spec, page_count: int) -> list:
    """Config value -> list of inclusive (lo, hi) ranges. None means the whole document."""
    if spec is None:
        return [(1, page_count)]
    if isinstance(spec, (int, bool)):
        return [parse_token(spec)]
    if isinstance(spec, str):
        parts = [p for p in spec.split(",")]
        if not any(p.strip() for p in parts):
            raise ValueError("pages is an empty string")
        return [parse_token(p) for p in parts if p.strip()]
    if isinstance(spec, list):
        if not spec:
            raise ValueError("pages is an empty list")
        ranges = []
        for item in spec:
            ranges.extend(parse_spec(item, page_count) if isinstance(item, str) else [parse_token(item)])
        return ranges
    raise ValueError(f"unsupported pages value: {spec!r}")


def clip(ranges: list, page_count: int) -> tuple:
    """Keep only pages that exist. Returns (sorted unique pages, out-of-range descriptions)."""
    pages, skipped = set(), []
    for lo, hi in ranges:
        if hi < 1 or lo > page_count:
            skipped.append(fmt(lo, hi))
            continue
        if lo < 1:
            skipped.append(fmt(lo, 0))
        if hi > page_count:
            skipped.append(fmt(page_count + 1, hi))
        pages.update(range(max(lo, 1), min(hi, page_count) + 1))
    return sorted(pages), list(dict.fromkeys(skipped))


def process(doc: dict) -> tuple:
    """Returns (document record, failed?). Missing PDFs and bad page numbers do not crash the run."""
    pdf_path = RAW / f"{doc['id']}.pdf"
    record = {
        "id": doc["id"],
        "title": doc.get("title"),
        "file": pdf_path.relative_to(ROOT).as_posix(),
        "status": "ok",
        "requested_pages": doc.get("pages"),
        "pdf_page_count": None,
        "pdf_sha256": None,
        "skipped_pages": [],
        "pages": [],
    }
    if not pdf_path.exists():
        record["status"] = "missing_pdf"
        print(f"[SKIP] {doc['id']}: {pdf_path} not found")
        return record, False

    try:
        record["pdf_sha256"] = sha256(pdf_path)
        with pymupdf.open(pdf_path) as pdf:
            if pdf.needs_pass:
                raise ValueError("PDF is password protected")
            record["pdf_page_count"] = pdf.page_count
            ranges = parse_spec(doc.get("pages"), pdf.page_count)
            selected, skipped = clip(ranges, pdf.page_count)
            record["skipped_pages"] = skipped
            if skipped:
                print(f"[WARN] {doc['id']}: ignored out-of-range pages {skipped} (PDF has {pdf.page_count})")
            for number in selected:
                text = pdf.load_page(number - 1).get_text("text")
                record["pages"].append({
                    "doc_id": doc["id"],
                    "page_number": number,
                    "text": text,
                    "text_char_count": len(text),
                    "has_text": bool(text.strip()),
                })
    except Exception as e:  # malformed spec, corrupt or encrypted PDF, extraction error
        record["status"] = "error"
        # PyMuPDF puts the absolute path in its messages; swap it for the repo-relative one
        # so the output file does not depend on where the project lives.
        message = str(e).replace(str(pdf_path), record["file"])
        record["error"] = f"{type(e).__name__}: {message}"
        record["pages"] = []
        print(f"[FAIL] {doc['id']}: {record['error']}")
        return record, True

    if not record["pages"]:
        record["status"] = "no_valid_pages"
        print(f"[WARN] {doc['id']}: no valid pages selected")
    else:
        empty = sum(1 for p in record["pages"] if not p["has_text"])
        print(f"[ OK ] {doc['id']}: extracted {len(record['pages'])} pages ({empty} without text)")
    return record, False


def main() -> int:
    docs = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["documents"]
    records, failures = [], 0
    for doc in docs:
        record, failed = process(doc)
        records.append(record)
        failures += failed

    # No timestamp: identical inputs must give a byte-identical file.
    payload = {"pymupdf_version": pymupdf.__version__, "documents": records}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())