"""Read-only: summary stats from data/pages/inventory.json plus the PDF's top-level outline.

Changes nothing. Run from the repo root:  python scripts/inspect_inventory_stats.py
"""
import json
import statistics
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data" / "pages" / "inventory.json"
LOW_TEXT = 200      # pages with fewer characters than this are "low text"
MAX_LIST = 15       # cap every printed list


def head(items):
    items = list(items)
    more = f" ... (+{len(items) - MAX_LIST} more)" if len(items) > MAX_LIST else ""
    return str(items[:MAX_LIST]) + more


def main() -> None:
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    for doc in inv["documents"]:
        pages = doc["pages"]
        print(f"=== {doc['id']}: {doc['page_count']} pages ===")

        chars = [p["text_char_count"] for p in pages]
        print("\n[text characters per page]")
        print(f"  min={min(chars)}  median={statistics.median(chars):.0f}  max={max(chars)}")
        no_text = [p["page_number"] for p in pages if not p["has_text"]]
        print(f"  has_text == False: {len(no_text)} pages {head(no_text)}")
        low = [p["page_number"] for p in pages if p["text_char_count"] < LOW_TEXT]
        print(f"  under {LOW_TEXT} chars: {len(low)} pages {head(low)}")

        print("\n[embedded images per page]")
        hist = Counter(min(p["image_count"], 3) for p in pages)
        print(f"  0 images: {hist[0]}   1: {hist[1]}   2: {hist[2]}   3+: {hist[3]}")
        top = sorted(pages, key=lambda p: p["image_count"], reverse=True)[:10]
        print("  top pages by image_count:", [(p["page_number"], p["image_count"]) for p in top])

        print("\n[page sizes]")
        sizes = Counter((p["width"], p["height"]) for p in pages)
        for (w, h), n in sizes.most_common(5):
            print(f"  {w} x {h}: {n} pages")

        pdf_path = ROOT / Path(doc["file"].replace("\\", "/"))
        with pymupdf.open(pdf_path) as pdf:
            toc = pdf.get_toc()  # entries: [level, title, page]
        top_level = [e for e in toc if e[0] == 1]
        print(f"\n[outline] {len(toc)} entries total, {len(top_level)} at level 1")
        for _, title, page in top_level[:80]:
            print(f"  p{page:>5}  {title}")
        if len(top_level) > 80:
            print(f"  ... (+{len(top_level) - 80} more)")


if __name__ == "__main__":
    main()