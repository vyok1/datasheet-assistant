
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "corpus.yaml"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "pages" / "inventory.json"


def main():
    if not CONFIG_PATH.exists():
        print(f"[ERROR] Configuration not found: {CONFIG_PATH}")
        return 1

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    documents = config.get("documents", [])
    inventory = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": [],
    }

    processed = 0

    for item in documents:
        doc_id = item["id"]
        pdf_path = RAW_DIR / f"{doc_id}.pdf"

        if not pdf_path.is_file():
            print(f"[SKIP] {doc_id}: PDF not found at {pdf_path}")
            continue

        try:
            with pymupdf.open(pdf_path) as pdf:
                pages = []

                for index, page in enumerate(pdf):
                    text = page.get_text("text").strip()

                    pages.append({
                        "page_number": index + 1,
                        "width": round(page.rect.width, 2),
                        "height": round(page.rect.height, 2),
                        "text_char_count": len(text),
                        "has_text": bool(text),
                        "image_count": len(page.get_images()),
                        "text_preview": " ".join(text.split())[:250],
                    })

                inventory["documents"].append({
                    "id": doc_id,
                    "title": item.get("title", doc_id),
                    "file": str(pdf_path.relative_to(ROOT)),
                    "page_count": len(pdf),
                    "pages": pages,
                })

                processed += 1
                print(
                    f"[OK] {doc_id}: {len(pdf)} pages inventoried"
                )

        except Exception as exc:
            print(f"[ERROR] Could not read {doc_id}: {exc}")
            return 1

    if processed == 0:
        print("[ERROR] No PDFs were inventoried.")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[SAVED] {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"[DONE] Inventoried {processed} document(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
