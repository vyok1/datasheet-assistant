
import sys
from pathlib import Path

import pymupdf
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "corpus.yaml"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "pages" / "rendered"


SCALE = 1.3


def main():
    if not CONFIG_PATH.is_file():
        print(f"[ERROR] Configuration not found: {CONFIG_PATH}")
        return 1

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    documents = config.get("documents", [])
    if not documents:
        print("[ERROR] No documents configured.")
        return 1

    rendered_count = 0

    for item in documents:
        doc_id = item.get("id")
        selected_pages = item.get("pages")
        pdf_path = RAW_DIR / f"{doc_id}.pdf"

        if not selected_pages:
            print(f"[SKIP] {doc_id}: no pages selected.")
            continue

        if not pdf_path.is_file():
            print(f"[SKIP] {doc_id}: PDF not found at {pdf_path}")
            continue

        if (
            not isinstance(selected_pages, list)
            or any(type(page) is not int or page < 1 for page in selected_pages)
        ):
            print(f"[ERROR] {doc_id}: pages must be a list of positive integers.")
            return 1

        try:
            with pymupdf.open(pdf_path) as pdf:
                for page_number in selected_pages:
                    if page_number > len(pdf):
                        print(
                            f"[ERROR] {doc_id}: page {page_number} is outside "
                            f"the PDF's {len(pdf)} pages."
                        )
                        return 1

                    page = pdf[page_number - 1]
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(SCALE, SCALE),
                        alpha=False,
                    )

                    page_dir = OUTPUT_DIR / doc_id
                    page_dir.mkdir(parents=True, exist_ok=True)
                    output_path = page_dir / f"page_{page_number:04d}.png"
                    pixmap.save(output_path)

                    rendered_count += 1
                    print(f"[OK] {doc_id} page {page_number} -> {output_path}")

        except Exception as exc:
            print(f"[ERROR] Could not process {doc_id}: {exc}")
            return 1

    print(f"[DONE] Rendered {rendered_count} page(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
