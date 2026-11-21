"""Split extracted page text into chunks.  No embeddings, no retrieval.

Reads  data/pages/extracted.json   (written by scripts/extract_pages.py)
Writes data/pages/chunks.json

Strategy "page-bounded-structure-aware-v1"
  1. Chunks never cross a page, so page_number is exact for citations.
  2. A page is cut into blocks at numbered section headings ("8.1" + title, on one line or
     on two) and at "Figure N." / "Table N." caption lines. These patterns are heuristics:
     a wrong guess only moves a preferred split point, it never changes or loses text.
  3. Blocks are packed greedily into chunks. A block that does not fit is split at line
     boundaries; a line that does not fit is split at whitespace (mid-word only if a single
     word is longer than the limit).
  4. A chunk is closed before a section heading once it is at least half full.
  5. Overlap: every chunk after the first on a page starts with up to --overlap characters
     taken from the end of the previous chunk, moved forward to a word start. To keep
     chunks within --max-chars, new content per chunk is limited to max_chars - overlap.

Offset convention (explicit)
  start_char / end_char are 0-based, end-exclusive indices into the ORIGINAL page text
  (the "text" string of that page in extracted.json, Python code-point indices), and
  chunk["text"] == page_text[start_char:end_char] exactly. Leading and trailing whitespace
  is excluded from the range; nothing inside the range is modified.
  For chunk k > 0, overlap_chars is the number of characters shared with chunk k-1:
  page_text[start_char : start_char + overlap_chars] is the end of chunk k-1's text.
  (If the previous chunk is shorter than the overlap, the whole previous chunk is repeated.)
  section_heading is the numbered heading in effect where the chunk's new content starts, or
  the first heading inside the chunk if none has appeared yet on the page, else null.

Run from the repo root:  python scripts/chunk_pages.py [--max-chars 1000] [--overlap 150]
"""
import argparse
import bisect
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "pages" / "extracted.json"
DEFAULT_OUTPUT = ROOT / "data" / "pages" / "chunks.json"
DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 150
STRATEGY = "page-bounded-structure-aware-v1"
OFFSET_CONVENTION = (
    "start_char/end_char: 0-based, end-exclusive code-point indices into the original page "
    "text in extracted.json; text == page_text[start_char:end_char]"
)

HEADING_NUMBER_ONLY = re.compile(r"^\s*\d+(?:\.\d+)+\s*$")   # "8.1 " (title on the next line)
HEADING_ONE_LINE = re.compile(r"^\s*\d+(?:\.\d+)+\s+[A-Z]")  # "8.3 GPIO functional description"
CAPTION = re.compile(r"^\s*(?:Figure|Table)\s+\d+[.:]")


class ChunkingError(Exception):
    """Bad parameters, missing input, or a malformed record."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ChunkingError(message)


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------- chunking one page

def validate_params(max_chars, overlap_chars) -> None:
    require(is_int(max_chars) and max_chars >= 1, f"max_chars must be an integer >= 1, got {max_chars!r}")
    require(is_int(overlap_chars) and overlap_chars >= 0,
            f"overlap must be an integer >= 0, got {overlap_chars!r}")
    require(overlap_chars <= max_chars // 2,
            f"overlap ({overlap_chars}) must be at most half of max_chars ({max_chars})")


def trim(text: str, start: int, end: int) -> tuple:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def line_spans(text: str) -> list:
    """(start, end) of every line, excluding the "\\n" itself."""
    spans, pos = [], 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def find_boundaries(text: str, lines: list) -> list:
    """Returns (offset, heading_text_or_None, is_section_heading), sorted by offset."""
    found, i = [], 0
    while i < len(lines):
        s, e = lines[i]
        line = text[s:e]
        if HEADING_NUMBER_ONLY.match(line) and i + 1 < len(lines):
            ns, ne = lines[i + 1]
            title = text[ns:ne].strip()
            if len(title) >= 3 and title[0].isupper():
                found.append((s, " ".join((line + " " + title).split()), True))
                i += 2
                continue
        if HEADING_ONE_LINE.match(line):
            found.append((s, " ".join(line.split()), True))
        elif CAPTION.match(line):
            found.append((s, None, False))
        i += 1
    return found


def split_long(text: str, start: int, end: int, limit: int) -> list:
    """Split [start, end) into trimmed spans of at most `limit` chars, at whitespace if possible."""
    spans, pos = [], start
    while end - pos > limit:
        cut = pos + limit  # text[cut] exists because end - pos > limit
        piece_end = cut
        for i in range(cut, pos, -1):
            if text[i].isspace():
                piece_end = i
                break
        a, b = trim(text, pos, piece_end)
        if a < b:
            spans.append((a, b))
        pos = piece_end
        while pos < end and text[pos].isspace():
            pos += 1
    a, b = trim(text, pos, end)
    if a < b:
        spans.append((a, b))
    return spans


def block_units(text: str, a: int, b: int, limit: int) -> list:
    """Units (spans of at most `limit` chars) for one trimmed block [a, b)."""
    if b - a <= limit:
        return [(a, b)]
    units, pos = [], a
    for line in text[a:b].split("\n"):
        ls, le = pos, pos + len(line)
        pos = le + 1
        ta, tb = trim(text, ls, le)
        if ta >= tb:
            continue
        if tb - ta <= limit:
            units.append((ta, tb))
        else:
            units.extend(split_long(text, ta, tb, limit))
    return units


def word_start(text: str, i: int) -> bool:
    return i == 0 or (text[i - 1].isspace() and not text[i].isspace())


def chunk_page_text(text: str, max_chars: int = DEFAULT_MAX_CHARS,
                    overlap_chars: int = DEFAULT_OVERLAP) -> list:
    """Chunk one page. Returns dicts with start, end, overlap_chars, section_heading."""
    validate_params(max_chars, overlap_chars)
    limit = max_chars - overlap_chars  # room for new content in each chunk

    lines = line_spans(text)
    boundaries = find_boundaries(text, lines)
    headings = [(o, h) for o, h, is_heading in boundaries if is_heading]
    heading_offsets = [o for o, _ in headings]
    section_starts = set(heading_offsets)

    starts = sorted({0} | {o for o, _, _ in boundaries})
    ends = starts[1:] + [len(text)]
    units = []  # (start, end, begins_a_section)
    for bs, be in zip(starts, ends):
        a, b = trim(text, bs, be)
        if a >= b:
            continue
        for k, (ua, ub) in enumerate(block_units(text, a, b, limit)):
            units.append((ua, ub, bs in section_starts and k == 0))

    # Greedy packing of new content.
    bases, cur_s, cur_e = [], None, None
    for ua, ub, begins_section in units:
        if cur_s is None:
            cur_s, cur_e = ua, ub
        elif begins_section and cur_e - cur_s >= limit // 2:
            bases.append((cur_s, cur_e))
            cur_s, cur_e = ua, ub
        elif ub - cur_s <= limit:
            cur_e = ub
        else:
            bases.append((cur_s, cur_e))
            cur_s, cur_e = ua, ub
    if cur_s is not None:
        bases.append((cur_s, cur_e))

    # Add overlap, then attach metadata.
    result, prev = [], None
    for s, e in bases:
        d = s
        if prev is not None and overlap_chars > 0:
            d = max(s - overlap_chars, prev[0])
            while d < s and not word_start(text, d):
                d += 1
        overlap = max(0, prev[1] - d) if prev is not None else 0
        # Heading in effect where the chunk's new content starts; if none has appeared yet on
        # this page (e.g. only the running header precedes it), the first heading inside the chunk.
        idx = bisect.bisect_right(heading_offsets, s) - 1
        if idx >= 0:
            heading = headings[idx][1]
        elif headings and heading_offsets[0] < e:
            heading = headings[0][1]
        else:
            heading = None
        result.append({
            "start": d,
            "end": e,
            "overlap_chars": overlap,
            "section_heading": heading,
        })
        prev = (d, e)
    return result


# ---------------------------------------------------------------- reading the input

def load_documents(path: Path) -> tuple:
    """Validate extracted.json. Returns (documents to chunk, skipped documents)."""
    require(path.exists(), f"input file not found: {path} (run scripts/extract_pages.py first)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ChunkingError(f"{path} is not valid UTF-8 JSON: {e}") from None
    require(isinstance(data, dict) and isinstance(data.get("documents"), list),
            f"{path}: expected an object with a 'documents' list")

    documents, skipped, seen_docs = [], [], set()
    for di, doc in enumerate(data["documents"]):
        require(isinstance(doc, dict), f"documents[{di}] is not an object")
        doc_id = doc.get("id")
        require(isinstance(doc_id, str) and doc_id, f"documents[{di}] has no valid 'id'")
        require(doc_id not in seen_docs, f"duplicate document id {doc_id!r}")
        seen_docs.add(doc_id)

        status = doc.get("status")
        if status in ("missing_pdf", "no_valid_pages"):
            skipped.append({"id": doc_id, "status": status})
            continue
        require(status != "error", f"document {doc_id!r} failed extraction: {doc.get('error', 'unknown error')}")
        require(status == "ok", f"document {doc_id!r} has unknown status {status!r}")
        require(isinstance(doc.get("pages"), list), f"document {doc_id!r}: 'pages' must be a list")

        seen_pages, pages = set(), []
        for pi, page in enumerate(doc["pages"]):
            where = f"document {doc_id!r}, pages[{pi}]"
            require(isinstance(page, dict), f"{where}: not an object")
            require(page.get("doc_id") == doc_id,
                    f"{where}: doc_id {page.get('doc_id')!r} does not match document id")
            number = page.get("page_number")
            require(is_int(number) and number >= 1, f"{where}: invalid page_number {number!r}")
            text = page.get("text")
            require(isinstance(text, str), f"{where} (page {number}): 'text' must be a string")
            require(isinstance(page.get("has_text"), bool), f"{where} (page {number}): 'has_text' must be a boolean")
            count = page.get("text_char_count")
            require(is_int(count) and count == len(text),
                    f"{where} (page {number}): text_char_count {count!r} does not match text length {len(text)}")
            require(number not in seen_pages, f"document {doc_id!r}: duplicate page_number {number}")
            seen_pages.add(number)
            pages.append({"page_number": number, "text": text, "has_text": page["has_text"]})
        pages.sort(key=lambda p: p["page_number"])
        documents.append({"id": doc_id, "pages": pages})
    return documents, skipped


# ---------------------------------------------------------------- building the output

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run(input_path: Path, output_path: Path, max_chars: int, overlap_chars: int) -> dict:
    validate_params(max_chars, overlap_chars)
    documents, skipped = load_documents(input_path)

    chunks, pages_chunked, pages_blank = [], 0, 0
    for doc in documents:
        for page in doc["pages"]:
            text = page["text"]
            if not page["has_text"] or not text.strip():
                pages_blank += 1
                continue
            pages_chunked += 1
            for index, c in enumerate(chunk_page_text(text, max_chars, overlap_chars)):
                piece = text[c["start"]:c["end"]]
                chunks.append({
                    "chunk_id": f"{doc['id']}-p{page['page_number']:04d}-c{index:03d}",
                    "doc_id": doc["id"],
                    "page_number": page["page_number"],
                    "chunk_index": index,
                    "text": piece,
                    "char_count": len(piece),
                    "start_char": c["start"],
                    "end_char": c["end"],
                    "overlap_chars": c["overlap_chars"],
                    "section_heading": c["section_heading"],
                    "text_sha256": sha256_bytes(piece.encode("utf-8")),
                })

    summary = {
        "documents_chunked": len(documents),
        "documents_skipped": skipped,
        "pages_chunked": pages_chunked,
        "pages_without_text": pages_blank,
        "chunk_count": len(chunks),
    }
    payload = {
        "chunking": {
            "strategy": STRATEGY,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            "offset_convention": OFFSET_CONVENTION,
            "source": display_path(input_path),
            "source_sha256": sha256_bytes(input_path.read_bytes()),
            "summary": summary,
        },
        "chunks": chunks,
    }
    # Written only after everything succeeded; no timestamp, so output is reproducible.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output_path)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chunk extracted page text (no embeddings).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args(argv)
    try:
        summary = run(args.input, args.output, args.max_chars, args.overlap)
    except ChunkingError as e:
        print(f"[ERROR] {e}")
        return 1
    for doc in summary["documents_skipped"]:
        print(f"[SKIP] {doc['id']}: status {doc['status']}")
    print(f"[ OK ] {summary['chunk_count']} chunks from {summary['pages_chunked']} pages "
          f"({summary['pages_without_text']} pages without text) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())