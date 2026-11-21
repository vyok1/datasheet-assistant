"""Tests for scripts/chunk_pages.py. Only temporary files are used; nothing in data/ is touched."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chunk_pages.py"
spec = importlib.util.spec_from_file_location("chunk_pages", SCRIPT)
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)

PARAMS = [(1000, 150), (200, 50), (120, 0), (60, 30), (40, 10), (10, 5), (1, 0)]
WORD_SAFE_PARAMS = [(200, 50), (120, 0), (60, 30)]  # limit >= longest word in SAMPLE


def words(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def wrap(ws, per_line=9):
    return "\n".join(" ".join(ws[i:i + per_line]) for i in range(0, len(ws), per_line))


SAMPLE = (
    "RM0090 Rev 22\n269/1741\n"
    "8.1 \nFirst section\n" + wrap(words("alpha", 60)) + "\n"
    "Figure 3. A caption line\n" + wrap(words("beta", 30)) + "\n"
    "8.2 Second section title\n" + wrap(words("gamma", 80)) + "\n"
    "  \n\n   trailing   \n"
)


def check_invariants(text, chunks, max_chars, overlap):
    assert chunks, "expected at least one chunk"
    covered = [False] * len(text)
    for k, c in enumerate(chunks):
        s, e = c["start"], c["end"]
        piece = text[s:e]
        assert 0 <= s < e <= len(text)
        assert piece == piece.strip() and piece != ""
        assert len(piece) <= max_chars
        assert c["overlap_chars"] <= overlap
        for i in range(s, e):
            covered[i] = True
        if k == 0:
            assert c["overlap_chars"] == 0
            continue
        prev = chunks[k - 1]
        assert s >= prev["start"] and e > prev["end"]
        if c["overlap_chars"] > 0:
            ov = c["overlap_chars"]
            assert s + ov == prev["end"]                       # overlap ends where previous chunk ends
            assert text[s:prev["end"]] == text[prev["start"]:prev["end"]][-ov:]  # verbatim tail of previous
            assert s == 0 or text[s - 1].isspace()             # starts at a word start
        else:
            assert s >= prev["end"]
    for i, ch in enumerate(text):
        if not ch.isspace():
            assert covered[i], f"character at offset {i} ({ch!r}) is in no chunk"


# ------------------------------------------------------------------ chunking behaviour

def test_short_page_is_one_chunk_with_trimmed_offsets():
    text = "  \nHello world.\nSecond line.  \n\n"
    (c,) = cp.chunk_page_text(text, 1000, 150)
    assert text[c["start"]:c["end"]] == "Hello world.\nSecond line."
    assert c["overlap_chars"] == 0 and c["section_heading"] is None


@pytest.mark.parametrize("max_chars,overlap", PARAMS)
def test_offsets_overlap_size_and_coverage_hold_for_many_settings(max_chars, overlap):
    chunks = cp.chunk_page_text(SAMPLE, max_chars, overlap)
    check_invariants(SAMPLE, chunks, max_chars, overlap)


def test_small_limit_really_splits_into_many_chunks():
    assert len(cp.chunk_page_text(SAMPLE, 200, 50)) > 5


@pytest.mark.parametrize("max_chars,overlap", WORD_SAFE_PARAMS)
def test_no_word_is_cut_when_words_fit(max_chars, overlap):
    for c in cp.chunk_page_text(SAMPLE, max_chars, overlap):
        assert c["start"] == 0 or SAMPLE[c["start"] - 1].isspace()
        assert c["end"] == len(SAMPLE) or SAMPLE[c["end"]].isspace()


def test_overlap_zero_means_no_shared_text():
    chunks = cp.chunk_page_text(SAMPLE, 120, 0)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur["overlap_chars"] == 0 and cur["start"] >= prev["end"]


def test_positive_overlap_is_actually_produced():
    chunks = cp.chunk_page_text(SAMPLE, 200, 50)
    assert any(c["overlap_chars"] > 0 for c in chunks[1:])
    assert all(c["overlap_chars"] <= 50 for c in chunks)


def test_small_blocks_are_packed_together():
    text = "First paragraph line.\nSecond paragraph line.\nThird one."
    assert len(cp.chunk_page_text(text, 1000, 100)) == 1


def test_section_heading_starts_new_chunk_when_previous_is_half_full():
    sec1 = "1.1 \nAlpha section\n" + "a" * 600
    sec2 = "\n1.2 \nBeta section\n" + "b" * 300
    chunks = cp.chunk_page_text(sec1 + sec2, 1000, 0)   # 900+ chars would fit in one chunk
    assert len(chunks) == 2
    page = sec1 + sec2
    assert page[chunks[1]["start"]:].startswith("1.2")
    assert [c["section_heading"] for c in chunks] == ["1.1 Alpha section", "1.2 Beta section"]


def test_short_sections_are_merged_not_split():
    page = "1.1 \nAlpha section\n" + "a" * 40 + "\n1.2 \nBeta section\n" + "b" * 40
    chunks = cp.chunk_page_text(page, 1000, 0)
    assert len(chunks) == 1 and chunks[0]["section_heading"] == "1.1 Alpha section"


def test_section_heading_follows_new_content_not_overlap():
    sec1 = "1.1 \nAlpha section\n" + wrap(words("a", 70))
    sec2 = "\n1.2 \nBeta section\n" + wrap(words("b", 70))
    page_text = sec1 + sec2
    chunks = cp.chunk_page_text(page_text, 500, 100)
    check_invariants(page_text, chunks, 500, 100)
    second = next(c for c in chunks if c["section_heading"] == "1.2 Beta section")
    assert second["overlap_chars"] > 0
    assert second["start"] < page_text.index("1.2")   # the chunk begins inside section 1.1 ...
    assert chunks[0]["section_heading"] == "1.1 Alpha section"  # ... but is labelled by its new content


def test_first_heading_inside_chunk_is_used_when_only_running_header_precedes_it():
    page_text = "RM0090 Rev 22\n269/1741\n8.1 \nGPIO introduction\nbody text"
    assert cp.chunk_page_text(page_text, 1000, 100)[0]["section_heading"] == "8.1 GPIO introduction"


@pytest.mark.parametrize("text,expected", [
    ("RM0090 Rev 22\n269/1741\n8.1 \nGPIO introduction\nEach port has registers.", "8.1 GPIO introduction"),
    ("8.3 GPIO functional description\nSubject to the specific hardware", "8.3 GPIO functional description"),
    ("269/1741\n290\n8 \nGeneral-purpose I/Os (GPIO)\nThis section applies", None),  # chapter-only: not detected
    ("MOSI\nMISO\n0\n1\nNSS\n0\nDFF\nSSOE", None),                                   # diagram labels
    ("Figure 32. DMA block diagram\nThe DMA controller performs", None),            # caption is not a heading
])
def test_section_heading_detection(text, expected):
    assert cp.chunk_page_text(text, 1000, 100)[0]["section_heading"] == expected


def test_long_line_without_spaces_is_hard_split_and_still_consistent():
    text = "x" * 250
    chunks = cp.chunk_page_text(text, 100, 20)
    check_invariants(text, chunks, 100, 20)
    assert len(chunks) > 2


def test_long_line_with_spaces_splits_at_whitespace():
    text = " ".join(words("w", 120))
    chunks = cp.chunk_page_text(text, 100, 0)
    assert len(chunks) > 3
    for c in chunks:
        assert c["start"] == 0 or text[c["start"] - 1] == " "


@pytest.mark.parametrize("text", ["", "   ", "\n\n \t\n"])
def test_empty_or_whitespace_page_gives_no_chunks(text):
    assert cp.chunk_page_text(text, 100, 10) == []


def test_unicode_bullets_and_mojibake_are_preserved_verbatim():
    text = "Features\n\u2022 Up to 16 I/Os\n\u00e2\u20ac\u00a2 odd bytes stay as they are\n\u03a9 ohms"
    chunks = cp.chunk_page_text(text, 1000, 100)
    assert text[chunks[0]["start"]:chunks[-1]["end"]] == text
    assert "\u2022" in text[chunks[0]["start"]:chunks[0]["end"]]


@pytest.mark.parametrize("max_chars,overlap", [(0, 0), (-5, 0), (100, -1), (100, 51), (10.5, 0), (True, 0), ("100", 0)])
def test_invalid_parameters_are_rejected(max_chars, overlap):
    with pytest.raises(cp.ChunkingError):
        cp.chunk_page_text("text", max_chars, overlap)


def test_overlap_of_exactly_half_is_allowed():
    assert cp.chunk_page_text(SAMPLE, 100, 50)


# ------------------------------------------------------------------ run(): files, metadata, ids

def page(doc_id, number, text, has_text=None):
    return {"doc_id": doc_id, "page_number": number, "text": text,
            "text_char_count": len(text), "has_text": bool(text.strip()) if has_text is None else has_text}


def doc(doc_id, pages, status="ok", **extra):
    return {"id": doc_id, "status": status, "pages": pages, **extra}


def write_input(tmp_path, documents):
    path = tmp_path / "extracted.json"
    path.write_text(json.dumps({"pymupdf_version": "x", "documents": documents}), encoding="utf-8")
    return path


def run_ok(tmp_path, documents, max_chars=200, overlap=50):
    src = write_input(tmp_path, documents)
    out = tmp_path / "out" / "chunks.json"
    summary = cp.run(src, out, max_chars, overlap)
    return json.loads(out.read_text(encoding="utf-8")), summary, src, out


def test_metadata_ids_and_source_page_are_preserved(tmp_path):
    docs = [doc("d1", [page("d1", 7, SAMPLE), page("d1", 3, "Short page text.")])]
    data, summary, _, _ = run_ok(tmp_path, docs)
    chunks = data["chunks"]
    assert [c["page_number"] for c in chunks][:1] == [3]          # pages sorted ascending
    assert chunks[0]["chunk_id"] == "d1-p0003-c000"
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    by_page = {}
    for c in chunks:
        assert c["doc_id"] == "d1"
        assert c["char_count"] == len(c["text"]) == c["end_char"] - c["start_char"]
        assert c["text"] == (SAMPLE if c["page_number"] == 7 else "Short page text.")[c["start_char"]:c["end_char"]]
        assert c["text_sha256"] == cp.sha256_bytes(c["text"].encode("utf-8"))
        by_page.setdefault(c["page_number"], []).append(c["chunk_index"])
    assert all(idx == list(range(len(idx))) for idx in by_page.values())
    meta = data["chunking"]
    assert meta["max_chars"] == 200 and meta["overlap_chars"] == 50
    assert meta["strategy"] == cp.STRATEGY and "text == page_text" in meta["offset_convention"]
    assert summary["chunk_count"] == len(chunks) == meta["summary"]["chunk_count"]


def test_output_is_deterministic_and_has_no_timestamp(tmp_path):
    docs = [doc("d1", [page("d1", 1, SAMPLE)])]
    _, _, src, out = run_ok(tmp_path, docs)
    first = out.read_bytes()
    cp.run(src, out, 200, 50)
    assert out.read_bytes() == first
    assert "generated_at" not in first.decode("utf-8")


def test_ids_stay_stable_when_unrelated_pages_are_added(tmp_path):
    a, _, _, _ = run_ok(tmp_path, [doc("d1", [page("d1", 5, SAMPLE)])])
    (tmp_path / "second").mkdir()
    b, _, _, _ = run_ok(tmp_path / "second", [doc("d1", [page("d1", 5, SAMPLE), page("d1", 9, "More text.")])])
    ids_a = [c["chunk_id"] for c in a["chunks"]]
    assert [c["chunk_id"] for c in b["chunks"] if c["page_number"] == 5] == ids_a


def test_blank_pages_and_missing_pdf_documents_are_skipped_and_reported(tmp_path):
    docs = [
        doc("d1", [page("d1", 1, "Real text here."), page("d1", 2, "   \n"), page("d1", 3, "x", has_text=False)]),
        doc("esp32_datasheet", [], status="missing_pdf", pdf_page_count=None),
        doc("other", [], status="no_valid_pages"),
    ]
    data, summary, _, _ = run_ok(tmp_path, docs)
    assert summary["pages_chunked"] == 1 and summary["pages_without_text"] == 2
    assert summary["documents_chunked"] == 1
    assert summary["documents_skipped"] == [{"id": "esp32_datasheet", "status": "missing_pdf"},
                                            {"id": "other", "status": "no_valid_pages"}]
    assert {c["page_number"] for c in data["chunks"]} == {1}


# ------------------------------------------------------------------ errors and malformed input

def expect_error(tmp_path, documents, pattern, raw=None):
    src = tmp_path / "extracted.json"
    src.write_text(raw if raw is not None else json.dumps({"documents": documents}), encoding="utf-8")
    out = tmp_path / "chunks.json"
    out.write_text("PREVIOUS OUTPUT", encoding="utf-8")
    with pytest.raises(cp.ChunkingError, match=pattern):
        cp.run(src, out, 200, 50)
    assert out.read_text(encoding="utf-8") == "PREVIOUS OUTPUT"  # nothing written on failure
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_input_file_is_a_clear_error(tmp_path):
    with pytest.raises(cp.ChunkingError, match="input file not found"):
        cp.run(tmp_path / "nope.json", tmp_path / "chunks.json", 200, 50)
    assert not (tmp_path / "chunks.json").exists()


def test_invalid_json(tmp_path):
    expect_error(tmp_path, None, "not valid", raw="{ this is not json")


@pytest.mark.parametrize("raw", ["[]", '{"documents": "x"}', '{"other": 1}'])
def test_wrong_top_level_shape(tmp_path, raw):
    expect_error(tmp_path, None, "'documents' list", raw=raw)


@pytest.mark.parametrize("mutate,pattern", [
    (lambda p: p.pop("text"), r"'text' must be a string"),
    (lambda p: p.update(text=123), r"'text' must be a string"),
    (lambda p: p.update(page_number="4"), r"invalid page_number"),
    (lambda p: p.update(page_number=0), r"invalid page_number"),
    (lambda p: p.update(page_number=True), r"invalid page_number"),
    (lambda p: p.update(doc_id="other"), r"does not match document id"),
    (lambda p: p.update(has_text="yes"), r"'has_text' must be a boolean"),
    (lambda p: p.update(text_char_count=5), r"does not match text length"),
    (lambda p: p.pop("text_char_count"), r"does not match text length"),
])
def test_malformed_page_records_stop_with_a_clear_error(tmp_path, mutate, pattern):
    p = page("d1", 4, "Some text on the page.")
    mutate(p)
    expect_error(tmp_path, [doc("d1", [p])], pattern)


def test_duplicate_page_and_duplicate_document(tmp_path):
    expect_error(tmp_path, [doc("d1", [page("d1", 2, "a"), page("d1", 2, "b")])], "duplicate page_number 2")
    expect_error(tmp_path, [doc("d1", []), doc("d1", [])], "duplicate document id")


@pytest.mark.parametrize("document,pattern", [
    ({"status": "ok", "pages": []}, "no valid 'id'"),
    ("not an object", "not an object"),
    ({"id": "d1", "status": "ok", "pages": "x"}, "'pages' must be a list"),
    ({"id": "d1", "status": "weird", "pages": []}, "unknown status"),
    ({"id": "d1", "status": "error", "error": "FileDataError: boom", "pages": []}, "failed extraction: FileDataError: boom"),
])
def test_malformed_documents(tmp_path, document, pattern):
    expect_error(tmp_path, [document], pattern)


def test_main_returns_1_with_message_on_error_and_0_on_success(tmp_path, capsys):
    assert cp.main(["--input", str(tmp_path / "missing.json"), "--output", str(tmp_path / "o.json")]) == 1
    assert "[ERROR] input file not found" in capsys.readouterr().out

    src = write_input(tmp_path, [doc("d1", [page("d1", 1, SAMPLE)]), doc("e", [], status="missing_pdf")])
    out = tmp_path / "o.json"
    assert cp.main(["--input", str(src), "--output", str(out), "--max-chars", "300", "--overlap", "60"]) == 0
    printed = capsys.readouterr().out
    assert "[SKIP] e: status missing_pdf" in printed and "[ OK ]" in printed
    assert json.loads(out.read_text(encoding="utf-8"))["chunking"]["max_chars"] == 300


def test_main_rejects_bad_parameters_with_exit_1(tmp_path, capsys):
    src = write_input(tmp_path, [doc("d1", [page("d1", 1, "text")])])
    assert cp.main(["--input", str(src), "--output", str(tmp_path / "o.json"), "--max-chars", "100", "--overlap", "80"]) == 1
    assert "at most half" in capsys.readouterr().out