"""Failure-case tests for scripts/download_corpus.py. No network access: requests.get is faked."""
import importlib.util
import json
from pathlib import Path

import pytest
import requests
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_corpus.py"
VALID_PDF = b"%PDF-1.7\n% fake but valid header\n" + b"x" * 100
HTML_PAGE = b"<html><body>Access denied</body></html>"


@pytest.fixture
def dc(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("download_corpus", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Redirect every path into pytest's temp folder so your real data/ and configs/ are never touched.
    monkeypatch.setattr(mod, "RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(mod, "CONFIG", tmp_path / "corpus.yaml")
    monkeypatch.setattr(mod, "LOCK", tmp_path / "corpus.lock.json")
    return mod


def write_config(dc, url="https://example.invalid/a.pdf"):
    cfg = {"documents": [{"id": "doc_a", "title": "Doc A", "url": url, "pages": None}]}
    dc.CONFIG.write_text(yaml.safe_dump(cfg), encoding="utf-8")


class FakeResponse:
    """Context-manager response whose stream yields chunks, then optionally raises."""

    def __init__(self, chunks, fail_after=None, status_error=None):
        self.chunks, self.fail_after, self.status_error = chunks, fail_after, status_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def iter_content(self, chunk_size):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise requests.ConnectionError("connection dropped mid-stream")
            yield c


def test_part_file_removed_when_stream_fails_midway(dc, monkeypatch):
    dc.RAW.mkdir(parents=True)
    dest = dc.RAW / "doc_a.pdf"
    monkeypatch.setattr(
        dc.requests, "get", lambda *a, **k: FakeResponse([b"%PDF-1.7", b"more"], fail_after=1)
    )
    with pytest.raises(requests.ConnectionError):
        dc.download("https://example.invalid/a.pdf", dest)
    assert not dest.with_suffix(".part").exists()
    assert not dest.exists()


def test_part_file_removed_on_http_error(dc, monkeypatch):
    dc.RAW.mkdir(parents=True)
    dest = dc.RAW / "doc_a.pdf"
    err = requests.HTTPError("403 Forbidden")
    monkeypatch.setattr(dc.requests, "get", lambda *a, **k: FakeResponse([], status_error=err))
    with pytest.raises(requests.HTTPError):
        dc.download("https://example.invalid/a.pdf", dest)
    assert list(dc.RAW.iterdir()) == []


def test_main_failed_download_returns_1_and_leaves_no_files(dc, monkeypatch, capsys):
    write_config(dc)
    monkeypatch.setattr(
        dc.requests, "get", lambda *a, **k: FakeResponse([b"%PDF-1", b"x"], fail_after=1)
    )
    assert dc.main() == 1
    assert list(dc.RAW.iterdir()) == []
    assert "[FAIL] doc_a" in capsys.readouterr().out
    assert json.loads(dc.LOCK.read_text()) == {}  # nothing recorded for a failed doc


def test_downloaded_non_pdf_is_removed_and_not_locked(dc, monkeypatch):
    write_config(dc)
    monkeypatch.setattr(dc.requests, "get", lambda *a, **k: FakeResponse([HTML_PAGE]))
    assert dc.main() == 1
    assert not (dc.RAW / "doc_a.pdf").exists()
    assert list(dc.RAW.glob("*.part")) == []
    assert json.loads(dc.LOCK.read_text()) == {}


def test_preexisting_valid_pdf_is_preserved_and_not_redownloaded(dc, monkeypatch):
    write_config(dc)
    dc.RAW.mkdir(parents=True)
    pdf = dc.RAW / "doc_a.pdf"
    pdf.write_bytes(VALID_PDF)

    def boom(*a, **k):
        raise AssertionError("must not download when a file already exists")

    monkeypatch.setattr(dc.requests, "get", boom)
    assert dc.main() == 0
    assert pdf.read_bytes() == VALID_PDF
    lock = json.loads(dc.LOCK.read_text())
    assert lock["doc_a"]["sha256"] == dc.sha256(pdf)
    assert lock["doc_a"]["bytes"] == len(VALID_PDF)


def test_preexisting_invalid_file_is_not_deleted_but_fails(dc, monkeypatch):
    write_config(dc)
    dc.RAW.mkdir(parents=True)
    f = dc.RAW / "doc_a.pdf"
    f.write_bytes(HTML_PAGE)
    assert dc.main() == 1
    assert f.read_bytes() == HTML_PAGE  # user's file untouched


def test_successful_download_still_streams_hashes_and_locks(dc, monkeypatch):
    write_config(dc)
    monkeypatch.setattr(
        dc.requests, "get", lambda *a, **k: FakeResponse([VALID_PDF[:20], VALID_PDF[20:]])
    )
    assert dc.main() == 0
    pdf = dc.RAW / "doc_a.pdf"
    assert pdf.read_bytes() == VALID_PDF
    assert list(dc.RAW.glob("*.part")) == []
    entry = json.loads(dc.LOCK.read_text())["doc_a"]
    assert entry["sha256"] == dc.sha256(pdf)
    assert entry["url"] == "https://example.invalid/a.pdf"


def test_missing_url_and_file_is_skipped_with_exit_1(dc):
    write_config(dc, url=None)
    assert dc.main() == 1