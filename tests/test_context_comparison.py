"""Tests for context expansion evaluation and comparison."""

import importlib.util
import json
from pathlib import Path
import pytest

from datasheet_assistant.evaluation.context_comparison import (
    context_recall,
    context_size,
)
from datasheet_assistant.retrieval.bm25 import BM25Retriever

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_context_expansion.py"
spec = importlib.util.spec_from_file_location("compare_context_expansion", SCRIPT)
cce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cce)


# -----------------------------------------------------------------------------
# Metric Unit Tests
# -----------------------------------------------------------------------------

def test_context_recall_perfect_match():
    assert context_recall(["c1", "c2", "c3"], ["c1", "c2"]) == 1.0


def test_context_recall_partial_match():
    assert context_recall(["c1", "c4"], ["c1", "c2"]) == 0.5


def test_context_recall_no_relevant_found():
    assert context_recall(["c3", "c4"], ["c1", "c2"]) == 0.0


def test_context_recall_empty_context():
    assert context_recall([], ["c1", "c2"]) == 0.0


def test_context_recall_empty_relevant_ids():
    assert context_recall(["c1", "c2"], []) == 0.0


def test_context_recall_both_empty():
    assert context_recall([], []) == 0.0


def test_context_recall_ignores_duplicate_context_ids():
    # If the context list somehow has duplicates, it should not artificially inflate recall
    assert context_recall(["c1", "c1", "c1"], ["c1", "c2"]) == 0.5


def test_context_size_empty():
    assert context_size([]) == {"chunk_count": 0, "total_chars": 0}


def test_context_size_calculates_correct_chunks_and_chars():
    chunks = [
        {"chunk_id": "c1", "text": "hello"},
        {"chunk_id": "c2", "text": " world!"},
        {"chunk_id": "c3"},  # missing text key should default to empty string
    ]
    sz = context_size(chunks)
    assert sz["chunk_count"] == 3
    assert sz["total_chars"] == 12


# -----------------------------------------------------------------------------
# Integration & Expansion Configuration Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def multi_page_corpus(tmp_path):
    """Fixture providing a 2-page document where chunks can be expanded."""
    # Page 1: 3 chunks (c0, c1, c2)
    # Page 2: 2 chunks (c0, c1)
    # Page 3 of another doc: 1 chunk (c0)
    data = {
        "chunks": [
            {
                "chunk_id": "docA-p1-c0",
                "doc_id": "docA",
                "page_number": 1,
                "section_heading": "Section 1",
                "text": "Intro on page 1",
            },
            {
                "chunk_id": "docA-p1-c1",
                "doc_id": "docA",
                "page_number": 1,
                "section_heading": "Section 1.1",
                "text": "Detailed registers on page 1",
            },
            {
                "chunk_id": "docA-p1-c2",
                "doc_id": "docA",
                "page_number": 1,
                "section_heading": "Section 1.2",
                "text": "Pin description on page 1",
            },
            {
                "chunk_id": "docA-p2-c0",
                "doc_id": "docA",
                "page_number": 2,
                "section_heading": "Section 2",
                "text": "Timers overview on page 2",
            },
            {
                "chunk_id": "docA-p2-c1",
                "doc_id": "docA",
                "page_number": 2,
                "section_heading": "Section 2.1",
                "text": "Timer clock config on page 2",
            },
            {
                "chunk_id": "docB-p1-c0",
                "doc_id": "docB",
                "page_number": 1,
                "section_heading": "DocB Section 1",
                "text": "Different document text",
            },
        ]
    }
    path = tmp_path / "test_chunks.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_window_1_and_window_2_expansion(multi_page_corpus):
    retriever = BM25Retriever(multi_page_corpus)

    # Search for "Detailed" -> hit is docA-p1-c1
    hits = retriever.search("Detailed", top_k=1)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "docA-p1-c1"

    # Baseline context
    baseline_ids = [h["chunk_id"] for h in hits]
    assert baseline_ids == ["docA-p1-c1"]

    # Window 1 expansion (pulls c0 and c2 from docA page 1)
    w1_hits = retriever.expand_context(hits, window=1)
    w1_ids = [h["chunk_id"] for h in w1_hits]
    assert w1_ids == ["docA-p1-c0", "docA-p1-c1", "docA-p1-c2"]

    # Window 2 expansion (clamped by page boundary to all 3 chunks on page 1)
    w2_hits = retriever.expand_context(hits, window=2)
    w2_ids = [h["chunk_id"] for h in w2_hits]
    assert w2_ids == ["docA-p1-c0", "docA-p1-c1", "docA-p1-c2"]


def test_expansion_does_not_cross_page_or_doc_boundaries(multi_page_corpus):
    retriever = BM25Retriever(multi_page_corpus)

    # Search for "Pin" -> hit is docA-p1-c2 (last chunk on page 1)
    hits = retriever.search("Pin", top_k=1)
    assert hits[0]["chunk_id"] == "docA-p1-c2"

    w2_hits = retriever.expand_context(hits, window=2)
    w2_ids = [h["chunk_id"] for h in w2_hits]

    # Must include docA-p1 chunks only, NEVER docA-p2 or docB
    assert w2_ids == ["docA-p1-c0", "docA-p1-c1", "docA-p1-c2"]
    assert not any("p2" in cid or "docB" in cid for cid in w2_ids)


def test_expansion_recovers_unretrieved_relevant_neighbor(multi_page_corpus):
    retriever = BM25Retriever(multi_page_corpus)

    # Suppose a question has relevant chunks: docA-p1-c0 and docA-p1-c1
    relevant = ["docA-p1-c0", "docA-p1-c1"]

    # BM25 top-1 only finds docA-p1-c1
    hits = retriever.search("Detailed", top_k=1)
    bm25_ids = [h["chunk_id"] for h in hits]
    assert context_recall(bm25_ids, relevant) == 0.5

    # Window 1 expansion expands to c0, c1, c2 -> captures c0 as well!
    w1_hits = retriever.expand_context(hits, window=1)
    w1_ids = [h["chunk_id"] for h in w1_hits]
    assert context_recall(w1_ids, relevant) == 1.0


def test_run_comparison_output_structure(multi_page_corpus):
    retriever = BM25Retriever(multi_page_corpus)
    mock_golden_set = {
        "version": 1,
        "questions": [
            {
                "id": "q1",
                "question": "What is the detailed register on page 1?",
                "category": "register_lookup",
                "relevant_chunk_ids": ["docA-p1-c0", "docA-p1-c1"],
            },
            {
                "id": "q2",
                "question": "Different document query",
                "category": "functional_description",
                "relevant_chunk_ids": ["docB-p1-c0"],
            },
        ],
    }

    results = cce.run_comparison(retriever, mock_golden_set, top_k=1, repeats=2)

    assert "metadata" in results
    assert "aggregate_summary" in results
    assert "questions" in results
    assert len(results["questions"]) == 2

    # Verify deterministic fields: no timestamps in top-level or metadata
    assert "run_at" not in results
    assert "timestamp" not in results["metadata"]

    agg = results["aggregate_summary"]
    assert agg["question_count"] == 2
    assert "baseline" in agg["configurations"]
    assert "window_1" in agg["configurations"]
    assert "window_2" in agg["configurations"]

    # Verify original BM25 metrics are preserved separately
    assert "bm25_retrieval_quality" in agg
    assert "mean_recall_at_1" in agg["bm25_retrieval_quality"]
    assert "mean_mrr" in agg["bm25_retrieval_quality"]

    # Check q1 context recall improved from 0.5 to 1.0
    q1 = results["questions"][0]
    assert q1["configurations"]["baseline"]["context_recall"] == 0.5
    assert q1["configurations"]["window_1"]["context_recall"] == 1.0
    assert q1["configurations"]["window_2"]["context_recall"] == 1.0


def test_cli_runner_and_file_generation(multi_page_corpus, tmp_path, capsys):
    golden_path = tmp_path / "mock_gold.json"
    golden_path.write_text(
        json.dumps({
            "questions": [
                {
                    "id": "t1",
                    "question": "Different document text",
                    "category": "text",
                    "relevant_chunk_ids": ["docB-p1-c0"],
                }
            ]
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "output_comparison.json"

    exit_code = cce.main([
        "--chunks", str(multi_page_corpus),
        "--golden-set", str(golden_path),
        "--output", str(output_path),
        "--top-k", "2",
        "--repeats", "2",
    ])

    assert exit_code == 0
    assert output_path.is_file()

    captured = capsys.readouterr().out
    assert "CONTEXT EXPANSION EVALUATION SUMMARY" in captured
    assert "Configuration" in captured
    assert "window_1" in captured

    saved_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_data["aggregate_summary"]["question_count"] == 1
