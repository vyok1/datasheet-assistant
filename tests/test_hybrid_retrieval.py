"""Tests for Dense retrieval, Hybrid retrieval (RRF), and hybrid evaluation.

All tests run completely offline and use mock/fake embedders with dependency injection.
Zero models are downloaded during testing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import numpy as np
import pytest

from datasheet_assistant.evaluation.hybrid_comparison import (
    evaluate_retriever_ranking,
    percentile,
    run_hybrid_comparison,
)
from datasheet_assistant.retrieval.bm25 import BM25Retriever
from datasheet_assistant.retrieval.dense import DenseRetriever
from datasheet_assistant.retrieval.hybrid import (
    DEFAULT_RRF_K,
    HybridRetriever,
    reciprocal_rank_fusion,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_hybrid_retrieval.py"
spec = importlib.util.spec_from_file_location("evaluate_hybrid_retrieval", SCRIPT_PATH)
eval_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_cli)


class MockEmbedder:
    """Deterministic offline embedder for unit testing."""

    def __init__(self, dim: int = 4, mapping: dict[str, list[float]] | None = None) -> None:
        self.dim = dim
        self.mapping = mapping or {}

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = []
        for text in texts:
            if text in self.mapping:
                vec = np.array(self.mapping[text], dtype=np.float32)
            else:
                # Deterministic pseudo-vector from text characters
                seed = sum(ord(c) for c in text)
                vec = np.array([float((seed * (i + 1)) % 100 + 1) for i in range(self.dim)], dtype=np.float32)

            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "c001",
            "doc_id": "doc1",
            "page_number": 1,
            "section_heading": "GPIO Setup",
            "text": "GPIO configuration registers for pin control.",
        },
        {
            "chunk_id": "c002",
            "doc_id": "doc1",
            "page_number": 1,
            "section_heading": "DMA Stream",
            "text": "Direct Memory Access controller transfers data between memory and peripherals.",
        },
        {
            "chunk_id": "c003",
            "doc_id": "doc1",
            "page_number": 2,
            "section_heading": "SPI Timing",
            "text": "Serial Peripheral Interface clock polarity and phase control.",
        },
    ]


@pytest.fixture
def sample_chunks_file(tmp_path: Path, sample_chunks: list[dict[str, Any]]) -> Path:
    p = tmp_path / "chunks.json"
    p.write_text(json.dumps({"chunks": sample_chunks}), encoding="utf-8")
    return p


@pytest.fixture
def sample_golden_set() -> dict[str, Any]:
    return {
        "version": 1,
        "questions": [
            {
                "id": "q1",
                "question": "How to configure GPIO?",
                "relevant_chunk_ids": ["c001"],
                "category": "gpio",
            },
            {
                "id": "q2",
                "question": "How does DMA transfer work?",
                "relevant_chunk_ids": ["c002"],
                "category": "dma",
            },
        ],
    }


# ==============================================================================
# DenseRetriever Tests
# ==============================================================================


def test_dense_retriever_initialization_and_alignment(sample_chunks: list[dict[str, Any]]):
    mock = MockEmbedder(dim=4)
    retriever = DenseRetriever(chunks_source=sample_chunks, embedder=mock)

    assert retriever.embedding_dim == 4
    assert len(retriever.chunks) == 3
    assert retriever.chunk_embeddings.shape == (3, 4)

    # Verify embeddings are normalized
    norms = np.linalg.norm(retriever.chunk_embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_dense_retriever_ranking_cosine_similarity(sample_chunks: list[dict[str, Any]]):
    # Orthogonal vectors so similarity can be directly asserted
    mapping = {
        sample_chunks[0]["text"]: [1.0, 0.0, 0.0, 0.0],
        sample_chunks[1]["text"]: [0.0, 1.0, 0.0, 0.0],
        sample_chunks[2]["text"]: [0.0, 0.0, 1.0, 0.0],
        "gpio pin config": [1.0, 0.0, 0.0, 0.0],
    }
    mock = MockEmbedder(dim=4, mapping=mapping)
    retriever = DenseRetriever(chunks_source=sample_chunks, embedder=mock)

    results = retriever.search("gpio pin config", top_k=3)
    assert len(results) == 3
    assert results[0]["chunk_id"] == "c001"
    assert pytest.approx(results[0]["score"], rel=1e-4) == 1.0
    assert results[1]["score"] == 0.0
    assert results[2]["score"] == 0.0


def test_dense_retriever_tie_breaking(sample_chunks: list[dict[str, Any]]):
    # All chunks get identical zero embeddings -> identical scores
    mapping = {
        sample_chunks[0]["text"]: [1.0, 0.0, 0.0, 0.0],
        sample_chunks[1]["text"]: [1.0, 0.0, 0.0, 0.0],
        sample_chunks[2]["text"]: [1.0, 0.0, 0.0, 0.0],
        "query": [1.0, 0.0, 0.0, 0.0],
    }
    mock = MockEmbedder(dim=4, mapping=mapping)
    retriever = DenseRetriever(chunks_source=sample_chunks, embedder=mock)

    results = retriever.search("query", top_k=3)
    # Tie broken deterministically by chunk index
    assert [r["chunk_id"] for r in results] == ["c001", "c002", "c003"]


def test_dense_retriever_validation():
    mock = MockEmbedder(dim=4)
    with pytest.raises(ValueError, match="No chunks found"):
        DenseRetriever(chunks_source=[], embedder=mock)

    retriever = DenseRetriever(
        chunks_source=[{"chunk_id": "c1", "doc_id": "d1", "page_number": 1, "text": "abc"}],
        embedder=mock,
    )
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        retriever.search("")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.search("abc", top_k=0)


# ==============================================================================
# RRF and HybridRetriever Tests
# ==============================================================================


def test_rrf_formula_and_scoring():
    # Run 1: c001 (rank 1), c002 (rank 2)
    # Run 2: c002 (rank 1), c003 (rank 2)
    run1 = [
        {"chunk_id": "c001", "score": 10.0, "text": "t1"},
        {"chunk_id": "c002", "score": 5.0, "text": "t2"},
    ]
    run2 = [
        {"chunk_id": "c002", "score": 0.9, "text": "t2"},
        {"chunk_id": "c003", "score": 0.1, "text": "t3"},
    ]

    fused = reciprocal_rank_fusion([("bm25", run1), ("dense", run2)], rrf_k=60)

    # c002 is rank 2 in run1 and rank 1 in run2:
    # 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    assert fused[0]["chunk_id"] == "c002"
    expected_c002 = 1.0 / (60 + 2) + 1.0 / (60 + 1)
    assert pytest.approx(fused[0]["score"], rel=1e-5) == expected_c002
    assert fused[0]["bm25_rank"] == 2
    assert fused[0]["bm25_score"] == 5.0
    assert fused[0]["dense_rank"] == 1
    assert fused[0]["dense_score"] == 0.9

    # c001 has only bm25 rank 1: 1/61 = 0.016393
    # c003 has only dense rank 2: 1/62 = 0.016129
    assert fused[1]["chunk_id"] == "c001"
    assert fused[1]["bm25_rank"] == 1
    assert fused[1]["dense_rank"] is None

    assert fused[2]["chunk_id"] == "c003"
    assert fused[2]["bm25_rank"] is None
    assert fused[2]["dense_rank"] == 2


def test_rrf_duplicate_chunk_in_single_run():
    # If a run contains duplicate chunk IDs, only the first occurrence should count
    run = [
        {"chunk_id": "c001", "score": 10.0, "text": "t1"},
        {"chunk_id": "c001", "score": 5.0, "text": "t1"},
    ]
    fused = reciprocal_rank_fusion([("bm25", run)], rrf_k=60)
    assert len(fused) == 1
    assert pytest.approx(fused[0]["score"], rel=1e-5) == 1.0 / 61


def test_rrf_tie_breaking():
    # Two chunks with identical score: tie-breaker is chunk_id ascending
    run = [
        {"chunk_id": "z_chunk", "score": 1.0},
    ]
    run2 = [
        {"chunk_id": "a_chunk", "score": 1.0},
    ]
    fused = reciprocal_rank_fusion([("r1", run), ("r2", run2)], rrf_k=60)
    assert len(fused) == 2
    assert fused[0]["score"] == fused[1]["score"]
    assert fused[0]["chunk_id"] == "a_chunk"
    assert fused[1]["chunk_id"] == "z_chunk"


def test_hybrid_retriever_search(sample_chunks_file: Path, sample_chunks: list[dict[str, Any]]):
    bm25 = BM25Retriever(sample_chunks_file)
    mock = MockEmbedder(dim=4)
    dense = DenseRetriever(chunks_source=sample_chunks_file, embedder=mock)
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("GPIO configuration", top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] in {"c001", "c002", "c003"}
    assert "score" in results[0]
    assert "bm25_rank" in results[0]
    assert "dense_rank" in results[0]


# ==============================================================================
# Evaluation and Comparison Tests
# ==============================================================================


def test_evaluate_retriever_ranking():
    hits = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}]
    metrics = evaluate_retriever_ranking(hits, relevant_ids=["c2"], top_k=3)
    assert metrics["recall_at_3"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5  # Rank 2 -> 1/2


def test_run_hybrid_comparison(sample_chunks_file: Path, sample_golden_set: dict[str, Any]):
    bm25 = BM25Retriever(sample_chunks_file)
    mock = MockEmbedder(dim=4)
    dense = DenseRetriever(chunks_source=sample_chunks_file, embedder=mock)
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    output = run_hybrid_comparison(
        bm25_retriever=bm25,
        dense_retriever=dense,
        hybrid_retriever=hybrid,
        golden_set=sample_golden_set,
        top_k=2,
        repeats=2,
        model_init_time_ms=12.34,
        model_name="test-mock-model",
    )

    assert "metadata" in output
    assert "aggregate_summary" in output
    assert "questions" in output
    assert len(output["questions"]) == 2

    agg = output["aggregate_summary"]
    assert "retrievers" in agg
    for strat in ["bm25", "dense", "hybrid"]:
        assert f"mean_recall_at_2" in agg["retrievers"][strat]
        assert "mean_mrr" in agg["retrievers"][strat]
        assert f"mean_ndcg_at_2" in agg["retrievers"][strat]
        assert "latency_ms" in agg["retrievers"][strat]


def test_percentile_helper():
    assert percentile([], 95) == 0.0
    assert percentile([10.0], 95) == 10.0
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert pytest.approx(percentile(data, 50)) == 3.0


# ==============================================================================
# CLI and Output Tests
# ==============================================================================


def test_cli_main_and_json_export(tmp_path: Path, sample_chunks_file: Path, sample_golden_set: dict[str, Any]):
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(sample_golden_set), encoding="utf-8")
    output_path = tmp_path / "out_hybrid.json"

    mock = MockEmbedder(dim=4)

    with patch.object(eval_cli, "SentenceTransformerEmbedder", return_value=mock):
        exit_code = eval_cli.main([
            "--chunks", str(sample_chunks_file),
            "--golden-set", str(golden_path),
            "--output", str(output_path),
            "--top-k", "2",
            "--repeats", "2",
            "--rrf-k", "60",
        ])

    assert exit_code == 0
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["aggregate_summary"]["question_count"] == 2
    assert "retrievers" in saved["aggregate_summary"]
    assert "bm25" in saved["aggregate_summary"]["retrievers"]
    assert "dense" in saved["aggregate_summary"]["retrievers"]
    assert "hybrid" in saved["aggregate_summary"]["retrievers"]


def test_cli_missing_files(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.json"
    assert eval_cli.main(["--chunks", str(non_existent)]) == 1
