"""Evaluation and comparison logic for BM25, Dense, and Hybrid retrieval strategies."""

from __future__ import annotations

import statistics
import time
from typing import Any, Sequence

from datasheet_assistant.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from datasheet_assistant.retrieval.bm25 import BM25Retriever
from datasheet_assistant.retrieval.dense import DenseRetriever
from datasheet_assistant.retrieval.hybrid import HybridRetriever


def percentile(values: list[float], percent: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def evaluate_retriever_ranking(
    retrieved_chunks: list[dict[str, Any]],
    relevant_ids: Sequence[str],
    top_k: int = 10,
) -> dict[str, float]:
    """Calculate Recall@k, MRR, and nDCG@k for a ranked list of chunks."""
    retrieved_ids = [str(c["chunk_id"]) for c in retrieved_chunks]
    gold_ids = list(relevant_ids)

    return {
        f"recall_at_{top_k}": recall_at_k(retrieved_ids, gold_ids, k=top_k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, gold_ids),
        f"ndcg_at_{top_k}": ndcg_at_k(retrieved_ids, gold_ids, k=top_k),
    }


def run_hybrid_comparison(
    bm25_retriever: BM25Retriever,
    dense_retriever: DenseRetriever,
    hybrid_retriever: HybridRetriever,
    golden_set: dict[str, Any],
    top_k: int = 10,
    repeats: int = 20,
    model_init_time_ms: float = 0.0,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Run comprehensive comparison of BM25, Dense, and Hybrid retrieval.

    Measures:
    - Retrieval metrics (Recall@k, MRR, nDCG@k)
    - Query latency (mean, p95) with separate warm-up
    - Per-question and aggregate results across question categories
    """
    questions = golden_set.get("questions", [])
    if not questions:
        raise ValueError("The golden set contains no questions.")

    per_question_results: list[dict[str, Any]] = []

    bm25_latencies_all: list[float] = []
    dense_latencies_all: list[float] = []
    hybrid_latencies_all: list[float] = []

    for item in questions:
        q_id = item["id"]
        query = item["question"]
        relevant_ids = item["relevant_chunk_ids"]
        category = item.get("category", "uncategorized")

        # 1. Warm-up runs (excluded from timing)
        bm25_retriever.search(query, top_k=top_k)
        dense_retriever.search(query, top_k=top_k)
        hybrid_retriever.search(query, top_k=top_k)

        # 2. Repeated measurements for BM25
        bm25_times: list[float] = []
        bm25_hits: list[dict[str, Any]] = []
        for _ in range(repeats):
            start = time.perf_counter()
            bm25_hits = bm25_retriever.search(query, top_k=top_k)
            bm25_times.append((time.perf_counter() - start) * 1000)
        bm25_latencies_all.extend(bm25_times)

        # 3. Repeated measurements for Dense
        dense_times: list[float] = []
        dense_hits: list[dict[str, Any]] = []
        for _ in range(repeats):
            start = time.perf_counter()
            dense_hits = dense_retriever.search(query, top_k=top_k)
            dense_times.append((time.perf_counter() - start) * 1000)
        dense_latencies_all.extend(dense_times)

        # 4. Repeated measurements for Hybrid
        hybrid_times: list[float] = []
        hybrid_hits: list[dict[str, Any]] = []
        for _ in range(repeats):
            start = time.perf_counter()
            hybrid_hits = hybrid_retriever.search(query, top_k=top_k)
            hybrid_times.append((time.perf_counter() - start) * 1000)
        hybrid_latencies_all.extend(hybrid_times)

        # Compute ranking metrics
        bm25_metrics = evaluate_retriever_ranking(bm25_hits, relevant_ids, top_k=top_k)
        dense_metrics = evaluate_retriever_ranking(dense_hits, relevant_ids, top_k=top_k)
        hybrid_metrics = evaluate_retriever_ranking(hybrid_hits, relevant_ids, top_k=top_k)

        q_entry = {
            "id": q_id,
            "question": query,
            "category": category,
            "relevant_chunk_ids": relevant_ids,
            "retrieval": {
                "bm25": {
                    "metrics": bm25_metrics,
                    "retrieved_chunk_ids": [c["chunk_id"] for c in bm25_hits],
                    "latency_ms": {
                        "mean": statistics.mean(bm25_times),
                        "p95": percentile(bm25_times, 95),
                    },
                },
                "dense": {
                    "metrics": dense_metrics,
                    "retrieved_chunk_ids": [c["chunk_id"] for c in dense_hits],
                    "latency_ms": {
                        "mean": statistics.mean(dense_times),
                        "p95": percentile(dense_times, 95),
                    },
                },
                "hybrid": {
                    "metrics": hybrid_metrics,
                    "retrieved_chunk_ids": [c["chunk_id"] for c in hybrid_hits],
                    "latency_ms": {
                        "mean": statistics.mean(hybrid_times),
                        "p95": percentile(hybrid_times, 95),
                    },
                    "fused_debug": [
                        {
                            "chunk_id": c["chunk_id"],
                            "rrf_score": c["score"],
                            "bm25_rank": c.get("bm25_rank"),
                            "bm25_score": c.get("bm25_score"),
                            "dense_rank": c.get("dense_rank"),
                            "dense_score": c.get("dense_score"),
                        }
                        for c in hybrid_hits
                    ],
                },
            },
        }
        per_question_results.append(q_entry)

    count = len(per_question_results)

    # Categories breakdown
    categories: dict[str, list[dict[str, Any]]] = {}
    for q in per_question_results:
        categories.setdefault(q["category"], []).append(q)

    category_summary = {}
    for cat, cat_questions in sorted(categories.items()):
        category_summary[cat] = {
            "question_count": len(cat_questions),
            "bm25": {
                f"recall_at_{top_k}": statistics.mean(
                    q["retrieval"]["bm25"]["metrics"][f"recall_at_{top_k}"] for q in cat_questions
                ),
                "mrr": statistics.mean(
                    q["retrieval"]["bm25"]["metrics"]["reciprocal_rank"] for q in cat_questions
                ),
                f"ndcg_at_{top_k}": statistics.mean(
                    q["retrieval"]["bm25"]["metrics"][f"ndcg_at_{top_k}"] for q in cat_questions
                ),
            },
            "dense": {
                f"recall_at_{top_k}": statistics.mean(
                    q["retrieval"]["dense"]["metrics"][f"recall_at_{top_k}"] for q in cat_questions
                ),
                "mrr": statistics.mean(
                    q["retrieval"]["dense"]["metrics"]["reciprocal_rank"] for q in cat_questions
                ),
                f"ndcg_at_{top_k}": statistics.mean(
                    q["retrieval"]["dense"]["metrics"][f"ndcg_at_{top_k}"] for q in cat_questions
                ),
            },
            "hybrid": {
                f"recall_at_{top_k}": statistics.mean(
                    q["retrieval"]["hybrid"]["metrics"][f"recall_at_{top_k}"] for q in cat_questions
                ),
                "mrr": statistics.mean(
                    q["retrieval"]["hybrid"]["metrics"]["reciprocal_rank"] for q in cat_questions
                ),
                f"ndcg_at_{top_k}": statistics.mean(
                    q["retrieval"]["hybrid"]["metrics"][f"ndcg_at_{top_k}"] for q in cat_questions
                ),
            },
        }

    # Aggregate summaries
    aggregate_summary = {
        "question_count": count,
        "top_k": top_k,
        "repeats_per_question": repeats,
        "rrf_k": hybrid_retriever.rrf_k,
        "embedding_model": {
            "name": model_name,
            "dimensions": dense_retriever.embedding_dim,
            "initialization_time_ms": round(model_init_time_ms, 3),
        },
        "retrievers": {
            "bm25": {
                f"mean_recall_at_{top_k}": statistics.mean(
                    q["retrieval"]["bm25"]["metrics"][f"recall_at_{top_k}"] for q in per_question_results
                ),
                "mean_mrr": statistics.mean(
                    q["retrieval"]["bm25"]["metrics"]["reciprocal_rank"] for q in per_question_results
                ),
                f"mean_ndcg_at_{top_k}": statistics.mean(
                    q["retrieval"]["bm25"]["metrics"][f"ndcg_at_{top_k}"] for q in per_question_results
                ),
                "latency_ms": {
                    "mean": statistics.mean(bm25_latencies_all),
                    "p95": percentile(bm25_latencies_all, 95),
                },
            },
            "dense": {
                f"mean_recall_at_{top_k}": statistics.mean(
                    q["retrieval"]["dense"]["metrics"][f"recall_at_{top_k}"] for q in per_question_results
                ),
                "mean_mrr": statistics.mean(
                    q["retrieval"]["dense"]["metrics"]["reciprocal_rank"] for q in per_question_results
                ),
                f"mean_ndcg_at_{top_k}": statistics.mean(
                    q["retrieval"]["dense"]["metrics"][f"ndcg_at_{top_k}"] for q in per_question_results
                ),
                "latency_ms": {
                    "mean": statistics.mean(dense_latencies_all),
                    "p95": percentile(dense_latencies_all, 95),
                },
            },
            "hybrid": {
                f"mean_recall_at_{top_k}": statistics.mean(
                    q["retrieval"]["hybrid"]["metrics"][f"recall_at_{top_k}"] for q in per_question_results
                ),
                "mean_mrr": statistics.mean(
                    q["retrieval"]["hybrid"]["metrics"]["reciprocal_rank"] for q in per_question_results
                ),
                f"mean_ndcg_at_{top_k}": statistics.mean(
                    q["retrieval"]["hybrid"]["metrics"][f"ndcg_at_{top_k}"] for q in per_question_results
                ),
                "latency_ms": {
                    "mean": statistics.mean(hybrid_latencies_all),
                    "p95": percentile(hybrid_latencies_all, 95),
                },
            },
        },
        "by_category": category_summary,
    }

    corpus_chunks = bm25_retriever.chunks
    corpus_page_count = len({(chunk["doc_id"], chunk["page_number"]) for chunk in corpus_chunks})
    corpus_document_count = len({chunk["doc_id"] for chunk in corpus_chunks})
    top_k_corpus_count = min(top_k, len(corpus_chunks))
    top_k_corpus_percentage = (
        100 * top_k_corpus_count / len(corpus_chunks) if corpus_chunks else 0.0
    )

    return {
        "metadata": {
            "description": "Evaluation and comparison of BM25, Dense, and Hybrid (RRF) retrieval strategies",
            "model_name": model_name,
            "embedding_dimensions": dense_retriever.embedding_dim,
            "corpus_chunk_count": len(corpus_chunks),
            "corpus_page_count": corpus_page_count,
            "corpus_document_count": corpus_document_count,
            "question_count": count,
            "top_k": top_k,
            "rrf_k": hybrid_retriever.rrf_k,
            "metric_definitions": {
                "recall_at_k": f"Fraction of relevant gold chunks retrieved in the top {top_k}",
                "mrr": "Reciprocal rank (1/rank) of the first relevant chunk retrieved",
                "ndcg_at_k": f"Normalized discounted cumulative gain at rank {top_k} with binary relevance",
                "rrf_score": "Reciprocal Rank Fusion score: sum(1 / (rrf_k + rank_i)) across BM25 and Dense",
            },
            "corpus_limitation_note": (
                f"The indexed corpus contains {len(corpus_chunks):,} chunks across "
                f"{corpus_page_count:,} pages and {corpus_document_count:,} document(s). "
                "Evaluation conclusions are limited to the document coverage and relevance "
                "labels in the golden set. "
                f"Top-{top_k} can include at most {top_k_corpus_count:,} chunks "
                f"({top_k_corpus_percentage:.2f}% of the indexed corpus)."
            ),
            "timing_methodology": (
                "Each query is warmed up before measurement. Timing is measured over repeated "
                f"iterations ({repeats} repeats) using time.perf_counter(). Dense query latency includes "
                "query embedding and cosine similarity computation, excluding one-time model initialization."
            ),
        },
        "aggregate_summary": aggregate_summary,
        "questions": per_question_results,
    }
