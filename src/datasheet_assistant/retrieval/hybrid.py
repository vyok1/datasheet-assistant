"""Hybrid retrieval combining lexical (BM25) and dense semantic retrieval via RRF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from datasheet_assistant.retrieval.bm25 import BM25Retriever
from datasheet_assistant.retrieval.dense import DenseRetriever

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_runs: Sequence[tuple[str, list[dict[str, Any]]]],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked retrieval runs using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = sum_{run} 1 / (rrf_k + rank_{run}(d))

    Args:
        ranked_runs: Sequence of (run_name, ranked_results_list), where each item
                     in ranked_results_list is a dict containing at least 'chunk_id'.
                     Ranks are 1-based according to list order.
        rrf_k: Smoothing constant (default: 60).

    Returns:
        Deduplicated list of chunk dicts sorted by RRF score descending,
        breaking ties deterministically by chunk_id ascending.
        Each dict preserves original chunk fields and includes per-run ranks and scores.
    """
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
        raise ValueError(f"rrf_k must be a positive integer, got {rrf_k!r}")

    # Map chunk_id -> aggregate data
    fused_entries: dict[str, dict[str, Any]] = {}

    for run_name, run_results in ranked_runs:
        seen_in_run: set[str] = set()
        for rank, item in enumerate(run_results, start=1):
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id:
                continue

            # If duplicate chunk_id exists in the same run, keep only the first (best) occurrence
            if chunk_id in seen_in_run:
                continue
            seen_in_run.add(chunk_id)

            rrf_contribution = 1.0 / (rrf_k + rank)

            if chunk_id not in fused_entries:
                fused_entries[chunk_id] = {
                    "chunk_id": chunk_id,
                    "doc_id": item.get("doc_id"),
                    "page_number": item.get("page_number"),
                    "section_heading": item.get("section_heading"),
                    "text": item.get("text", ""),
                    "score": 0.0,
                    "ranks": {},
                    "scores": {},
                }

            fused_entries[chunk_id]["score"] += rrf_contribution
            fused_entries[chunk_id]["ranks"][run_name] = rank
            fused_entries[chunk_id]["scores"][run_name] = item.get("score")

    # Sort deterministically: descending RRF score, ascending chunk_id for tie-breaking
    sorted_chunks = sorted(
        fused_entries.values(),
        key=lambda entry: (-entry["score"], entry["chunk_id"]),
    )

    # Format output records with explicit run metadata
    output: list[dict[str, Any]] = []
    for entry in sorted_chunks:
        ranks = entry.pop("ranks")
        scores = entry.pop("scores")

        record: dict[str, Any] = {
            "chunk_id": entry["chunk_id"],
            "doc_id": entry["doc_id"],
            "page_number": entry["page_number"],
            "section_heading": entry["section_heading"],
            "score": float(entry["score"]),
            "text": entry["text"],
            "bm25_rank": ranks.get("bm25"),
            "bm25_score": scores.get("bm25"),
            "dense_rank": ranks.get("dense"),
            "dense_score": scores.get("dense"),
            "run_ranks": ranks,
            "run_scores": scores,
        }
        output.append(record)

    return output


class HybridRetriever:
    """Hybrid retriever combining BM25 and Dense semantic retrieval via RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top_k chunks by fusing BM25 and Dense rankings via RRF.

        Args:
            query: Non-empty query string.
            top_k: Number of final fused chunks to return.
            candidate_k: Number of candidates to retrieve from each retriever before fusion.
                         If None, retrieves all available chunks in the corpus.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        corpus_size = len(self.bm25.chunks)
        pool_k = candidate_k if candidate_k is not None else corpus_size
        pool_k = max(pool_k, top_k)

        bm25_results = self.bm25.search(query, top_k=pool_k)
        dense_results = self.dense.search(query, top_k=pool_k)

        fused = reciprocal_rank_fusion(
            ranked_runs=[("bm25", bm25_results), ("dense", dense_results)],
            rrf_k=self.rrf_k,
        )

        return fused[:top_k]
