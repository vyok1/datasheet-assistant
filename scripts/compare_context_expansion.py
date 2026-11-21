"""CLI and library to evaluate and compare baseline retrieval vs context expansion.

Configurations evaluated:
- baseline: BM25 retrieval without expansion
- window_1: BM25 retrieval + expand_context(window=1)
- window_2: BM25 retrieval + expand_context(window=2)

Calculates:
- Context recall (coverage of relevant gold chunks in the context)
- Context size (number of chunks and total characters)
- Original BM25 ranking metrics (Recall@10, MRR, nDCG@10)
- Local latency measurements with warm-up and repeated iterations
"""

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from datasheet_assistant.evaluation.context_comparison import (
    context_recall,
    context_size,
)
from datasheet_assistant.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from datasheet_assistant.retrieval.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "pages" / "chunks.json"
DEFAULT_GOLDEN_SET = ROOT / "data" / "golden_set.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "context_expansion_comparison.json"
DEFAULT_TOP_K = 10
DEFAULT_REPEATS = 20


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


def run_comparison(
    retriever: BM25Retriever,
    golden_set: dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Run comparison of baseline, window=1, and window=2 context expansion.

    Returns deterministic data dictionary (without wall-clock timestamps).
    """
    questions = golden_set.get("questions", [])
    if not questions:
        raise ValueError("The golden set contains no questions.")

    per_question_results = []

    # Store repeated latencies across all questions for aggregate latency stats
    latencies_bm25_all: list[float] = []
    latencies_w1_all: list[float] = []
    latencies_w2_all: list[float] = []
    overhead_w1_all: list[float] = []
    overhead_w2_all: list[float] = []

    for item in questions:
        q_id = item["id"]
        query = item["question"]
        relevant_ids = item["relevant_chunk_ids"]
        category = item.get("category", "uncategorized")

        # Warm-up calls (excluded from timing)
        warmup_bm25 = retriever.search(query, top_k=top_k)
        retriever.expand_context(warmup_bm25, window=1)
        retriever.expand_context(warmup_bm25, window=2)

        bm25_times: list[float] = []
        w1_times: list[float] = []
        w2_times: list[float] = []
        w1_overheads: list[float] = []
        w2_overheads: list[float] = []

        last_bm25_hits: list[dict[str, Any]] = []
        last_w1_hits: list[dict[str, Any]] = []
        last_w2_hits: list[dict[str, Any]] = []

        for _ in range(repeats):
            # 1. BM25 search
            t0 = time.perf_counter()
            bm25_hits = retriever.search(query, top_k=top_k)
            t1 = time.perf_counter()

            # 2. Window 1 expansion
            t2 = time.perf_counter()
            w1_hits = retriever.expand_context(bm25_hits, window=1)
            t3 = time.perf_counter()

            # 3. Window 2 expansion
            t4 = time.perf_counter()
            w2_hits = retriever.expand_context(bm25_hits, window=2)
            t5 = time.perf_counter()

            bm25_ms = (t1 - t0) * 1000
            w1_expand_ms = (t3 - t2) * 1000
            w2_expand_ms = (t5 - t4) * 1000

            bm25_times.append(bm25_ms)
            w1_times.append(bm25_ms + w1_expand_ms)
            w2_times.append(bm25_ms + w2_expand_ms)
            w1_overheads.append(w1_expand_ms)
            w2_overheads.append(w2_expand_ms)

            last_bm25_hits = bm25_hits
            last_w1_hits = w1_hits
            last_w2_hits = w2_hits

        latencies_bm25_all.extend(bm25_times)
        latencies_w1_all.extend(w1_times)
        latencies_w2_all.extend(w2_times)
        overhead_w1_all.extend(w1_overheads)
        overhead_w2_all.extend(w2_overheads)

        # Baseline original BM25 results
        bm25_chunk_ids = [hit["chunk_id"] for hit in last_bm25_hits]
        w1_chunk_ids = [hit["chunk_id"] for hit in last_w1_hits]
        w2_chunk_ids = [hit["chunk_id"] for hit in last_w2_hits]

        # Ranking metrics strictly for BM25 retrieval
        bm25_ranking_metrics = {
            f"recall_at_{top_k}": recall_at_k(bm25_chunk_ids, relevant_ids, k=top_k),
            "reciprocal_rank": reciprocal_rank(bm25_chunk_ids, relevant_ids),
            f"ndcg_at_{top_k}": ndcg_at_k(bm25_chunk_ids, relevant_ids, k=top_k),
        }

        # Context sizes
        bm25_sz = context_size(last_bm25_hits)
        w1_sz = context_size(last_w1_hits)
        w2_sz = context_size(last_w2_hits)

        # Context recalls
        bm25_ctx_recall = context_recall(bm25_chunk_ids, relevant_ids)
        w1_ctx_recall = context_recall(w1_chunk_ids, relevant_ids)
        w2_ctx_recall = context_recall(w2_chunk_ids, relevant_ids)

        q_entry = {
            "id": q_id,
            "question": query,
            "category": category,
            "relevant_chunk_ids": relevant_ids,
            "bm25_retrieval_ranking": bm25_ranking_metrics,
            "configurations": {
                "baseline": {
                    "context_recall": bm25_ctx_recall,
                    "chunk_count": bm25_sz["chunk_count"],
                    "total_chars": bm25_sz["total_chars"],
                    "latency_ms": {
                        "mean": statistics.mean(bm25_times),
                        "p95": percentile(bm25_times, 95),
                    },
                    "chunk_ids": bm25_chunk_ids,
                },
                "window_1": {
                    "context_recall": w1_ctx_recall,
                    "chunk_count": w1_sz["chunk_count"],
                    "total_chars": w1_sz["total_chars"],
                    "latency_ms": {
                        "mean": statistics.mean(w1_times),
                        "p95": percentile(w1_times, 95),
                    },
                    "expansion_overhead_ms": {
                        "mean": statistics.mean(w1_overheads),
                        "p95": percentile(w1_overheads, 95),
                    },
                    "chunk_ids": w1_chunk_ids,
                },
                "window_2": {
                    "context_recall": w2_ctx_recall,
                    "chunk_count": w2_sz["chunk_count"],
                    "total_chars": w2_sz["total_chars"],
                    "latency_ms": {
                        "mean": statistics.mean(w2_times),
                        "p95": percentile(w2_times, 95),
                    },
                    "expansion_overhead_ms": {
                        "mean": statistics.mean(w2_overheads),
                        "p95": percentile(w2_overheads, 95),
                    },
                    "chunk_ids": w2_chunk_ids,
                },
            },
        }
        per_question_results.append(q_entry)

    count = len(per_question_results)

    # Aggregates across all questions
    aggregate_summary = {
        "question_count": count,
        "top_k": top_k,
        "repeats_per_question": repeats,
        "bm25_retrieval_quality": {
            f"mean_recall_at_{top_k}": statistics.mean(
                q["bm25_retrieval_ranking"][f"recall_at_{top_k}"] for q in per_question_results
            ),
            "mean_mrr": statistics.mean(
                q["bm25_retrieval_ranking"]["reciprocal_rank"] for q in per_question_results
            ),
            f"mean_ndcg_at_{top_k}": statistics.mean(
                q["bm25_retrieval_ranking"][f"ndcg_at_{top_k}"] for q in per_question_results
            ),
        },
        "configurations": {
            "baseline": {
                "mean_context_recall": statistics.mean(
                    q["configurations"]["baseline"]["context_recall"] for q in per_question_results
                ),
                "mean_chunk_count": statistics.mean(
                    q["configurations"]["baseline"]["chunk_count"] for q in per_question_results
                ),
                "mean_total_chars": statistics.mean(
                    q["configurations"]["baseline"]["total_chars"] for q in per_question_results
                ),
                "latency_ms": {
                    "mean": statistics.mean(latencies_bm25_all),
                    "p95": percentile(latencies_bm25_all, 95),
                },
            },
            "window_1": {
                "mean_context_recall": statistics.mean(
                    q["configurations"]["window_1"]["context_recall"] for q in per_question_results
                ),
                "mean_chunk_count": statistics.mean(
                    q["configurations"]["window_1"]["chunk_count"] for q in per_question_results
                ),
                "mean_total_chars": statistics.mean(
                    q["configurations"]["window_1"]["total_chars"] for q in per_question_results
                ),
                "latency_ms": {
                    "mean": statistics.mean(latencies_w1_all),
                    "p95": percentile(latencies_w1_all, 95),
                },
                "expansion_overhead_ms": {
                    "mean": statistics.mean(overhead_w1_all),
                    "p95": percentile(overhead_w1_all, 95),
                },
            },
            "window_2": {
                "mean_context_recall": statistics.mean(
                    q["configurations"]["window_2"]["context_recall"] for q in per_question_results
                ),
                "mean_chunk_count": statistics.mean(
                    q["configurations"]["window_2"]["chunk_count"] for q in per_question_results
                ),
                "mean_total_chars": statistics.mean(
                    q["configurations"]["window_2"]["total_chars"] for q in per_question_results
                ),
                "latency_ms": {
                    "mean": statistics.mean(latencies_w2_all),
                    "p95": percentile(latencies_w2_all, 95),
                },
                "expansion_overhead_ms": {
                    "mean": statistics.mean(overhead_w2_all),
                    "p95": percentile(overhead_w2_all, 95),
                },
            },
        },
    }

    # Deterministic output dictionary (no non-deterministic timestamps)
    output = {
        "metadata": {
            "description": "Empirical comparison of baseline BM25 retrieval vs context expansion (window=1, window=2)",
            "retriever": "BM25Okapi",
            "top_k": top_k,
            "repeats_per_question": repeats,
            "note_on_metrics": (
                "Context expansion increases evidence coverage (Context Recall) around retrieved chunks "
                "on the same page. Original BM25 ranking metrics (Recall@10, MRR, nDCG@10) reflect ranking "
                "quality and are not altered by expansion. Expansion chunks receive score=0.0."
            ),
            "timing_note": "Latencies are local measurements with warm-up and repeated execution.",
        },
        "aggregate_summary": aggregate_summary,
        "questions": per_question_results,
    }

    return output


def print_comparison_table(results: dict[str, Any]) -> None:
    """Print a clean CLI summary table comparing the three configurations."""
    agg = results["aggregate_summary"]
    cfgs = agg["configurations"]
    bm25_qual = agg["bm25_retrieval_quality"]

    print("\n" + "=" * 80)
    print("CONTEXT EXPANSION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Questions Evaluated: {agg['question_count']}")
    print(f"Top-K Retrieval Cutoff: {agg['top_k']}")
    print(f"Repeats Per Question: {agg['repeats_per_question']}")

    top_k = agg["top_k"]
    recall_key = f"mean_recall_at_{top_k}"
    ndcg_key = f"mean_ndcg_at_{top_k}"
    print(f"\n[Original BM25 Ranking Quality]")
    print(f"  Recall@{top_k}: {bm25_qual[recall_key]:.4f}")
    print(f"  MRR:       {bm25_qual['mean_mrr']:.4f}")
    print(f"  nDCG@{top_k}:   {bm25_qual[ndcg_key]:.4f}")

    print("\n[Configuration Comparison]")
    header = f"{'Configuration':<16} | {'Ctx Recall':<10} | {'Mean Chunks':<11} | {'Mean Chars':<11} | {'Mean Latency':<12} | {'p95 Latency':<12}"
    print(header)
    print("-" * len(header))

    for name in ["baseline", "window_1", "window_2"]:
        c = cfgs[name]
        print(
            f"{name:<16} | "
            f"{c['mean_context_recall']:<10.4f} | "
            f"{c['mean_chunk_count']:<11.1f} | "
            f"{c['mean_total_chars']:<11.1f} | "
            f"{c['latency_ms']['mean']:<10.3f} ms | "
            f"{c['latency_ms']['p95']:<10.3f} ms"
        )
    print("=" * 80 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare baseline retrieval against context expansion."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS,
        help=f"Path to chunks.json (default: {DEFAULT_CHUNKS})",
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=DEFAULT_GOLDEN_SET,
        help=f"Path to golden_set.json (default: {DEFAULT_GOLDEN_SET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write comparison JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Retrieval top-k cutoff (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Latency timing repeats (default: {DEFAULT_REPEATS})",
    )

    args = parser.parse_args(argv)

    if not args.chunks.is_file():
        print(f"[ERROR] Chunks file not found: {args.chunks}")
        return 1

    if not args.golden_set.is_file():
        print(f"[ERROR] Golden set file not found: {args.golden_set}")
        return 1

    with args.golden_set.open("r", encoding="utf-8") as f:
        golden_set = json.load(f)

    retriever = BM25Retriever(args.chunks)
    comparison = run_comparison(retriever, golden_set, top_k=args.top_k, repeats=args.repeats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print_comparison_table(comparison)
    print(f"Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
