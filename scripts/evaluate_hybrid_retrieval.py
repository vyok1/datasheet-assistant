"""CLI runner to evaluate and compare BM25, Dense, and Hybrid (RRF) retrieval strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from datasheet_assistant.evaluation.hybrid_comparison import (
    run_hybrid_comparison,
)
from datasheet_assistant.retrieval.bm25 import BM25Retriever
from datasheet_assistant.retrieval.dense import DenseRetriever, SentenceTransformerEmbedder
from datasheet_assistant.retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "pages" / "chunks.json"
DEFAULT_GOLDEN_SET = ROOT / "data" / "golden_set.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "hybrid_retrieval_comparison.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 10
DEFAULT_REPEATS = 20
DEFAULT_RRF_K = 60


def print_comparison_table(results: dict[str, Any]) -> None:
    """Print a clean CLI summary table comparing BM25, Dense, and Hybrid retrieval."""
    agg = results["aggregate_summary"]
    top_k = agg["top_k"]
    retrievers = agg["retrievers"]
    model_info = agg["embedding_model"]

    print("\n" + "=" * 88)
    print("HYBRID RETRIEVAL EVALUATION SUMMARY (BM25 vs Dense vs Hybrid)")
    print("=" * 88)
    print(f"Questions Evaluated:      {agg['question_count']}")
    print(f"Top-K Retrieval Cutoff:   {top_k}")
    print(f"RRF Constant (k):         {agg['rrf_k']}")
    print(f"Repeats Per Question:     {agg['repeats_per_question']}")
    print(f"Embedding Model:          {model_info['name']} ({model_info['dimensions']} dim)")
    print(f"Model Init Time:          {model_info['initialization_time_ms']:.2f} ms")

    recall_col = f"Recall@{top_k}"
    ndcg_col = f"nDCG@{top_k}"

    header = (
        f"{'Strategy':<10} | {recall_col:<10} | {'MRR':<8} | {ndcg_col:<9} | "
        f"{'Mean Latency':<14} | {'p95 Latency':<14}"
    )
    print("\n" + header)
    print("-" * len(header))

    recall_key = f"mean_recall_at_{top_k}"
    ndcg_key = f"mean_ndcg_at_{top_k}"

    for name in ["bm25", "dense", "hybrid"]:
        r = retrievers[name]
        mean_rec = r[recall_key]
        mean_mrr = r["mean_mrr"]
        mean_ndcg = r[ndcg_key]
        lat_mean = r["latency_ms"]["mean"]
        lat_p95 = r["latency_ms"]["p95"]

        print(
            f"{name.upper():<10} | "
            f"{mean_rec:<10.4f} | "
            f"{mean_mrr:<8.4f} | "
            f"{mean_ndcg:<9.4f} | "
            f"{lat_mean:<11.3f} ms | "
            f"{lat_p95:<11.3f} ms"
        )

    print("\n" + "-" * 88)
    print("PER-CATEGORY BREAKDOWN")
    print("-" * 88)
    cat_header = f"{'Category':<24} | {'Count':<5} | {'BM25 MRR':<9} | {'Dense MRR':<10} | {'Hybrid MRR':<11} | {'Hybrid nDCG':<11}"
    print(cat_header)
    print("-" * len(cat_header))

    by_cat = agg.get("by_category", {})
    cat_ndcg_key = f"ndcg_at_{top_k}"
    for cat_name, cat_data in by_cat.items():
        count = cat_data["question_count"]
        bm25_mrr = cat_data["bm25"]["mrr"]
        dense_mrr = cat_data["dense"]["mrr"]
        hybrid_mrr = cat_data["hybrid"]["mrr"]
        hybrid_ndcg = cat_data["hybrid"][cat_ndcg_key]
        print(
            f"{cat_name:<24} | "
            f"{count:<5} | "
            f"{bm25_mrr:<9.4f} | "
            f"{dense_mrr:<10.4f} | "
            f"{hybrid_mrr:<11.4f} | "
            f"{hybrid_ndcg:<11.4f}"
        )

    print("=" * 88 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare BM25, Dense, and Hybrid (RRF) retrieval strategies."
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
        help=f"Path to save comparison JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-K retrieval cutoff (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Latency measurement repeats per question (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"RRF smoothing constant (default: {DEFAULT_RRF_K})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"SentenceTransformers embedding model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embeddings (default: cpu)",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=None,
        help="Optional .npz cache for dense chunk embeddings",
    )

    args = parser.parse_args(argv)

    if not args.chunks.exists():
        print(f"Error: Chunks file not found at {args.chunks}", file=sys.stderr)
        return 1

    if not args.golden_set.exists():
        print(f"Error: Golden set file not found at {args.golden_set}", file=sys.stderr)
        return 1

    print(f"Loading chunks from: {args.chunks}")
    print(f"Loading golden set from: {args.golden_set}")

    with open(args.golden_set, "r", encoding="utf-8") as file:
        golden_set = json.load(file)

    # 1. Initialize BM25 retriever
    print("Initializing BM25 retriever...")
    bm25_retriever = BM25Retriever(args.chunks)

    # 2. Initialize Dense retriever
    print(f"Initializing Dense retriever with model '{args.model_name}' on {args.device}...")
    init_start = time.perf_counter()
    embedder = SentenceTransformerEmbedder(model_name=args.model_name, device=args.device)
    dense_retriever = DenseRetriever(
        chunks_source=args.chunks,
        embedder=embedder,
        model_name=args.model_name,
        device=args.device,
        embeddings_cache=args.embeddings_cache,
    )
    model_init_time_ms = (time.perf_counter() - init_start) * 1000
    print(f"Model initialized in {model_init_time_ms:.2f} ms. (Embedding dim: {dense_retriever.embedding_dim})")

    # 3. Initialize Hybrid retriever
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        rrf_k=args.rrf_k,
    )

    # 4. Run comparison
    print(f"Evaluating {len(golden_set.get('questions', []))} questions (top_k={args.top_k}, repeats={args.repeats})...")
    results = run_hybrid_comparison(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        golden_set=golden_set,
        top_k=args.top_k,
        repeats=args.repeats,
        model_init_time_ms=model_init_time_ms,
        model_name=args.model_name,
    )

    # 5. Print table
    print_comparison_table(results)

    # 6. Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Benchmark results successfully saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
