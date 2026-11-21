
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from datasheet_assistant.retrieval.bm25 import BM25Retriever
from datasheet_assistant.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "pages" / "chunks.json"
GOLDEN_SET_PATH = ROOT / "data" / "golden_set.json"
OUTPUT_DIR = ROOT / "benchmarks" / "baseline"
TOP_K = 10
REPEATS = 20


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


def main():
    with GOLDEN_SET_PATH.open("r", encoding="utf-8") as file:
        golden_set = json.load(file)

    retriever = BM25Retriever(CHUNKS_PATH)

    question_results = []
    latencies_ms = []

    for item in golden_set["questions"]:
        # Warm-up call: excluded from measured latency.
        retriever.search(item["question"], top_k=TOP_K)

        repeat_latencies = []
        results = []

        for _ in range(REPEATS):
            start = time.perf_counter()
            results = retriever.search(item["question"], top_k=TOP_K)
            elapsed_ms = (time.perf_counter() - start) * 1000
            repeat_latencies.append(elapsed_ms)

        latencies_ms.extend(repeat_latencies)

        retrieved_ids = [result["chunk_id"] for result in results]
        relevant_ids = item["relevant_chunk_ids"]

        metrics = {
            "recall_at_10": recall_at_k(
                retrieved_ids, relevant_ids, TOP_K
            ),
            "reciprocal_rank": reciprocal_rank(
                retrieved_ids, relevant_ids
            ),
            "ndcg_at_10": ndcg_at_k(
                retrieved_ids, relevant_ids, TOP_K
            ),
        }

        question_results.append({
            "id": item["id"],
            "question": item["question"],
            "category": item.get("category", "uncategorized"),
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "metrics": metrics,
        })

        print(
            f"{item['id']}: "
            f"Recall@10={metrics['recall_at_10']:.3f}, "
            f"MRR={metrics['reciprocal_rank']:.3f}, "
            f"nDCG@10={metrics['ndcg_at_10']:.3f}"
        )

    count = len(question_results)

    if count == 0:
        raise ValueError("The golden set contains no questions.")

    summary = {
        "question_count": count,
        "top_k": TOP_K,
        "repeats_per_question": REPEATS,
        "mean_recall_at_10": statistics.mean(
            row["metrics"]["recall_at_10"]
            for row in question_results
        ),
        "mean_mrr": statistics.mean(
            row["metrics"]["reciprocal_rank"]
            for row in question_results
        ),
        "mean_ndcg_at_10": statistics.mean(
            row["metrics"]["ndcg_at_10"]
            for row in question_results
        ),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "mean_latency_ms": statistics.mean(latencies_ms),
    }

    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "retriever": "BM25Okapi",
        "chunks_file": "data/pages/chunks.json",
        "golden_set_file": "data/golden_set.json",
        "summary": summary,
        "questions": question_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "bm25_initial.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== BM25 BASELINE SUMMARY ===")
    print(f"Questions:       {count}")
    print(f"Mean Recall@10:  {summary['mean_recall_at_10']:.3f}")
    print(f"Mean MRR:        {summary['mean_mrr']:.3f}")
    print(f"Mean nDCG@10:    {summary['mean_ndcg_at_10']:.3f}")
    print(f"Mean latency:    {summary['mean_latency_ms']:.3f} ms")
    print(f"p95 latency:     {summary['p95_latency_ms']:.3f} ms")
    print(f"Saved report:    {output_path}")


if __name__ == "__main__":
    main()
