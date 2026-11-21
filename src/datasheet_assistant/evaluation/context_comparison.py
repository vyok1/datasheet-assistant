"""Evaluation and comparison functions for context expansion.

Calculates context coverage (context recall), context size (chunks, characters),
and compares baseline retrieval against expanded context configurations.
"""

from typing import Any, Sequence


def context_recall(context_chunk_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of relevant chunk IDs present in the context.

    context recall = |context_chunk_ids ∩ relevant_ids| / |relevant_ids|

    Edge cases:
    - If relevant_ids is empty, returns 0.0 (consistent with retrieval_metrics.recall_at_k).
    - If context_chunk_ids is empty, returns 0.0.
    """
    if not relevant_ids:
        return 0.0

    context_set = set(context_chunk_ids)
    relevant_set = set(relevant_ids)

    return len(context_set & relevant_set) / len(relevant_set)


def context_size(chunks: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Calculate the size of a retrieved or expanded context.

    Returns:
        chunk_count: Total number of chunks in the context.
        total_chars: Sum of length of all chunk text in the context.
    """
    return {
        "chunk_count": len(chunks),
        "total_chars": sum(len(chunk.get("text", "")) for chunk in chunks),
    }
