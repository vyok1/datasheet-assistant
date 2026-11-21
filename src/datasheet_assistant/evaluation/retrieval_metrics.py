
import math


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """Fraction of relevant chunks retrieved in the top k."""
    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Reciprocal rank of the first relevant result."""
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """Normalized discounted cumulative gain for binary relevance."""
    if not relevant_ids:
        return 0.0

    relevant = set(relevant_ids)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0