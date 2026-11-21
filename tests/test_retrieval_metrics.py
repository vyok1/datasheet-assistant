
import pytest

from datasheet_assistant.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["b", "c"], 2) == 0.5


def test_recall_when_all_relevant_found():
    assert recall_at_k(["a", "b"], ["a", "b"]) == 1.0


def test_reciprocal_rank_first_result():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0


def test_reciprocal_rank_second_result():
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["a", "b"], ["c"]) == 0.0


def test_ndcg_perfect_ranking():
    assert ndcg_at_k(["a", "b"], ["a", "b"]) == pytest.approx(1.0)



def test_ndcg_reversed_ranking():
    # Only "a" is relevant, so placing it second must reduce nDCG.
    assert ndcg_at_k(["b", "a"], ["a"]) < 1.0



def test_empty_relevant_ids():
    assert recall_at_k(["a"], []) == 0.0
    assert reciprocal_rank(["a"], []) == 0.0
    assert ndcg_at_k(["a"], []) == 0.0