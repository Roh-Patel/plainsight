import math

import pytest

from eval.metrics import (judged_at_k, mrr_at_k, ndcg_at_k, paper_hit_at_k,
                          paper_of, recall_at_k)


def test_ndcg_hand_computed():
    qrel = {"a": 2, "b": 1}
    got = ndcg_at_k(["x", "a", "b"], qrel, 10)
    dcg = 3 / math.log2(3) + 1 / math.log2(4)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)
    assert got == pytest.approx(dcg / ideal)
    assert got == pytest.approx(0.659, abs=1e-3)


def test_ndcg_perfect_and_order():
    qrel = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], qrel) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], qrel) < 1.0


def test_ndcg_cutoff_and_unjudged():
    qrel = {"a": 2}
    assert ndcg_at_k(["x", "y", "a"], qrel, 2) == 0.0
    assert ndcg_at_k(["x", "y", "a"], qrel, 3) == pytest.approx(1 / math.log2(4) * 1.0 / 1.0 * 3 / 3)


def test_recall_thresholds():
    qrel = {"a": 2, "b": 2, "c": 1, "d": 0}
    ranked = ["a", "x", "c", "b"]
    assert recall_at_k(ranked, qrel, 2) == 0.5
    assert recall_at_k(ranked, qrel, 4) == 1.0
    assert recall_at_k(ranked, qrel, 2, min_grade=1) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, qrel, 3, min_grade=1) == pytest.approx(2 / 3)


def test_mrr():
    assert mrr_at_k(["x", "y", "a"], {"a": 2}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "a"], {"a": 2}, 2) == 0.0
    assert mrr_at_k(["c", "a"], {"a": 2, "c": 1}) == 0.5
    assert mrr_at_k(["c", "a"], {"a": 2, "c": 1}, min_grade=1) == 1.0


def test_paper_hit():
    qrel = {"P1::001::000": 2, "P1::001::001": 0, "P2::000::000": 1}
    assert paper_of("P1::001::000") == "P1"
    assert paper_hit_at_k(["P3::000::000", "P1::004::002"], qrel, 2) == 1.0
    assert paper_hit_at_k(["P3::000::000", "P1::004::002"], qrel, 1) == 0.0
    assert paper_hit_at_k(["P2::000::000"], qrel, 10) == 0.0
    assert paper_hit_at_k(["P2::000::000"], qrel, 10, min_grade=1) == 1.0


def test_judged():
    qrel = {"a": 2, "c": 0}
    assert judged_at_k(["a", "x", "c"], qrel, 3) == pytest.approx(2 / 3)
    assert judged_at_k(["a", "x", "c"], qrel, 5) == pytest.approx(2 / 5)


def test_undefined_raises():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a": 1}, 10)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 0})