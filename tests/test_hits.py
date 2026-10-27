from retrieval.hits import order_hits
from retrieval.hits import order_by_score_stable


def test_sorts_by_score_desc():
    assert order_hits([(1, 0.2), (2, 0.9), (3, 0.5)]) == [(2, 0.9), (3, 0.5), (1, 0.2)]


def test_ties_break_by_lower_id():
    assert order_hits([(9, 0.5), (4, 0.5), (7, 0.5)]) == [(4, 0.5), (7, 0.5), (9, 0.5)]


def test_empty():
    assert order_hits([]) == []


def test_stable_order_keeps_input_order_on_ties():
    hits = [(9, 1.0), (4, 2.0), (7, 2.0), (1, 1.0)]
    assert order_by_score_stable(hits) == [(4, 2.0), (7, 2.0), (9, 1.0), (1, 1.0)]